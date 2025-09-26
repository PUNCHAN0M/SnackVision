"""
ByteTrack Multi-Object Detection and Pose Estimation System

This system performs:
1. Person tracking with ByteTracker
2. Object detection and classification
3. Human pose estimation with keypoint detection
4. Spatial relationship analysis between persons and objects

Key Features:
- Optimized for real-time processing with GPU acceleration
- Spatial filtering based on pose regions (wrist/body areas)
- Configurable detection thresholds and processing intervals
- Comprehensive CSV output with tracking statistics

Author: Senior Developer
Version: 2.0 (Optimized)
"""

# Standard library imports
import csv
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party imports
import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from tqdm import tqdm
from ultralytics import YOLO

# Computer vision and tracking imports
from onemetric.cv.utils.iou import box_iou_batch
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.video.source import get_video_frames_generator
from yolox.tracker.byte_tracker import BYTETracker, STrack


# ================================================================================================
# CONFIGURATION CONSTANTS
# ================================================================================================

# Pose keypoint indices (COCO format)
POSE_KEYPOINTS = {
    5: "L_Shoulder",
    6: "R_Shoulder",
    7: "L_Elbow",
    8: "R_Elbow",
    9: "L_Wrist",
    10: "R_Wrist",
}


# Visual rendering settings
class RenderConfig:
    KEYPOINT_RADIUS = 2
    KEYPOINT_THICKNESS = -1  # Filled circle
    SKELETON_COLOR = (255, 0, 0)  # Blue for keypoints/skeleton
    SKELETON_THICKNESS = 1

    POSE_BOX_COLOR = (0, 0, 255)  # Red for pose bounding box
    WRIST_BOX_COLOR = (255, 255, 0)  # Yellow for wrist regions
    WRIST_BOX_THICKNESS = 1

    BOUNDING_BOX_PADDING = 0.10  # 10% padding for visualization
    WRIST_REGION_PADDING = 0.50  # 50% padding for wrist detection area


# Detection and tracking thresholds
class ProcessingConfig:
    PERSON_CONFIDENCE_THRESHOLD = 0.80
    OBJECT_CONFIDENCE_THRESHOLD = 0.65
    KEYPOINT_CONFIDENCE_THRESHOLD = 0.4
    MINIMUM_HEIGHT_RATIO = 0.4  # Person must be at least 40% of frame height

    # Object filtering settings
    MINIMUM_DETECTION_FREQUENCY = 15  # Filter out objects with frequency < this value

    # Performance optimization settings
    DETECTION_IMAGE_SIZE = 640  # Reduced size for faster inference
    POSE_COMPUTATION_INTERVAL = 2  # Compute pose every N frames per track


# Class filtering (adjust based on your model's class indices)
class ClassConfig:
    PERSON_CLASSES = [4]  # Person class ID
    OBJECT_CLASSES = [0, 1, 2, 3]  # Object class IDs to track


# ================================================================================================
# DATA MODELS
# ================================================================================================


@dataclass
class DetectedObject:
    """Represents a detected object with its classification and frequency."""

    object_class: str
    item_count: int
    detection_frequency: int = 1

    def increment_frequency(self) -> None:
        """Increment the detection frequency counter."""
        self.detection_frequency += 1


@dataclass
class TrackedPerson:
    """
    Comprehensive data structure for a tracked person including:
    - Basic tracking info (ID, bounding box, frame count)
    - Pose estimation data (keypoints, pose regions)
    - Associated objects within pose/wrist regions
    """

    tracker_id: int
    bounding_box: Tuple[int, int, int, int]

    # Pose estimation data
    pose_bounding_box: Optional[Tuple[int, int, int, int]] = None
    left_wrist_region: Optional[Tuple[int, int, int, int]] = None
    right_wrist_region: Optional[Tuple[int, int, int, int]] = None
    keypoints_absolute: Optional[np.ndarray] = None

    # Tracking and object association
    detected_objects: List[DetectedObject] = field(default_factory=list)
    frames_tracked: int = 0

    def add_detected_object(self, object_class: str) -> None:
        """Add or update detected object frequency."""
        existing_obj = next(
            (obj for obj in self.detected_objects if obj.object_class == object_class),
            None,
        )
        if existing_obj:
            existing_obj.increment_frequency()
        else:
            new_obj = DetectedObject(
                object_class=object_class, item_count=1, detection_frequency=1
            )
            self.detected_objects.append(new_obj)

    def cleanup_objects(self) -> None:
        """Remove objects with zero detection frequency."""
        self.detected_objects = [
            obj for obj in self.detected_objects if obj.detection_frequency > 0
        ]

    def get_filtered_objects(self, min_frequency: int = None) -> List[DetectedObject]:
        """Get objects that meet the minimum frequency threshold."""
        if min_frequency is None:
            min_frequency = ProcessingConfig.MINIMUM_DETECTION_FREQUENCY
        return [
            obj
            for obj in self.detected_objects
            if obj.detection_frequency >= min_frequency
        ]


# ================================================================================================
# TRACKING & DETECTION UTILITIES
# ================================================================================================


def detections2boxes(detections: Detections) -> np.ndarray:
    """Convert detection objects to bounding box format for tracker."""
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    """Extract bounding boxes from tracking results."""
    return np.array([track.tlbr for track in tracks], dtype=float)


def match_detections_with_tracks(
    detections: Detections, tracks: List[STrack]
) -> List[Optional[int]]:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    tracker_ids = [None] * len(detections)
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id
    return tracker_ids


def filter_by_tracker_id(detections: Detections) -> Detections:
    mask = np.array([tid is not None for tid in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    return detections


def filter_min_height(
    detections: Detections, frame: np.ndarray, ratio: float = None
) -> Detections:
    """Filter detections by minimum height ratio relative to frame."""
    if len(detections) == 0:
        return detections
    frame_height = frame.shape[0]
    xyxy = detections.xyxy
    heights = xyxy[:, 3] - xyxy[:, 1]
    relative_heights = heights / frame_height
    mask = relative_heights >= ratio
    detections = detections.filter(mask=mask, inplace=True)
    return detections


# ================================================================================================
# POSE ESTIMATION UTILITIES
# ================================================================================================


def compute_pose_bounding_box(single_keypoints_xy: np.ndarray) -> Optional[tuple]:
    if single_keypoints_xy is None or single_keypoints_xy.shape[0] == 0:
        return None
    xs = single_keypoints_xy[:, 0]
    ys = single_keypoints_xy[:, 1]
    valid_mask = (xs > 0) & (ys > 0)
    if not np.any(valid_mask):
        return None
    xs = xs[valid_mask]
    ys = ys[valid_mask]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def compute_single_wrist_region(
    elbow, wrist
) -> Optional[Tuple[float, float, float, float]]:
    if elbow is None or wrist is None:
        return None
    ex, ey = elbow
    wx, wy = wrist
    dist = ((wx - ex) ** 2 + (wy - ey) ** 2) ** 0.5
    if dist <= 0:
        return None
    r = int(RenderConfig.KEYPOINT_RADIUS + dist * RenderConfig.WRIST_REGION_PADDING)
    return (wx - r, wy - r, wx + r, wy + r)


def extract_pose_info(single_keypoints_xy: np.ndarray):
    if single_keypoints_xy is None or single_keypoints_xy.shape[0] == 0:
        return None, None, None
    pose_box = compute_pose_bounding_box(single_keypoints_xy)

    def get_point(idx):
        if idx >= single_keypoints_xy.shape[0]:
            return None
        x, y = single_keypoints_xy[idx]
        if x <= 0 or y <= 0:
            return None
        return (float(x), float(y))

    left_elbow = get_point(7)
    left_wrist = get_point(9)
    right_elbow = get_point(8)
    right_wrist = get_point(10)

    wrist_left_box = compute_single_wrist_region(left_elbow, left_wrist)
    wrist_right_box = compute_single_wrist_region(right_elbow, right_wrist)

    return pose_box, wrist_left_box, wrist_right_box


# ================================================================================================
# GEOMETRY & VISUALIZATION UTILITIES
# ================================================================================================
def pad_box(
    box: Tuple[float, float, float, float], pad_ratio: float
) -> Tuple[int, int, int, int]:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    px = w * pad_ratio
    py = h * pad_ratio
    return (int(x1 - px), int(y1 - py), int(x2 + px), int(y2 + py))


def translate_box(
    box: Tuple[float, float, float, float], dx: float, dy: float
) -> Tuple[float, float, float, float]:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def clip_box_to_frame(
    box: Tuple[int, int, int, int], frame_shape
) -> Tuple[int, int, int, int]:
    if box is None:
        return None
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return (x1, y1, x2, y2)


def draw_rect(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
):
    if box is None:
        return
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_keypoints_and_arms(frame: np.ndarray, keypoints_abs_xy: np.ndarray):
    if keypoints_abs_xy is None or keypoints_abs_xy.size == 0:
        return
    # draw selected keypoints
    for idx in POSE_KEYPOINTS.keys():
        if idx < len(keypoints_abs_xy):
            x, y = keypoints_abs_xy[idx]
            if x > 0 and y > 0:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    RenderConfig.KEYPOINT_RADIUS,
                    RenderConfig.SKELETON_COLOR,
                    RenderConfig.KEYPOINT_THICKNESS,
                )

    # draw simple arm lines: shoulder->elbow->wrist
    def safe_pt(i):
        if i < len(keypoints_abs_xy):
            x, y = keypoints_abs_xy[i]
            if x > 0 and y > 0:
                return int(x), int(y)
        return None

    left_pts = [safe_pt(5), safe_pt(7), safe_pt(9)]
    right_pts = [safe_pt(6), safe_pt(8), safe_pt(10)]
    for triplet in (left_pts, right_pts):
        for a, b in zip(triplet, triplet[1:]):
            if a is not None and b is not None:
                cv2.line(
                    frame,
                    a,
                    b,
                    RenderConfig.SKELETON_COLOR,
                    RenderConfig.SKELETON_THICKNESS,
                )


def box_center(
    xyxy: np.ndarray | Tuple[float, float, float, float],
) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def point_in_box(
    pt: Tuple[float, float],
    box: Tuple[float, float, float, float] | Tuple[int, int, int, int],
) -> bool:
    if box is None or pt is None:
        return False
    x, y = pt
    x1, y1, x2, y2 = box
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)


# ================================================================================================
# CORE PROCESSING PIPELINE
# ================================================================================================
def update_person_info(
    detections_person: Detections,
    detections_objects: Detections,
    class_objects_names_dict: dict,
    persons: dict[int, TrackedPerson],
    frame: np.ndarray,
):
    for xyxy, conf, cls_id, tid in detections_person:
        if tid is None:
            continue
        bbox = tuple(map(int, xyxy))

        if tid in persons:
            person = persons[tid]
            person.bounding_box = bbox
            person.frames_tracked += 1
        else:
            person = TrackedPerson(
                tracker_id=tid,
                bounding_box=bbox,
                pose_bounding_box=None,
                left_wrist_region=None,
                right_wrist_region=None,
                detected_objects=[],
                frames_tracked=1,
            )
            persons[tid] = person

        # ---------- POSE ----------
        x1, y1, x2, y2 = bbox
        crop_img = frame[y1:y2, x1:x2]
        do_pose_this_frame = (
            person.frames_tracked % ProcessingConfig.POSE_COMPUTATION_INTERVAL == 0
        ) or (person.keypoints_absolute is None)
        if crop_img.size > 0 and do_pose_this_frame:
            pose_result = model_pose(
                crop_img,
                conf=0.5,
                imgsz=ProcessingConfig.DETECTION_IMAGE_SIZE,
                verbose=False,
            )
            if len(pose_result[0].keypoints) > 0:
                keypoints = pose_result[0].keypoints.xy[0].cpu().numpy()
                # convert to absolute frame coordinates
                keypoints_abs = keypoints + np.array([x1, y1], dtype=keypoints.dtype)
                pose_box, wrist_left, wrist_right = extract_pose_info(keypoints)
                # translate boxes to absolute
                person.pose_bounding_box = translate_box(pose_box, x1, y1)
                person.left_wrist_region = translate_box(wrist_left, x1, y1)
                person.right_wrist_region = translate_box(wrist_right, x1, y1)
                person.keypoints_absolute = keypoints_abs

        # ---------- OBJECTS ----------
        for obj_xyxy, obj_conf, obj_cls in zip(
            detections_objects.xyxy,
            detections_objects.confidence,
            detections_objects.class_id,
        ):
            obj_name = class_objects_names_dict[obj_cls]
            cx, cy = box_center(obj_xyxy)

            in_left_wrist = point_in_box((cx, cy), person.left_wrist_region)
            in_right_wrist = point_in_box((cx, cy), person.right_wrist_region)
            in_pose_region = False

            if not (in_left_wrist or in_right_wrist):
                in_pose_region = point_in_box((cx, cy), person.pose_bounding_box)

            if in_left_wrist or in_right_wrist or in_pose_region:
                person.add_detected_object(obj_name)

        person.cleanup_objects()


def video_process(
    _video_input_path: Path,
    _video_output_path: Path,
    _class_person_names_dict: dict,
    _class_objects_names_dict: dict,
    _filter_class_person: List[int],
    _filter_class_object: List[int],
    persons: dict[int, TrackedPerson],
):
    video_info = VideoInfo.from_video_path(_video_input_path)
    generator = get_video_frames_generator(_video_input_path)
    with VideoSink(_video_output_path, video_info) as sink:
        for frame in tqdm(generator, total=video_info.total_frames):

            # Single detection pass, filter by classes
            detected_all = model_person(
                frame,
                conf=ProcessingConfig.OBJECT_CONFIDENCE_THRESHOLD,
                classes=None,
                imgsz=ProcessingConfig.DETECTION_IMAGE_SIZE,
                verbose=False,
            )
            boxes = detected_all[0].boxes
            all_xyxy = boxes.xyxy.cpu().numpy()
            all_conf = boxes.conf.cpu().numpy()
            all_cls = boxes.cls.cpu().numpy().astype(int)

            person_mask = np.isin(all_cls, _filter_class_person) & (
                all_conf >= ProcessingConfig.PERSON_CONFIDENCE_THRESHOLD
            )
            object_mask = np.isin(all_cls, _filter_class_object) & (
                all_conf >= ProcessingConfig.OBJECT_CONFIDENCE_THRESHOLD
            )

            detections_person = Detections(
                xyxy=all_xyxy[person_mask],
                confidence=all_conf[person_mask],
                class_id=all_cls[person_mask],
            )
            detections_person = filter_min_height(
                detections_person, frame, ratio=ProcessingConfig.MINIMUM_HEIGHT_RATIO
            )

            detections_objects = Detections(
                xyxy=all_xyxy[object_mask],
                confidence=all_conf[object_mask],
                class_id=all_cls[object_mask],
            )

            # ___________FILTER PERSON_____________
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections_person),
                img_info=frame.shape,
                img_size=frame.shape,
            )
            tracker_id = match_detections_with_tracks(
                detections=detections_person, tracks=tracks
            )
            detections_person.tracker_id = np.array(tracker_id)
            detections_person = filter_by_tracker_id(detections_person)

            update_person_info(
                detections_person,
                detections_objects,
                _class_objects_names_dict,
                persons,
                frame,
            )
            # __________END FILTER___________

            # __________DRAW BOX___________
            labels_person = [
                f"#{tracker_id} {_class_person_names_dict[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections_person
            ]
            frame = box_annotator.annotate(
                frame=frame, detections=detections_person, labels=labels_person
            )

            labels_objects = [
                f"{_class_objects_names_dict[cid]} {conf:0.2f}"
                for cid, conf in zip(
                    detections_objects.class_id, detections_objects.confidence
                )
            ]
            frame = box_annotator.annotate(
                frame=frame, detections=detections_objects, labels=labels_objects
            )

            # Draw pose keypoints and padded bounding boxes for wrists and pose
            for _, _, _, tid in detections_person:
                if tid is None:
                    continue
                p = persons.get(tid)
                if p is None:
                    continue
                # keypoints
                if p.keypoints_absolute is not None:
                    draw_keypoints_and_arms(frame, p.keypoints_absolute)

                # pose box with padding
                if p.pose_bounding_box is not None:
                    pb = pad_box(p.pose_bounding_box, RenderConfig.BOUNDING_BOX_PADDING)
                    pb = clip_box_to_frame(pb, frame.shape)
                    draw_rect(frame, pb, RenderConfig.POSE_BOX_COLOR, 1)

                # wrist left/right with padding
                if p.left_wrist_region is not None:
                    wl = pad_box(p.left_wrist_region, RenderConfig.BOUNDING_BOX_PADDING)
                    wl = clip_box_to_frame(wl, frame.shape)
                    draw_rect(
                        frame,
                        wl,
                        RenderConfig.WRIST_BOX_COLOR,
                        RenderConfig.WRIST_BOX_THICKNESS,
                    )
                if p.right_wrist_region is not None:
                    wr = pad_box(
                        p.right_wrist_region, RenderConfig.BOUNDING_BOX_PADDING
                    )
                    wr = clip_box_to_frame(wr, frame.shape)
                    draw_rect(
                        frame,
                        wr,
                        RenderConfig.WRIST_BOX_COLOR,
                        RenderConfig.WRIST_BOX_THICKNESS,
                    )
            sink.write_frame(frame)


def save_results_to_csv(
    persons: dict[int, TrackedPerson],
    class_objects_names_dict: dict,
    video_name: str,
    frame_interest: int,
    min_frequency: int = None,
):
    """Save tracking results to CSV with frequency filtering.

    Args:
        persons: Dictionary of tracked persons
        class_objects_names_dict: Object class names mapping
        video_name: Name of the processed video
        frame_interest: Minimum frames required to include a person
        min_frequency: Minimum detection frequency to include objects (default: ProcessingConfig.MINIMUM_DETECTION_FREQUENCY)
    """
    if min_frequency is None:
        min_frequency = ProcessingConfig.MINIMUM_DETECTION_FREQUENCY

    output_path = Path(f"./results/text/{video_name}{take}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = list(class_objects_names_dict.values())
    # CSV template: tracker_id + [class_counts[c] for c in class_names]
    header = ["tracker_id"] + [f"count_{c}" for c in class_names]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for tid, person in persons.items():
            if person.frames_tracked < frame_interest:
                continue

            # Get objects that meet frequency threshold
            filtered_objects = person.get_filtered_objects(min_frequency)

            # Count objects by class (only those meeting frequency threshold)
            class_counts = {c: 0 for c in class_names}
            for obj in filtered_objects:
                if obj.object_class in class_counts:
                    class_counts[obj.object_class] += 1

            # CSV row: tracker_id + counts for each class
            row = [tid] + [class_counts[c] for c in class_names]
            writer.writerow(row)

    print(f"[INFO] Results saved → {output_path}")


# ================================================================================================
# MODEL MANAGEMENT & INITIALIZATION
# ================================================================================================


def model_initialization(
    model_person_path: Optional[str] = None,
    model_object_path: Optional[str] = None,
    model_pose_path: Optional[str] = None,
):
    model_person = YOLO(model_person_path)
    model_object = YOLO(model_object_path)
    model_pose = YOLO(model_pose_path)
    class_person_names_dict = model_person.model.names
    class_objects_names_dict = model_object.model.names
    model_person.eval()
    model_object.eval()
    model_pose.eval()
    model_person.fuse()
    model_object.fuse()
    model_pose.fuse()
    # Use GPU if available and enable FP16 for speed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model_person.to(device)
        model_object.to(device)
        model_pose.to(device)
        if device == "cuda":
            model_person.model.half()
            model_object.model.half()
            model_pose.model.half()
            print("[INFO] Models initialized and moved to GPU.")
    except Exception:
        pass
    return (
        model_person,
        model_object,
        model_pose,
        class_person_names_dict,
        class_objects_names_dict,
    )


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================
if __name__ == "__main__":
    start_time = time.time()
    persons: dict[int, TrackedPerson] = {}

    # =========== CONFIGURABLE PARAMETERS ===========
    # Adjust frequency filter threshold here (objects with detection_frequency < this value will be excluded from CSV)
    # ProcessingConfig.MINIMUM_DETECTION_FREQUENCY = 10  # Uncomment and modify as needed

    @dataclass(frozen=True)
    class BYTETrackerArgs:
        track_thresh: float = 0.8
        track_buffer: int = 50
        match_thresh: float = 0.8
        aspect_ratio_thresh: float = None
        min_box_area: float = 1.0
        mot20: bool = False

    byte_tracker = BYTETracker(BYTETrackerArgs())

    (
        model_person,
        model_object,
        model_pose,
        class_person_names_dict,
        class_objects_names_dict,
    ) = model_initialization(
        model_person_path="./models/best_done.pt",
        model_object_path="./models/best_done.pt",
        model_pose_path="./models/yolo11s-pose.pt",
    )

    filter_class_person = ClassConfig.PERSON_CLASSES
    filter_class_object = ClassConfig.OBJECT_CLASSES
    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1
    )
    video_name = "1"
    take = ""
    video_input_path = Path(f"./videos/{video_name}.avi")
    video_output_path = Path(f"./result_test/videos/{video_name}{take}.avi")

    video_process(
        _video_input_path=video_input_path,
        _video_output_path=video_output_path,
        _class_person_names_dict=class_person_names_dict,
        _class_objects_names_dict=class_objects_names_dict,
        _filter_class_person=filter_class_person,
        _filter_class_object=filter_class_object,
        persons=persons,
    )
    print(f"[INFO] Processing time: {time.time() - start_time:.2f} seconds")
    save_results_to_csv(
        persons=persons,
        class_objects_names_dict=class_objects_names_dict,
        video_name=video_name,
        frame_interest=350,
        min_frequency=ProcessingConfig.MINIMUM_DETECTION_FREQUENCY,
    )
