"""
Detect
1. person (Tracking),
2. object
3. pose (keypoints)

Filter by
1. id
2. min height ratio
3. wrist / pose boundingbox
"""

import time
import cv2
import csv
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from onemetric.cv.utils.iou import box_iou_batch
from supervision.video.sink import VideoSink
from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.tools.detections import Detections, BoxAnnotator
from yolox.tracker.byte_tracker import BYTETracker, STrack


# ----------------- CONSTANTS -----------------
SELECTED_KEYPOINTS = {
    5: "L_Shoulder",
    6: "R_Shoulder",
    7: "L_Elbow",
    8: "R_Elbow",
    9: "L_Wrist",
    10: "R_Wrist",
}
POINT_RADIUS, POINT_THICKNESS, LINE_COLOR, LINE_THICKNESS = 2, -1, (255, 0, 0), 1
BOUNDING_COLOR, BOUNDING_PADDING_RATIO = (0, 0, 255), 0.10
WRIST_OUTER_COLOR, WRIST_OUTER_THICKNESS, WRIST_PADDING_RATIO = (255, 255, 0), 1, 0.50
KEYPOINT_CONFIDENCE_THRESHOLD = 0.4

# --------- SPEED/QUALITY KNOBS ---------
PERSON_CONF_TH = 0.80
OBJECT_CONF_TH = 0.65
DETECT_IMGSZ = 640  # reduce inference size for speed; adjust as needed
POSE_EVERY_N = 2  # compute pose every N frames per track


# ----------------- DATA CLASSES -----------------
@dataclass
class ObjectsInfo:
    object_class: str
    item_onhand: int
    frequency_occurrences: int = 1


@dataclass
class PersonInfo:
    tracker_id: int
    boundingbox: Tuple[int, int, int, int]
    boundingbox_pose: Optional[Tuple[int, int, int, int]]
    boundingbox_wrist_left: Optional[Tuple[int, int, int, int]]
    boundingbox_wrist_right: Optional[Tuple[int, int, int, int]]
    # store absolute frame keypoints (x, y) in pixels for drawing
    keypoints_xy: Optional[np.ndarray] = None
    objects: List[ObjectsInfo] = field(default_factory=list)
    tracker_in_frame: int = 0

    def cleanup_objects(self):
        self.objects = [obj for obj in self.objects if obj.frequency_occurrences > 0]


# ----------------- HELPERS -----------------
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
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
    if len(detections) == 0:
        return detections
    frame_height = frame.shape[0]
    xyxy = detections.xyxy
    heights = xyxy[:, 3] - xyxy[:, 1]
    relative_heights = heights / frame_height
    mask = relative_heights >= ratio
    detections = detections.filter(mask=mask, inplace=True)
    return detections


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
    r = int(POINT_RADIUS + dist * WRIST_PADDING_RATIO)
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


# ----------------- DRAWING / GEOMETRY HELPERS -----------------
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
    for idx in SELECTED_KEYPOINTS.keys():
        if idx < len(keypoints_abs_xy):
            x, y = keypoints_abs_xy[idx]
            if x > 0 and y > 0:
                cv2.circle(
                    frame, (int(x), int(y)), POINT_RADIUS, LINE_COLOR, POINT_THICKNESS
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
                cv2.line(frame, a, b, LINE_COLOR, LINE_THICKNESS)


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


# ----------------- MAIN LOGIC -----------------
def update_person_info(
    detections_person: Detections,
    detections_objects: Detections,
    class_objects_names_dict: dict,
    persons: dict[int, PersonInfo],
    frame: np.ndarray,
):
    for xyxy, conf, cls_id, tid in detections_person:
        if tid is None:
            continue
        bbox = tuple(map(int, xyxy))

        if tid in persons:
            person = persons[tid]
            person.boundingbox = bbox
            person.tracker_in_frame += 1
        else:
            person = PersonInfo(
                tracker_id=tid,
                boundingbox=bbox,
                boundingbox_pose=None,
                boundingbox_wrist_left=None,
                boundingbox_wrist_right=None,
                objects=[],
                tracker_in_frame=1,
            )
            persons[tid] = person

        # ---------- POSE ----------
        x1, y1, x2, y2 = bbox
        crop_img = frame[y1:y2, x1:x2]
        do_pose_this_frame = (person.tracker_in_frame % POSE_EVERY_N == 0) or (
            person.keypoints_xy is None
        )
        if crop_img.size > 0 and do_pose_this_frame:
            pose_result = model_pose(
                crop_img, conf=0.5, imgsz=DETECT_IMGSZ, verbose=False
            )
            if len(pose_result[0].keypoints) > 0:
                keypoints = pose_result[0].keypoints.xy[0].cpu().numpy()
                # convert to absolute frame coordinates
                keypoints_abs = keypoints + np.array([x1, y1], dtype=keypoints.dtype)
                pose_box, wrist_left, wrist_right = extract_pose_info(keypoints)
                # translate boxes to absolute
                person.boundingbox_pose = translate_box(pose_box, x1, y1)
                person.boundingbox_wrist_left = translate_box(wrist_left, x1, y1)
                person.boundingbox_wrist_right = translate_box(wrist_right, x1, y1)
                person.keypoints_xy = keypoints_abs

        # ---------- OBJECTS ----------
        for obj_xyxy, obj_conf, obj_cls in zip(
            detections_objects.xyxy,
            detections_objects.confidence,
            detections_objects.class_id,
        ):
            obj_name = class_objects_names_dict[obj_cls]
            cx, cy = box_center(obj_xyxy)

            in_left = point_in_box((cx, cy), person.boundingbox_wrist_left)
            in_right = point_in_box((cx, cy), person.boundingbox_wrist_right)
            in_pose = False
            if not (in_left or in_right):
                in_pose = point_in_box((cx, cy), person.boundingbox_pose)

            if in_left or in_right or in_pose:
                matched_obj = next(
                    (o for o in person.objects if o.object_class == obj_name), None
                )
                if matched_obj:
                    matched_obj.frequency_occurrences += 1
                else:
                    new_obj = ObjectsInfo(
                        object_class=obj_name,
                        item_onhand=1,
                        frequency_occurrences=1,
                    )
                    person.objects.append(new_obj)
        person.cleanup_objects()


def video_process(
    _video_input_path: Path,
    _video_output_path: Path,
    _class_person_names_dict: dict,
    _class_objects_names_dict: dict,
    _filter_class_person: List[int],
    _filter_class_object: List[int],
    persons: dict[int, PersonInfo],
):
    video_info = VideoInfo.from_video_path(_video_input_path)
    generator = get_video_frames_generator(_video_input_path)
    with VideoSink(_video_output_path, video_info) as sink:
        for frame in tqdm(generator, total=video_info.total_frames):

            # Single detection pass, filter by classes
            detected_all = model_person(
                frame,
                conf=OBJECT_CONF_TH,
                classes=None,
                imgsz=DETECT_IMGSZ,
                verbose=False,
            )
            boxes = detected_all[0].boxes
            all_xyxy = boxes.xyxy.cpu().numpy()
            all_conf = boxes.conf.cpu().numpy()
            all_cls = boxes.cls.cpu().numpy().astype(int)

            person_mask = np.isin(all_cls, _filter_class_person) & (
                all_conf >= PERSON_CONF_TH
            )
            object_mask = np.isin(all_cls, _filter_class_object) & (
                all_conf >= OBJECT_CONF_TH
            )

            detections_person = Detections(
                xyxy=all_xyxy[person_mask],
                confidence=all_conf[person_mask],
                class_id=all_cls[person_mask],
            )
            detections_person = filter_min_height(detections_person, frame, ratio=0.4)

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
                if p.keypoints_xy is not None:
                    draw_keypoints_and_arms(frame, p.keypoints_xy)

                # pose box with padding
                if p.boundingbox_pose is not None:
                    pb = pad_box(p.boundingbox_pose, BOUNDING_PADDING_RATIO)
                    pb = clip_box_to_frame(pb, frame.shape)
                    draw_rect(frame, pb, BOUNDING_COLOR, 1)

                # wrist left/right with padding
                if p.boundingbox_wrist_left is not None:
                    wl = pad_box(p.boundingbox_wrist_left, BOUNDING_PADDING_RATIO)
                    wl = clip_box_to_frame(wl, frame.shape)
                    draw_rect(frame, wl, WRIST_OUTER_COLOR, WRIST_OUTER_THICKNESS)
                if p.boundingbox_wrist_right is not None:
                    wr = pad_box(p.boundingbox_wrist_right, BOUNDING_PADDING_RATIO)
                    wr = clip_box_to_frame(wr, frame.shape)
                    draw_rect(frame, wr, WRIST_OUTER_COLOR, WRIST_OUTER_THICKNESS)
            sink.write_frame(frame)


def save_results_to_csv(
    persons: dict[int, PersonInfo],
    class_objects_names_dict: dict,
    video_name: str,
    frame_interest: int,
):
    output_path = Path(f"./results/text/{video_name}{take}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = list(class_objects_names_dict.values())
    header = (
        ["tracker_id", "len_objects", "tracker_in_frame"]
        + [f"count_{c}" for c in class_names]
        # + ["objects", "pose_box", "wrist_left", "wrist_right"]
    )

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for tid, person in persons.items():
            if person.tracker_in_frame < frame_interest:
                continue

            objects_list = [
                f"{o.object_class}:{o.item_onhand}(f={o.frequency_occurrences})"
                for o in person.objects
            ]

            class_counts = {c: 0 for c in class_names}
            for o in person.objects:
                class_counts[o.object_class] += 1

            row = (
                [
                    tid,
                    len(person.objects),
                    person.tracker_in_frame,
                ]
                + [class_counts[c] for c in class_names]
                + [
                    objects_list,
                    person.boundingbox_pose,
                    person.boundingbox_wrist_left,
                    person.boundingbox_wrist_right,
                ]
            )
            writer.writerow(row)

    print(f"[INFO] Results saved → {output_path}")


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
    except Exception:
        pass
    return (
        model_person,
        model_object,
        model_pose,
        class_person_names_dict,
        class_objects_names_dict,
    )


# ----------------- MAIN -----------------
if __name__ == "__main__":
    start_time = time.time()
    persons: dict[int, PersonInfo] = {}

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

    filter_class_person = [4]
    filter_class_object = [0, 1, 2, 3]
    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1
    )
    video_name = "test"
    take = ""
    video_input_path = Path(f"./videos/{video_name}.avi")
    video_output_path = Path(f"./results/videos/{video_name}{take}.avi")

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
        persons, class_objects_names_dict, video_name, frame_interest=350
    )
