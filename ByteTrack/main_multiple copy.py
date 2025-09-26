"""
Detcet
1.prerson (Tracking),
2.object

Filter by
1.id
2.min height ratio

"""

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence
from onemetric.cv.utils.iou import box_iou_batch
from supervision.video.sink import VideoSink
from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.tools.detections import Detections, BoxAnnotator
from yolox.tracker.byte_tracker import BYTETracker, STrack


def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, tracks: List[STrack]
) -> Detections:
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
    detections: Detections, frame: np.ndarray, ratio: float = 0.3
) -> None:
    if len(detections) == 0:
        return
    frame_height = frame.shape[0]
    xyxy = detections.xyxy
    heights = xyxy[:, 3] - xyxy[:, 1]
    # Keep detections where heights/frame_height >= ratio
    relative_heights = heights / frame_height
    mask = relative_heights >= ratio
    detections.filter(mask=mask, inplace=True)


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


def draw_pose_keypoints(
    frame: np.ndarray,
    detections_person: Detections,
    keypoints_xy: np.ndarray,
    keypoints_conf: Optional[np.ndarray] = None,
) -> None:
    """Draw selected pose keypoints and arm connections with padding per tracked person.

    Assumes keypoints_xy is already filtered to align index-wise with detections_person after tracking id filtering.
    """
    if keypoints_xy is None:
        return
    frame_h, frame_w = frame.shape[:2]
    num_persons = keypoints_xy.shape[0]
    for person_idx in range(num_persons):
        tracker_id = detections_person.tracker_id[person_idx]
        valid_points = []  # list of (kp_index, x, y)
        for kp_index in SELECTED_KEYPOINTS.keys():
            x, y = keypoints_xy[person_idx, kp_index]
            conf = 1.0
            if keypoints_conf is not None:
                conf = keypoints_conf[person_idx, kp_index]
            if conf < KEYPOINT_CONFIDENCE_THRESHOLD or x <= 0 or y <= 0:
                continue
            valid_points.append((kp_index, float(x), float(y)))
            cv2.circle(
                frame,
                (int(x), int(y)),
                POINT_RADIUS,
                (0, 255, 0),
                POINT_THICKNESS,
            )
            cv2.putText(
                frame,
                f"{SELECTED_KEYPOINTS[kp_index]}#{tracker_id}",
                (int(x) + 3, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # connection helper
        def find_point(idx: int):
            for p in valid_points:
                if p[0] == idx:
                    return (int(p[1]), int(p[2]))
            return None

        left_chain = [5, 7, 9]
        right_chain = [6, 8, 10]
        connectors = (
            [(5, 6)]
            + list(zip(left_chain, left_chain[1:]))
            + list(zip(right_chain, right_chain[1:]))
        )
        for a_idx, b_idx in connectors:
            pa, pb = find_point(a_idx), find_point(b_idx)
            if pa and pb:
                cv2.line(frame, pa, pb, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

        # wrist padding circles
        def padded_wrist(elbow_idx: int, wrist_idx: int):
            elbow_pt = find_point(elbow_idx)
            wrist_pt = find_point(wrist_idx)
            if not elbow_pt or not wrist_pt:
                return
            ex, ey = elbow_pt
            wx, wy = wrist_pt
            dist = ((wx - ex) ** 2 + (wy - ey) ** 2) ** 0.5
            if dist <= 0:
                return
            radius = int(POINT_RADIUS + dist * WRIST_PADDING_RATIO)
            cv2.circle(
                frame,
                (wx, wy),
                radius,
                WRIST_OUTER_COLOR,
                WRIST_OUTER_THICKNESS,
                cv2.LINE_AA,
            )

        padded_wrist(7, 9)  # left
        padded_wrist(8, 10)  # right
        # bounding box around arms with 10% padding
        arm_points = [p for p in valid_points if 5 <= p[0] <= 10]
        if len(arm_points) >= 2:
            xs = [p[1] for p in arm_points]
            ys = [p[2] for p in arm_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w = max_x - min_x
            h = max_y - min_y
            if w > 0 and h > 0:
                pad_x = w * BOUNDING_PADDING_RATIO
                pad_y = h * BOUNDING_PADDING_RATIO
                x1 = max(0, int(min_x - pad_x))
                y1 = max(0, int(min_y - pad_y))
                x2 = min(frame_w - 1, int(max_x + pad_x))
                y2 = min(frame_h - 1, int(max_y + pad_y))
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOUNDING_COLOR, 1)
                cv2.putText(
                    frame,
                    f"Arms#{tracker_id}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    BOUNDING_COLOR,
                    1,
                    cv2.LINE_AA,
                )


def video_process(
    _video_input_path: Path,
    _video_output_path: Path,
    _class_person_names_dict: dict,
    _class_objects_names_dict: dict,
    _filter_class_person: List[int],
    _filter_class_object: List[int],
):
    video_info = VideoInfo.from_video_path(_video_input_path)
    generator = get_video_frames_generator(_video_input_path)
    with VideoSink(_video_output_path, video_info) as sink:
        for frame in tqdm(generator, total=video_info.total_frames):

            detected_person = model_person(
                frame, conf=0.8, classes=_filter_class_person, verbose=False
            )
            detected_objects = model_object(
                frame, conf=0.7, classes=_filter_class_object, verbose=False
            )
            detections_person = Detections(
                xyxy=detected_person[0].boxes.xyxy.cpu().numpy(),
                confidence=detected_person[0].boxes.conf.cpu().numpy(),
                class_id=detected_person[0].boxes.cls.cpu().numpy().astype(int),
            )
            # Pose keypoints raw (before filtering)
            pose_keypoints_xy = None
            pose_keypoints_conf = None
            if (
                hasattr(detected_person[0], "keypoints")
                and detected_person[0].keypoints is not None
            ):
                pose_keypoints_xy = (
                    detected_person[0].keypoints.xy.cpu().numpy()
                )  # (n,17,2)
                if getattr(detected_person[0].keypoints, "conf", None) is not None:
                    pose_keypoints_conf = (
                        detected_person[0].keypoints.conf.cpu().numpy()
                    )  # (n,17)
            # Apply minimum height ratio filter only for person detections
            # filter_min_height(detections_person, frame, ratio=0.5)
            detections_objects = Detections(
                xyxy=detected_objects[0].boxes.xyxy.cpu().numpy(),
                confidence=detected_objects[0].boxes.conf.cpu().numpy(),
                class_id=detected_objects[0].boxes.cls.cpu().numpy().astype(int),
            )
            # ___________FILTER_____________
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections_person),
                img_info=frame.shape,
                img_size=frame.shape,
            )
            tracker_id = match_detections_with_tracks(
                detections=detections_person, tracks=tracks
            )
            detections_person.tracker_id = np.array(tracker_id)
            # Build mask to filter keypoints in sync with detections
            mask_person_valid = np.array(
                [tid is not None for tid in detections_person.tracker_id],
                dtype=bool,
            )
            detections_person = filter_by_tracker_id(detections_person)
            filtered_keypoints_xy = None
            filtered_keypoints_conf = None
            if pose_keypoints_xy is not None:
                filtered_keypoints_xy = pose_keypoints_xy[mask_person_valid]
                if pose_keypoints_conf is not None:
                    filtered_keypoints_conf = pose_keypoints_conf[mask_person_valid]
            # __________END FILTER___________

            # __________CREATE LABEL AND DRAW BOX___________
            labels_person = [
                f"#{tracker_id} {_class_person_names_dict[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections_person
            ]
            frame = box_annotator.annotate(
                frame=frame, detections=detections_person, labels=labels_person
            )
            # Draw pose (after boxes so lines appear over boxes or adjust order if preferred)
            if (
                "filtered_keypoints_xy" in locals()
                and filtered_keypoints_xy is not None
            ):
                draw_pose_keypoints(
                    frame=frame,
                    detections_person=detections_person,
                    keypoints_xy=filtered_keypoints_xy,
                    keypoints_conf=filtered_keypoints_conf,
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
            sink.write_frame(frame)
            # __________END FOR___________


def model_initialization(
    model_person_path: Optional[str] = None, model_object_path: Optional[str] = None
):
    model_person = YOLO(model_person_path)
    model_object = YOLO(model_object_path)
    class_person_names_dict = model_person.model.names
    class_objects_names_dict = model_object.model.names
    model_person.eval()
    model_object.eval()
    model_person.fuse()
    model_object.fuse()
    return (
        model_person,
        model_object,
        class_person_names_dict,
        class_objects_names_dict,
    )


if __name__ == "__main__":

    @dataclass(frozen=True)
    class BYTETrackerArgs:
        track_thresh: float = 0.8
        track_buffer: int = 200
        match_thresh: float = 0.8
        aspect_ratio_thresh: float = 3.0
        min_box_area: float = 1.0
        mot20: bool = False

    byte_tracker = BYTETracker(BYTETrackerArgs())

    model_person, model_object, class_person_names_dict, class_objects_names_dict = (
        model_initialization(
            model_person_path="./models/yolo11l-pose.pt",
            model_object_path="./models/best_17Sep.pt",
        )
    )
    filter_class_person = [0]
    filter_class_object = [0, 1, 2, 3]  # OBJECT
    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1
    )

    video_input_path = Path("./videos/test.avi")
    video_output_path = Path("./results/videos/test-best_17Sep-pose.avi")
    video_process(
        _video_input_path=video_input_path,
        _video_output_path=video_output_path,
        _class_person_names_dict=class_person_names_dict,
        _class_objects_names_dict=class_objects_names_dict,
        _filter_class_person=filter_class_person,
        _filter_class_object=filter_class_object,
    )
