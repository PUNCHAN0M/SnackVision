""" """

import cv2
import numpy as np
from tqdm import tqdm
import logging
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
    detections: Detections, frame: np.ndarray, ratio: float = 0.5
) -> None:
    if len(detections) == 0:
        return
    frame_h = frame.shape[0]
    xyxy = detections.xyxy
    heights = xyxy[:, 3] - xyxy[:, 1]
    mask = heights >= ratio * frame_h
    detections.filter(mask, inplace=True)


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
                frame, conf=0.8, classes=_filter_class_person
            )
            detected_objects = model_object(
                frame, conf=0.8, classes=_filter_class_object
            )
            detections_person = Detections(
                xyxy=detected_person[0].boxes.xyxy.cpu().numpy(),
                confidence=detected_person[0].boxes.conf.cpu().numpy(),
                class_id=detected_person[0].boxes.cls.cpu().numpy().astype(int),
            )
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
            detections_person = filter_by_tracker_id(detections_person)
            # __________END FILTER___________

            # __________CREATE LABEL AND DRAW BOX___________
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
            sink.write_frame(frame)
            # __________END FOR___________


def model_initialization(
    model_person_path: Optional[str] = None, model_object_path: Optional[str] = None
):
    model_person = YOLO(model_person_path)
    model_object = YOLO(model_object_path)
    class_person_names_dict = model_person.model.names
    class__objects_names_dict = model_object.model.names
    model_person.eval()
    # model_object.eval()
    model_person.fuse()
    model_object.fuse()
    return (
        model_person,
        model_object,
        class_person_names_dict,
        class__objects_names_dict,
    )


if __name__ == "__main__":

    @dataclass(frozen=True)
    class BYTETrackerArgs:
        track_thresh: float = 0.8
        track_buffer: int = 100
        match_thresh: float = 0.8
        aspect_ratio_thresh: float = 3.0
        min_box_area: float = 1.0
        mot20: bool = False

    byte_tracker = BYTETracker(BYTETrackerArgs())

    model_person, model_object, class_person_names_dict, class_objects_names_dict = (
        model_initialization(
            model_person_path="./models/yolo11x.pt",
            model_object_path="./models/best_16Sep.pt",
        )
    )
    filter_class_person = [0]
    filter_class_object = [0, 1, 2, 3]  # OBJECT
    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=2, text_thickness=1, text_scale=1
    )

    video_input_path = Path("./videos/person.avi")
    video_output_path = Path("./results/videos/person.avi")
    video_process(
        _video_input_path=video_input_path,
        _video_output_path=video_output_path,
        _class_person_names_dict=class_person_names_dict,
        _class_objects_names_dict=class_objects_names_dict,
        _filter_class_person=filter_class_person,
        _filter_class_object=filter_class_object,
    )
