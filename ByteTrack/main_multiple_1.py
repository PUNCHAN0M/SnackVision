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
import config


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


def filter_by_class_id(detections: Detections, input_class_id: List[int]) -> Detections:

    mask = np.array(
        [class_id in input_class_id for class_id in detections.class_id], dtype=bool
    )
    detections.filter(mask=mask, inplace=True)
    return detections


def filter_by_tracker_id(detections: Detections) -> Detections:
    mask = np.array([tid is not None for tid in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    return detections


def video_process(
    _video_input_path: Path,
    _video_output_path: Path,
    _class_names_dict: dict,
    _filter_class_id: List[int],
):
    video_info = VideoInfo.from_video_path(_video_input_path)
    generator = get_video_frames_generator(_video_input_path)
    with VideoSink(_video_output_path, video_info) as sink:
        for frame in tqdm(generator, total=video_info.total_frames):
            results = model(frame, conf=0.5)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )
            # ___________FILTER_____________
            detections = filter_by_class_id(
                detections=detections, input_class_id=_filter_class_id
            )
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )
            tracker_id = match_detections_with_tracks(
                detections=detections, tracks=tracks
            )
            detections.tracker_id = np.array(tracker_id)
            detections = filter_by_tracker_id(detections)
            # __________END FILTER___________

            # __________CREATE LABEL AND DRAW BOX___________
            labels = [
                f"#{tracker_id} {_class_names_dict[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections
            ]
            frame = box_annotator.annotate(
                frame=frame, detections=detections, labels=labels
            )
            sink.write_frame(frame)
            # __________END FOR___________


def model_initialization(model_path: Optional[str] = None):
    model = YOLO(model_path)
    class_names_dict = model.model.names
    model.eval()  # 1. ตั้งเป็น evaluation mode ก่อน
    model.fuse()  # 2. รวม Conv + BN เพื่อ optimize
    return model, class_names_dict


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

    model, class_names_dict = model_initialization(
        model_path="./models/best_16Sep_2.pt"
    )
    filter_class_id = [0, 1, 2, 3, 4]

    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=2, text_thickness=1, text_scale=1
    )

    video_input_path = Path("./videos/test.avi")
    video_output_path = Path("./results/videos/test.avi")
    video_process(
        _video_input_path=video_input_path,
        _video_output_path=video_output_path,
        _class_names_dict=class_names_dict,
        _filter_class_id=filter_class_id,
    )
