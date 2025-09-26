"""
Detect
1. person (Tracking),
2. object

Filter by
1. id
2. min height ratio
"""

import time
import cv2
import numpy as np
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


# ----------------- DATA CLASSES -----------------
@dataclass
class ObjectsInfo:
    object_class: str  # ชื่อ class ของวัตถุ เช่น "drink"
    item_onhand: int  # ลำดับของ item ในมือ (1,2,...)
    frequency_occurrences: int = 1  # จำนวนครั้งที่พบ object นี้


@dataclass
class PersonInfo:
    tracker_id: int
    boundingbox: Tuple[int, int, int, int]
    boundingbox_pose: Tuple[int, int, int, int]
    boundingbox_wrist: Tuple[int, int, int, int]
    objects: List[ObjectsInfo] = field(default_factory=list)
    tracker_in_frame: int = 0

    def cleanup_objects(self):
        """ลบ ObjectsInfo ที่ frequency_occurrences = 0 ออกจาก list"""
        self.objects = [obj for obj in self.objects if obj.frequency_occurrences > 0]


# ----------------- HELPERS -----------------
def detections2boxes(detections: Detections) -> np.ndarray:
    """Convert detections to format usable by ByteTrack"""
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    """Convert tracks to np.array"""
    return np.array([track.tlbr for track in tracks], dtype=float)


def match_detections_with_tracks(
    detections: Detections, tracks: List[STrack]
) -> List[Optional[int]]:
    """Match detection boxes with tracker IDs"""
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
    """Keep detections that have tracker_id"""
    mask = np.array([tid is not None for tid in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    return detections


def filter_min_height(
    detections: Detections, frame: np.ndarray, ratio: float = None
) -> Detections:
    """Filter out persons shorter than ratio of frame height"""
    if len(detections) == 0:
        return detections
    frame_height = frame.shape[0]
    xyxy = detections.xyxy
    heights = xyxy[:, 3] - xyxy[:, 1]
    relative_heights = heights / frame_height
    mask = relative_heights >= ratio
    detections = detections.filter(mask=mask, inplace=True)
    return detections


# ----------------- MAIN LOGIC -----------------
def update_person_info(
    detections_person: Detections,
    detections_objects: Detections,
    class_objects_names_dict: dict,
    persons: dict[int, PersonInfo],
):
    """Update persons dictionary with new detections"""
    for xyxy, conf, cls_id, tid in detections_person:
        if tid is None:
            continue

        bbox = tuple(map(int, xyxy))  # (x1,y1,x2,y2)

        if tid in persons:
            person = persons[tid]
            person.boundingbox = bbox
            person.tracker_in_frame += 1  # ✅ เพิ่ม count
        else:
            # tracker_id ใหม่ → สร้าง PersonInfo
            person = PersonInfo(
                tracker_id=tid,
                boundingbox=bbox,
                boundingbox_pose=None,
                boundingbox_wrist=None,
                objects=[],
                tracker_in_frame=1,  # ✅ เริ่มต้นนับ 1
            )
            persons[tid] = person

        # ------------ update objects -------------
        current_frame_objects = []

        for obj_xyxy, obj_conf, obj_cls in zip(
            detections_objects.xyxy,
            detections_objects.confidence,
            detections_objects.class_id,
        ):
            obj_name = class_objects_names_dict[obj_cls]
            current_frame_objects.append(obj_name)

            matched_obj = None
            for o in person.objects:
                if o.object_class == obj_name:
                    matched_obj = o
                    break

            if matched_obj:
                matched_obj.frequency_occurrences += 1
            else:
                existing_count = sum(
                    1 for o in person.objects if o.object_class == obj_name
                )
                new_obj = ObjectsInfo(
                    object_class=obj_name,
                    item_onhand=existing_count + 1,
                    frequency_occurrences=1,
                )
                person.objects.append(new_obj)

        # for o in person.objects:
        #     if o.object_class not in current_frame_objects:
        #         o.frequency_occurrences -= 1
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

            detected_person = model_person(
                frame, conf=0.8, classes=_filter_class_person, verbose=False
            )
            detected_objects = model_object(
                frame, conf=0.65, classes=_filter_class_object, verbose=False
            )

            detections_person = Detections(
                xyxy=detected_person[0].boxes.xyxy.cpu().numpy(),
                confidence=detected_person[0].boxes.conf.cpu().numpy(),
                class_id=detected_person[0].boxes.cls.cpu().numpy().astype(int),
            )
            detections_person = filter_min_height(detections_person, frame, ratio=0.4)

            detections_objects = Detections(
                xyxy=detected_objects[0].boxes.xyxy.cpu().numpy(),
                confidence=detected_objects[0].boxes.conf.cpu().numpy(),
                class_id=detected_objects[0].boxes.cls.cpu().numpy().astype(int),
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

            # update persons dict
            update_person_info(
                detections_person,
                detections_objects,
                _class_objects_names_dict,
                persons,
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
            sink.write_frame(frame)
            # __________END FOR___________


import csv


def save_results_to_csv(
    persons: dict[int, PersonInfo], class_objects_names_dict: dict, video_name: str
):
    output_path = Path(f"./results/text/{video_name}{take}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = list(class_objects_names_dict.values())
    header = (
        ["tracker_id", "len_objects", "tracker_in_frame"]
        + [f"count_{c}" for c in class_names]
        + ["objects"]
    )

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for tid, person in persons.items():
            # ✅ ตัด tracker ที่เจอน้อยกว่า 400 frame
            if person.tracker_in_frame < 200:
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
                + [objects_list]
            )
            writer.writerow(row)

    print(f"[INFO] Results saved → {output_path}")


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

    model_person, model_object, class_person_names_dict, class_objects_names_dict = (
        model_initialization(
            model_person_path="./models/best_done.pt",
            model_object_path="./models/best_done.pt",
        )
    )
    filter_class_person = [4]
    filter_class_object = [0, 1, 2, 3]  # OBJECT
    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1
    )
    video_name = "test"
    take = "_1"
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
    save_results_to_csv(persons, class_objects_names_dict, video_name)
