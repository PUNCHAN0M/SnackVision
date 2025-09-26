# -*- coding: utf-8 -*-
"""ไฟล์อธิบาย main.py แบบบรรทัดต่อบรรทัด (ภาษาไทย)

หมายเหตุ:
- เป็นสำเนา (annotated copy) ไม่กระทบไฟล์รันจริง `main.py`
- ใช้สำหรับการเรียนรู้โครงสร้างระบบ YOLO + BYTETrack + การ annotate
- Flow หลัก: parse_cli -> สร้าง VideoTracker -> run()

Data Flow Summary (ภาพรวม):
1. parse_cli() อ่าน argument -> สร้าง TrackerConfig
2. main() สร้าง VideoTracker(config)
3. VideoTracker.run():
   - เปิดวิดีโอผ่าน get_video_frames_generator
   - แต่ละเฟรม -> FrameProcessor.detect_persons() ใช้ YOLO ตรวจจับ
   - assign_track_ids() ผูก detection กับ track id (BYTETracker)
   - update_track_infos() เก็บ center history + จัดการ track หาย
   - classify_objects_in_tracks() ตรวจจับวัตถุรอง (object model) แล้วแม็พเข้า track
   - build_labels() สร้าง label per detection
   - TrackingVisualizer.annotate_frame() วาดกรอบ + ออบเจกต์ + เส้นทาง
   - เขียนเฟรมออกผ่าน VideoSink

"""
from __future__ import (
    annotations,
)  # เปิดใช้ postponed annotations (PEP 563/649) ช่วยลด circular import
import logging  # โมดูล logging สำหรับ debug/info
from dataclasses import dataclass, field  # ใช้สร้างคลาส config และโครงสร้างข้อมูล
from pathlib import Path  # จัดการ path แบบ cross-platform
from typing import Iterable, List, Optional, Sequence  # ใช้กำหนด type hints
from tqdm import tqdm  # แสดง progress bar ระหว่าง loop เฟรม
from ultralytics import YOLO  # โมเดล YOLO จาก ultralytics
import numpy as np  # ใช้ประมวลผล array, คำนวณเวกเตอร์
import cv2  # OpenCV สำหรับอ่าน/เขียน/วาดภาพ
from cli_args import (
    parse_args as raw_parse_args,
)  # ฟังก์ชันอ่าน argument จากไฟล์ cli_args.py
from onemetric.cv.utils.iou import (
    box_iou_batch,
)  # ฟังก์ชันคำนวณ IoU ชุดใหญ่เพื่อ match detection-track
from supervision.tools.detections import (
    Detections,
    BoxAnnotator,
)  # Detections โครงสร้างเก็บผลตรวจ / BoxAnnotator วาดกรอบ
from supervision.draw.color import ColorPalette  # พาเลทสีสำเร็จรูป
from supervision.video.dataclasses import (
    VideoInfo,
)  # เก็บเมตะข้อมูลวิดีโอ (fps, frame_count, size)
from supervision.video.sink import VideoSink  # ตัวเขียนวิดีโอผลลัพธ์
from supervision.video.source import (
    get_video_frames_generator,
)  # ตัว generator อ่านเฟรมจากไฟล์วิดีโอ
from yolox.tracker.byte_tracker import (
    BYTETracker,
    STrack,
)  # อัลกอริทึม BYTETrack + โครงสร้าง track
from tracking_helpers import (
    FrameProcessor,
    TrackingVisualizer,
)  # โมดูลที่เราแยกออกมาช่วยจัดการ logic ย่อย

LOG = logging.getLogger(__name__)  # สร้าง logger ชื่อ module


# -------------------- BYTETracker Args --------------------
@dataclass(frozen=True)  # frozen เพื่อไม่ให้แก้ค่าภายหลัง (immutable semantics)
class BYTETrackerArgs:
    """Wrapper บรรจุพารามิเตอร์ให้ BYTETracker (ลดการกระจายตัวของค่า config)"""

    track_thresh: float  # ค่าความเชื่อมั่นขั้นต่ำเมื่อจะเริ่มตามวัตถุ
    track_buffer: int  # จำนวนเฟรม buffer สำหรับเก็บ track ที่หายชั่วคราว
    match_thresh: float  # เกณฑ์ IoU/association สำหรับจับคู่ detection กับ track
    aspect_ratio_thresh: float  # ใช้กรองบ็อกซ์ที่สัดส่วนผิดปกติ
    min_box_area: float  # พื้นที่ขั้นต่ำของกล่อง
    mot20: bool  # โหมดพิเศษถ้าใช้กับ MOT20 (ปรับ internal heuristic)


# -------------------- TrackerConfig --------------------
@dataclass
class TrackerConfig:
    # ฟิลด์จำเป็น (ไม่กำหนด default)
    model_person_path: Path  # path โมเดลหลัก (ตรวจคน หรือ class ที่สนใจ)
    model_object_path: Path  # path โมเดลวัตถุรอง (ของในมือ ฯลฯ)
    source_video_path: Path  # path วิดีโอต้นฉบับ
    target_video_path: Path  # path วิดีโอผลลัพธ์
    class_ids: Sequence[int]  # class id ที่จะเก็บจากโมเดลหลัก
    max_frames: Optional[int]  # จำกัดจำนวนเฟรม (None = ทั้งหมด)
    track_thresh: float
    track_buffer: int
    match_thresh: float
    aspect_ratio_thresh: float
    min_box_area: float
    mot20: bool
    progress: bool  # แสดง progress bar หรือไม่
    # ฟิลด์มีค่า default
    model_person_conf_threshold: float = 0.25  # ค่าความเชื่อมั่นขั้นต่ำสำหรับตรวจคน
    model_object_conf_threshold: float = 0.25  # ค่าความเชื่อมั่นขั้นต่ำสำหรับตรวจวัตถุรอง

    def to_byte_args(self) -> BYTETrackerArgs:  # แปลง config เป็นออบเจกต์ BYTETrackerArgs
        return BYTETrackerArgs(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            aspect_ratio_thresh=self.aspect_ratio_thresh,
            min_box_area=self.min_box_area,
            mot20=self.mot20,
        )


# -------------------- VideoTracker หลัก --------------------
class VideoTracker:
    @dataclass
    class TrackInfo:  # เก็บสถานะต่อ track (หนึ่งบุคคล/วัตถุที่กำลังตาม)
        last_xyxy: list[float]  # พิกัดกล่องล่าสุด
        center_history: list[tuple[float, float]] = field(
            default_factory=list
        )  # ประวัติจุดศูนย์กลาง
        hold_inhand: list[str] = field(default_factory=list)  # (สำรอง) รายการวัตถุที่ถือ
        object_classes: list[dict] = field(default_factory=list)  # [{name,index,count}]
        object_boxes: list[dict] = field(
            default_factory=list
        )  # กล่องวัตถุรองที่จับบนเฟรมล่าสุด

    def __init__(self, cfg: TrackerConfig):  # สร้างอินสแตนซ์โดยรับ config
        self.cfg = cfg
        self._model_person: Optional[YOLO] = None  # lazy load โมเดลหลัก
        self._model_object: Optional[YOLO] = None  # lazy load โมเดลวัตถุรอง
        self._byte_tracker: Optional[BYTETracker] = None  # lazy init tracker
        self._class_names: Optional[dict] = None  # เก็บ mapping id->ชื่อ class จากโมเดลหลัก
        self._box_annotator: Optional[BoxAnnotator] = (
            None  # ใช้วาดกล่อง det หลัก (ยังเผื่อไว้)
        )
        self._track_infos: dict[int, VideoTracker.TrackInfo] = (
            {}
        )  # track ปัจจุบัน (active)
        self._finished_track_infos: dict[int, VideoTracker.TrackInfo] = (
            {}
        )  # track ที่หายแล้ว (เก็บ info)
        self._frame_processor = FrameProcessor(
            self
        )  # helper สำหรับ detection/assign/build labels
        self._visualizer = TrackingVisualizer()  # helper สำหรับการวาด

    # ---------- Lazy property โหลดโมเดลหลัก ----------
    @property
    def model_person(self) -> YOLO:
        if self._model_person is None:  # โหลดครั้งแรกเท่านั้น
            LOG.info("Loading person model: %s", self.cfg.model_person_path)
            self._model_person = YOLO(str(self.cfg.model_person_path))  # สร้างโมเดล
            try:
                self._model_person.fuse()  # รวม layer เพื่อ optimize (ถ้ารองรับ)
            except Exception:
                pass  # ถ้า fuse ไม่ได้ก็ข้าม
            self._class_names = self._model_person.model.names  # ดึงชื่อ class
            LOG.debug("Loaded person model with %d classes", len(self._class_names))
        return self._model_person

    # ---------- Lazy property โหลดโมเดลวัตถุรอง ----------
    @property
    def model_object(self) -> YOLO:
        if self._model_object is None:
            path = self.cfg.model_object_path
            if not path.exists():  # ตรวจว่ามีไฟล์ไหม
                raise FileNotFoundError(f"Secondary object model not found: {path}")
            LOG.info("Loading object model: %s", path)
            self._model_object = YOLO(str(path))
            try:
                self._model_object.fuse()
            except Exception:
                pass
        return self._model_object

    # ---------- Lazy property สำหรับ BYTETracker ----------
    @property
    def byte_tracker(self) -> BYTETracker:
        if self._byte_tracker is None:
            args = self.cfg.to_byte_args()  # แปลง config
            self._byte_tracker = BYTETracker(args)  # สร้าง tracker
            LOG.debug("Initializing BYTETracker: %s", args)
        return self._byte_tracker

    # ---------- Lazy property สำหรับ BoxAnnotator (ยังคงไว้กรณีใช้นอกรอบ) ----------
    @property
    def box_annotator(self) -> BoxAnnotator:
        if self._box_annotator is None:
            self._box_annotator = BoxAnnotator(
                color=ColorPalette(), thickness=2, text_thickness=1, text_scale=1
            )
        return self._box_annotator

    # ---------- Utility แปลง Detections เป็น array (BYTETracker format) ----------
    @staticmethod
    def detections2boxes(detections: Detections) -> np.ndarray:
        return np.hstack(
            (detections.xyxy, detections.confidence[:, np.newaxis])
        )  # รวมพิกัดกับคอนฟิเดนซ์

    @staticmethod
    def tracks2boxes(
        tracks: List[STrack],
    ) -> np.ndarray:  # ดึง tlbr ของทุก track เป็น array
        return np.array([track.tlbr for track in tracks], dtype=float)

    @classmethod
    def match_detections_with_tracks(
        cls, detections: Detections, tracks: List[STrack]
    ) -> List[Optional[int]]:
        if len(detections) == 0 or len(tracks) == 0:  # ถ้าไม่มีอะไรให้ match
            return []
        tracks_boxes = cls.tracks2boxes(tracks=tracks)  # กล่องของ track
        iou = box_iou_batch(tracks_boxes, detections.xyxy)  # คำนวณ IoU matrix
        track2detection = np.argmax(iou, axis=1)  # หา detection ที่ IoU สูงสุดต่อ track
        tracker_ids: List[Optional[int]] = [None] * len(detections)  # เตรียมลิสต์ผลลัพธ์
        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:  # ถ้า IoU > 0
                tracker_ids[detection_index] = tracks[
                    tracker_index
                ].track_id  # กำหนด id
        return tracker_ids

    # ---------- อัปเดตสถานะ track (ตำแหน่ง+ประวัติ) ----------
    def update_track_infos(self, detections: Detections) -> None:
        current_ids: set[int] = set()  # track id ที่ยังอยู่ในเฟรมนี้
        for xyxy, conf, cid, tid in detections:  # loop ทุก detection
            if tid is None:
                continue  # ข้ามถ้าไม่มี track id
            current_ids.add(tid)
            x1, y1, x2, y2 = xyxy.tolist()
            cx = (x1 + x2) / 2.0  # center x
            cy = (y1 + y2) / 2.0  # center y
            if tid not in self._track_infos:  # ถ้าเพิ่งปรากฏ
                self._track_infos[tid] = VideoTracker.TrackInfo(
                    last_xyxy=[x1, y1, x2, y2]
                )
            info = self._track_infos[tid]
            info.last_xyxy = [x1, y1, x2, y2]  # อัปเดตกล่องล่าสุด
            info.center_history.append((cx, cy))  # เก็บประวัติ center
        disappeared = [
            tid for tid in self._track_infos if tid not in current_ids
        ]  # track ที่หายไป
        for tid in disappeared:
            self._finished_track_infos[tid] = self._track_infos.pop(
                tid
            )  # ย้ายไป finished

    def get_active_tracks(
        self,
    ) -> dict[int, "VideoTracker.TrackInfo"]:  # getter ใช้ภายนอก
        return self._track_infos

    def get_finished_tracks(
        self,
    ) -> dict[int, "VideoTracker.TrackInfo"]:  # getter tracks ที่จบแล้ว
        return self._finished_track_infos

    # ---------- ตรวจจับวัตถุรอง (object model) แล้วแม็พเข้า track ----------
    def classify_objects_in_tracks(
        self, frame: np.ndarray, detections: Detections
    ) -> None:
        for (
            info
        ) in self._track_infos.values():  # เคลียร์ object_boxes ก่อน (เฉพาะเฟรมปัจจุบัน)
            info.object_boxes = []
        if not self._track_infos:
            return  # ไม่มี track ก็ไม่ทำอะไร
        try:
            obj_results = self.model_object(
                frame, verbose=False, conf=self.cfg.model_object_conf_threshold
            )  # รันโมเดลวัตถุรอง
        except FileNotFoundError:
            return  # ไม่มีโมเดลก็เงียบ ๆ
        except Exception as e:  # ป้องกัน error อื่น
            LOG.warning("Object model inference failed: %s", e)
            return
        if not obj_results:
            return
        res = obj_results[0]  # ผลลัพธ์แรก (batch=1)
        names = getattr(res, "names", None)  # mapping id->ชื่อ class (ถ้ามี)
        boxes = getattr(res, "boxes", None)  # กล่องผลตรวจ
        if boxes is None:
            return
        cls_tensor = getattr(boxes, "cls", None)
        xyxy_tensor = getattr(boxes, "xyxy", None)
        conf_tensor = getattr(boxes, "conf", None)
        if cls_tensor is None or xyxy_tensor is None:
            return
        cls_ids = cls_tensor.cpu().numpy().astype(int)  # class id ทั้งหมด
        all_xyxy = xyxy_tensor.cpu().numpy()  # พิกัดทั้งหมด
        confidences = (
            conf_tensor.cpu().numpy()
            if conf_tensor is not None
            else np.zeros(len(cls_ids))
        )
        track_box_map: dict[int, np.ndarray] = {}  # map track id -> กล่องคนล่าสุด
        for xyxy, conf, cid, tid in detections:
            if tid is None:
                continue
            track_box_map[tid] = xyxy
        for cid, obj_box, conf_val in zip(
            cls_ids, all_xyxy, confidences
        ):  # loop กล่องวัตถุรอง
            x1o, y1o, x2o, y2o = obj_box
            obj_area = max(0.0, (x2o - x1o)) * max(0.0, (y2o - y1o))
            if obj_area <= 0:
                continue
            ratios: list[tuple[int, float]] = []  # เก็บ (track_id, overlap_ratio)
            for tid, pbox in track_box_map.items():  # คำนวณ overlap ต่อ track
                x1p, y1p, x2p, y2p = pbox
                ix1 = max(x1o, x1p)
                iy1 = max(y1o, y1p)
                ix2 = min(x2o, x2p)
                iy2 = min(y2o, y2p)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                ratio = inter / obj_area if obj_area > 0 else 0.0
                ratios.append((tid, ratio))
            if not ratios:
                continue
            ratios.sort(key=lambda x: x[1], reverse=True)  # เรียงจาก overlap มากสุด
            if ratios[0][1] <= 0:
                continue  # ไม่มี overlap
            if len(ratios) > 1 and abs(ratios[0][1] - ratios[1][1]) < 1e-6:
                continue  # ผูกไม่ได้เพราะเสมอกันเป๊ะ
            best_tid = ratios[0][0]
            if best_tid not in self._track_infos:
                continue
            info = self._track_infos[best_tid]
            cname = str(names[cid]) if names and cid in names else str(cid)
            info.object_boxes.append(
                {  # เก็บกล่องวัตถุรอง
                    "name": cname,
                    "xyxy": tuple(
                        map(
                            int,
                            obj_box.tolist() if hasattr(obj_box, "tolist") else obj_box,
                        )
                    ),
                    "confidence": float(conf_val),
                }
            )
            existing = [e for e in info.object_classes if e["name"] == cname]
            if not existing:
                info.object_classes.append({"name": cname, "index": 1, "count": 1})
            else:
                existing_sorted = sorted(existing, key=lambda x: x["index"])
                existing_sorted[0]["count"] += 1  # เพิ่มจำนวนครั้งที่เจอ

    # ---------- วนลูปหลักประมวลผลวิดีโอ ----------
    def run(self) -> None:
        if not self.cfg.source_video_path.exists():  # ตรวจว่ามีไฟล์ต้นทางไหม
            raise FileNotFoundError(
                f"Source video not found: {self.cfg.source_video_path}"
            )
        self.cfg.target_video_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # สร้างโฟลเดอร์ผลลัพธ์ถ้ายังไม่มี
        video_info = VideoInfo.from_video_path(
            str(self.cfg.source_video_path)
        )  # อ่านข้อมูลวิดีโอ
        frames: Iterable[np.ndarray] = get_video_frames_generator(
            str(self.cfg.source_video_path)
        )  # generator เฟรม
        total_frames = getattr(video_info, "total_frames", None)  # จำนวนเฟรม (ถ้ามี)
        if self.cfg.progress:
            frames = tqdm(frames, desc="Tracking", total=total_frames)  # wrap ด้วย tqdm
        with VideoSink(
            str(self.cfg.target_video_path), video_info
        ) as sink:  # เปิด writer
            for frame_index, frame in enumerate(frames):  # loop เฟรม
                if (
                    self.cfg.max_frames and frame_index >= self.cfg.max_frames
                ):  # จำกัดเฟรมถ้ามี
                    break
                detections = self._frame_processor.detect_persons(
                    frame
                )  # ตรวจคน/คลาสหลัก
                if detections is None:  # ไม่มี det -> บันทึกเฟรมดิบแล้วข้าม
                    sink.write_frame(frame)
                    continue
                self._frame_processor.assign_track_ids(
                    detections, frame.shape
                )  # ติด track id
                self.update_track_infos(detections)  # อัปเดต historian track
                self.classify_objects_in_tracks(frame, detections)  # ตรวจวัตถุรองและแม็พ
                labels = self._frame_processor.build_labels(detections)  # สร้าง label
                annotated = self._visualizer.annotate_frame(
                    frame, detections, labels, self._track_infos
                )  # วาดทั้งหมด
                sink.write_frame(annotated)  # เขียนออก
        LOG.info(
            "Completed tracking. Active=%d Finished=%d",
            len(self._track_infos),
            len(self._finished_track_infos),
        )


# ---------- parse_cli สร้าง config จาก arguments ----------
def parse_cli(argv: Optional[Sequence[str]] = None) -> TrackerConfig:
    args = raw_parse_args(argv)  # อ่าน args จาก cli_args.py
    max_frames = (
        args.max_frames if args.max_frames and args.max_frames > 0 else None
    )  # แปลง 0/None -> None
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )  # ตั้งค่า logging
    return TrackerConfig(
        model_person_path=args.model_person,
        model_object_path=args.model_object,
        source_video_path=args.source,
        target_video_path=args.target,
        class_ids=args.class_ids,
        max_frames=max_frames,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        aspect_ratio_thresh=args.aspect_ratio_thresh,
        min_box_area=args.min_box_area,
        mot20=args.mot20,
        progress=not args.no_progress,  # ถ้า user ใส่ --no-progress -> False
        model_person_conf_threshold=args.model_person_confidence_threshold,
        model_object_conf_threshold=args.model_object_confidence_threshold,
    )


# ---------- main entry ----------
def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_cli(argv)  # สร้าง config
    LOG.debug("Final config: %s", cfg)  # debug ค่า config
    tracker = VideoTracker(cfg)  # สร้าง tracker
    tracker.run()  # เริ่มประมวลผล


if __name__ == "__main__":  # ถ้ารันโดยตรง (ไม่ใช่ import)
    main()
