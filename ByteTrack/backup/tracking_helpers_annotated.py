# -*- coding: utf-8 -*-
"""ไฟล์อธิบาย tracking_helpers.py แบบบรรทัดต่อบรรทัด (ภาษาไทย)

ประกอบด้วย 2 คลาสหลัก:
- FrameProcessor : จัดการขั้นตอนตรวจจับ (YOLO), กรอง, ผูก track id, สร้าง label
- TrackingVisualizer : วาดกรอบ, วาดเส้นทางเคลื่อนที่, วาดกล่องวัตถุรอง, สร้างสี

การเชื่อมกับ VideoTracker:
VideoTracker ถืออินสแตนซ์ FrameProcessor และ TrackingVisualizer แล้วเรียกใช้ตามลำดับใน run():
 detect_persons -> assign_track_ids -> update_track_infos -> classify_objects_in_tracks -> build_labels -> annotate_frame
"""
from __future__ import annotations  # ใช้ postponed annotations ลดปัญหา circular import

from typing import (
    Optional,
    List,
    Dict,
    TYPE_CHECKING,
)  # type hints พื้นฐาน + ตรวจเฉพาะตอน type check

import cv2  # ใช้วาดสัญลักษณ์ (line, rectangle, text)
import numpy as np  # คำนวณเชิงเมทริกซ์/เวกเตอร์
from supervision.tools.detections import (
    Detections,
    BoxAnnotator,
)  # Detections = โครงสร้างผลตรวจ, BoxAnnotator = ตัวช่วยวาด
from supervision.draw.color import ColorPalette  # ให้ palette สีที่สุ่ม/สวยงามเป็นมาตรฐาน

if TYPE_CHECKING:  # โค้ดในบล็อคนี้รันเฉพาะตอน type checking (mypy/pyright) ไม่รัน runtime
    from .main import (
        VideoTracker,
    )  # import เพื่อใช้เป็น type hint เฉย ๆ ไม่ก่อให้เกิดวงกลมเวลารัน


class FrameProcessor:  # คลาสรวมกระบวนการเกี่ยวกับ detection + tracking mapping + label generation
    """Encapsulates detection, tracker assignment, label building, and filters."""

    def __init__(
        self, tracker: "VideoTracker"
    ):  # รับอ้างอิงกลับไปที่ VideoTracker (composition)
        self.tracker = tracker  # เก็บไว้ใช้เรียกโมเดลและ config

    def detect_persons(
        self, frame: np.ndarray
    ) -> Optional[Detections]:  # ตรวจจับบุคคล/คลาสหลักในเฟรม
        result = self.tracker.model_person(  # เรียกโมเดล YOLO หลัก
            frame, verbose=False, conf=self.tracker.cfg.model_person_conf_threshold
        )
        boxes = result[0].boxes  # ผลลัพธ์ batch แรก (เพราะส่งทีละเฟรม)
        detections = Detections(  # สร้างอ็อบเจกต์ Detections จากกล่อง/ความมั่นใจ/คลาส
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int),
        )
        detections.filter(  # กรองให้เหลือเฉพาะ class id ที่ config กำหนด
            mask=np.isin(detections.class_id, list(self.tracker.cfg.class_ids)),
            inplace=True,
        )
        if len(detections) == 0:  # ถ้าไม่มีอะไรเลย -> คืน None
            return None
        self.filter_min_height(
            detections, frame, ratio=0.0
        )  # (ตอนนี้ ratio=0.0 = ไม่กรองเพิ่ม แต่เว้น hook ไว้)
        if len(detections) == 0:
            return None
        return detections  # คืนผลตรวจที่ผ่านการกรอง

    def assign_track_ids(
        self, detections: Detections, frame_shape: tuple[int, ...]
    ) -> None:  # จับคู่ detection -> track id
        tracks = self.tracker.byte_tracker.update(  # อัปเดต BYTETracker ด้วยกล่องล่าสุด
            output_results=self.tracker.detections2boxes(detections),
            img_info=frame_shape,
            img_size=frame_shape,
        )
        tracker_id = self.tracker.match_detections_with_tracks(
            detections, tracks
        )  # หาว่า detection ไหนเป็น track id อะไร
        detections.tracker_id = np.array(
            tracker_id
        )  # เซ็ต field tracker_id ใน Detections
        if detections.tracker_id is not None:  # กรองทิ้งอันที่ไม่ได้ถูกจับคู่ (None)
            detections.filter(
                mask=np.array(
                    [tid is not None for tid in detections.tracker_id], dtype=bool
                ),
                inplace=True,
            )

    @staticmethod
    def filter_min_height(
        detections: Detections, frame: np.ndarray, ratio: float = 0.5
    ) -> None:  # กรองตามความสูงสัมพัทธ์เฟรม
        if len(detections) == 0:
            return
        frame_h = frame.shape[0]
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]  # h = y2 - y1
        mask = heights >= ratio * frame_h  # เลือกอันที่สูงพอ
        detections.filter(mask, inplace=True)

    def build_labels(
        self, detections: Detections
    ) -> list[str]:  # สร้างข้อความ label สำหรับแสดงบนกรอบ
        labels: list[str] = []
        track_infos = self.tracker._track_infos  # เข้าถึง track info ภายใน VideoTracker
        for xyxy, conf, cid, tid in detections:  # วนทุก detection
            if tid is None:  # ถ้าไม่มี track id
                labels.append(f"#? {conf:0.2f}")
                continue
            info = track_infos.get(tid)
            if info and info.object_classes:  # ถ้า track นั้นมีวัตถุรองสะสม
                ordered = sorted(
                    info.object_classes, key=lambda e: (e["name"], e["index"])
                )
                parts = [
                    f"{e['name']}({e['index']}:{e['count']})" for e in ordered[:4]
                ]  # จำกัด 4 รายการแรก
                labels.append(f"#{tid} {conf:0.2f} [{','.join(parts)}]")
            else:
                labels.append(f"#{tid} {conf:0.2f}")
        return labels


class TrackingVisualizer:  # คลาสสำหรับงานวาดทั้งหมด
    """Handles drawing of detection boxes, object boxes, centers and trajectories."""

    def __init__(
        self,
        palette: Optional[ColorPalette] = None,
        thickness: int = 2,
        text_thickness: int = 1,
        text_scale: float = 1.0,
    ):
        self.palette = palette or ColorPalette()  # ใช้ palette ค่าเริ่มต้นถ้าไม่ได้ส่งมา
        self.box_annotator = BoxAnnotator(  # สร้าง annotator สำหรับวาดกรอบ detection หลัก
            color=self.palette,
            thickness=thickness,
            text_thickness=text_thickness,
            text_scale=text_scale,
        )
        self._object_color_map: Dict[str, tuple[int, int, int]] = (
            {}
        )  # แคชสี per object name

    def _color_for_object(
        self, name: str
    ) -> tuple[int, int, int]:  # สร้างสีแบบ deterministic จากชื่อ
        if name in self._object_color_map:
            return self._object_color_map[name]
        hv = sum(ord(c) for c in name)  # แปลงสตริงเป็น hash เล็ก ๆ
        color = (
            (hv * 3) % 200 + 30,
            (hv * 7) % 200 + 30,
            (hv * 11) % 200 + 30,
        )  # กระจายค่าให้อยู่ในช่วงดูชัด
        self._object_color_map[name] = color
        return color

    def draw_centers(
        self, frame: np.ndarray, track_infos: Dict[int, object], tail: int = 15
    ) -> None:  # วาดจุด center และเส้นทางล่าสุด
        is_gray = len(frame.shape) == 2 or (
            len(frame.shape) == 3 and frame.shape[2] == 1
        )  # เช็คว่าเป็นภาพเทาหรือไม่
        for tid, info in track_infos.items():
            history = getattr(info, "center_history", [])
            if not history:
                continue
            b = int((37 * tid) % 255)
            g = int((17 * tid) % 255)
            r = int((97 * tid) % 255)
            color = (
                (b, g, r) if not is_gray else int(0.299 * r + 0.587 * g + 0.114 * b)
            )  # แปลงเป็น gray ถ้าจำเป็น
            pts = history[-tail:]  # ใช้เฉพาะจุดท้าย ๆ เพื่อลดความยาวเส้น
            for i in range(1, len(pts)):
                x1, y1 = map(int, pts[i - 1])
                x2, y2 = map(int, pts[i])
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            cx, cy = map(int, pts[-1])
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(
                frame,
                f"ID {tid}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def draw_object_detections(
        self, frame: np.ndarray, track_infos: Dict[int, object]
    ) -> None:  # วาดกล่องวัตถุรองตาม track
        for tid, info in track_infos.items():
            for obj in getattr(info, "object_boxes", []):
                name = obj["name"]
                x1, y1, x2, y2 = obj["xyxy"]
                conf = obj.get("confidence", 0.0)
                color = self._color_for_object(name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(
                    frame,
                    f"{name} {conf:.2f}",
                    (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: Detections,
        labels: List[str],
        track_infos: Dict[int, object],
    ) -> np.ndarray:  # รวมทุกการวาด
        annotated = self.box_annotator.annotate(
            frame=frame, detections=detections, labels=labels
        )  # วาดกรอบ+label หลัก
        self.draw_object_detections(annotated, track_infos)  # วาดกล่องวัตถุรอง
        self.draw_centers(annotated, track_infos)  # วาด center + trajectory
        return annotated  # คืนเฟรมที่วาดแล้ว
