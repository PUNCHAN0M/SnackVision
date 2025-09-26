"""Helper classes for frame processing and visualization.

Separated from `main.py` so that `VideoTracker` concentrates on orchestration only.
"""

from __future__ import annotations

from typing import Optional, List, Dict, TYPE_CHECKING

import cv2
import numpy as np
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette


if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .main import VideoTracker


class FrameProcessor:
    """Encapsulates detection, tracker assignment, label building, and filters."""

    def __init__(self, tracker: "VideoTracker"):
        self.tracker = tracker

    def detect_persons(self, frame: np.ndarray) -> Optional[Detections]:
        result = self.tracker.model_person(
            frame, verbose=False, conf=self.tracker.cfg.model_person_conf_threshold
        )
        boxes = result[0].boxes
        detections = Detections(
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int),
        )
        detections.filter(
            mask=np.isin(detections.class_id, list(self.tracker.cfg.class_ids)),
            inplace=True,
        )
        if len(detections) == 0:
            return None
        self.filter_min_height(detections, frame, ratio=0.0)
        if len(detections) == 0:
            return None
        return detections

    def assign_track_ids(
        self, detections: Detections, frame_shape: tuple[int, ...]
    ) -> None:
        tracks = self.tracker.byte_tracker.update(
            output_results=self.tracker.detections2boxes(detections),
            img_info=frame_shape,
            img_size=frame_shape,
        )
        tracker_id = self.tracker.match_detections_with_tracks(detections, tracks)
        detections.tracker_id = np.array(tracker_id)
        if detections.tracker_id is not None:
            detections.filter(
                mask=np.array(
                    [tid is not None for tid in detections.tracker_id], dtype=bool
                ),
                inplace=True,
            )

    @staticmethod
    def filter_min_height(
        detections: Detections, frame: np.ndarray, ratio: float = 0.5
    ) -> None:
        if len(detections) == 0:
            return
        frame_h = frame.shape[0]
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        mask = heights >= ratio * frame_h
        detections.filter(mask, inplace=True)

    def build_labels(self, detections: Detections) -> list[str]:
        labels: list[str] = []
        track_infos = self.tracker._track_infos  # internal access intentional
        for xyxy, conf, cid, tid in detections:
            if tid is None:
                labels.append(f"#? {conf:0.2f}")
                continue
            info = track_infos.get(tid)
            if info and info.object_classes:
                ordered = sorted(
                    info.object_classes, key=lambda e: (e["name"], e["index"])
                )
                parts = [f"{e['name']}({e['index']}:{e['count']})" for e in ordered[:4]]
                labels.append(f"#{tid} {conf:0.2f} [{','.join(parts)}]")
            else:
                labels.append(f"#{tid} {conf:0.2f}")
        return labels


class TrackingVisualizer:
    """Handles drawing of detection boxes, object boxes, centers and trajectories."""

    def __init__(
        self,
        palette: Optional[ColorPalette] = None,
        thickness: int = 2,
        text_thickness: int = 1,
        text_scale: float = 1.0,
    ):
        self.palette = palette or ColorPalette()
        self.box_annotator = BoxAnnotator(
            color=self.palette,
            thickness=thickness,
            text_thickness=text_thickness,
            text_scale=text_scale,
        )
        self._object_color_map: Dict[str, tuple[int, int, int]] = {}

    def _color_for_object(self, name: str) -> tuple[int, int, int]:
        if name in self._object_color_map:
            return self._object_color_map[name]
        hv = sum(ord(c) for c in name)
        color = ((hv * 3) % 200 + 30, (hv * 7) % 200 + 30, (hv * 11) % 200 + 30)
        self._object_color_map[name] = color
        return color

    def draw_centers(
        self, frame: np.ndarray, track_infos: Dict[int, object], tail: int = 15
    ) -> None:
        is_gray = len(frame.shape) == 2 or (
            len(frame.shape) == 3 and frame.shape[2] == 1
        )
        for tid, info in track_infos.items():
            history = getattr(info, "center_history", [])
            if not history:
                continue
            b = int((37 * tid) % 255)
            g = int((17 * tid) % 255)
            r = int((97 * tid) % 255)
            color = (b, g, r) if not is_gray else int(0.299 * r + 0.587 * g + 0.114 * b)
            pts = history[-tail:]
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
    ) -> None:
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
    ) -> np.ndarray:
        annotated = self.box_annotator.annotate(
            frame=frame, detections=detections, labels=labels
        )
        self.draw_object_detections(annotated, track_infos)
        self.draw_centers(annotated, track_infos)
        return annotated
