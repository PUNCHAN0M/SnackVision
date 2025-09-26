"""Entry point for running YOLO + BYTETrack video tracking with counting.

This module refactors the original script-style implementation into a
clean, reusable, testable class (`VideoTracker`). A command line interface
is provided under the classic ``if __name__ == "__main__"`` guard.

Example (defaults from `config.py`):
    python run.py --max-frames 200

Override model & paths:
    python run.py --model models/yolov11n.pt --source videos/person.avi \
        --target videos_result/out.avi --class-ids 0 2 5

All heavy objects (model, trackers) are instantiated lazily when `run()`
is called so that unit tests can import this file without side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
import cv2
from cli_args import parse_args as raw_parse_args
from onemetric.cv.utils.iou import box_iou_batch
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.video.source import get_video_frames_generator

from yolox.tracker.byte_tracker import BYTETracker, STrack

LOG = logging.getLogger(__name__)


# ค่า config BYTES_TRACK
@dataclass(frozen=True)
class BYTETrackerArgs:
    """
    Thin container passed into underlying BYTETracker.
    Defaults removed to avoid duplication; values always sourced from
    a single place (TrackerConfig / CLI). This keeps configuration DRY.
    """

    track_thresh: float
    track_buffer: int
    match_thresh: float
    aspect_ratio_thresh: float
    min_box_area: float
    mot20: bool


# -------------------- Main classes --------------------
@dataclass
class TrackerConfig:
    """
    High-level configuration for video tracking job.
    Acts as the single source of truth for all tracking & counting parameters.
    """

    model_body: Path
    model_object: Path
    source_video: Path
    target_video: Path
    class_ids: Sequence[int]
    max_frames: Optional[int]
    track_thresh: float
    track_buffer: int
    match_thresh: float
    aspect_ratio_thresh: float
    min_box_area: float
    mot20: bool
    progress: bool

    def to_byte_args(self) -> BYTETrackerArgs:
        return BYTETrackerArgs(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            aspect_ratio_thresh=self.aspect_ratio_thresh,
            min_box_area=self.min_box_area,
            mot20=self.mot20,
        )


class VideoTracker:
    """Encapsulates model loading, tracking loop, and annotations.

    Usage:
        tracker = VideoTracker(config)
        tracker.run()
    """

    @dataclass
    class TrackInfo:
        last_xyxy: list[float]
        center_history: list[tuple[float, float]] = field(default_factory=list)
        hold_inhand: list[str] = field(default_factory=list)
        object_classes: set[str] = field(default_factory=set)

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._model_body: Optional[YOLO] = None
        self._model_object: Optional[YOLO] = None
        self._byte_tracker: Optional[BYTETracker] = None
        self._class_names: Optional[dict] = None
        self._box_annotator: Optional[BoxAnnotator] = None
        # Per-track info storage
        self._track_infos: dict[int, VideoTracker.TrackInfo] = {}
        self._finished_track_infos: dict[int, VideoTracker.TrackInfo] = {}

    # -------------------- Lazy properties --------------------
    @property
    def model_body(self) -> YOLO:
        if self._model_body is None:
            LOG.info("Loading body model: %s", self.cfg.model_body)
            self._model_body = YOLO(str(self.cfg.model_body))
            try:
                self._model_body.fuse()  # speed optimization
            except Exception:  # noqa: BLE001
                pass
            self._class_names = self._model_body.model.names  # type: ignore[attr-defined]
            LOG.debug("Loaded body model with %d classes", len(self._class_names))
        return self._model_body

    @property
    def model_object(self) -> YOLO:
        if self._model_object is None:
            path = self.cfg.model_object
            if not path.exists():
                raise FileNotFoundError(f"Secondary object model not found: {path}")
            LOG.info("Loading object model: %s", path)
            self._model_object = YOLO(str(path))
            try:
                self._model_object.fuse()
            except Exception:  # noqa: BLE001 - fuse may not exist
                pass
        return self._model_object

    @property
    def byte_tracker(self) -> BYTETracker:
        if self._byte_tracker is None:
            args = self.cfg.to_byte_args()
            self._byte_tracker = BYTETracker(args)
            LOG.debug("Initializing BYTETracker: %s", args)
        return self._byte_tracker

    @property
    def box_annotator(self) -> BoxAnnotator:
        if self._box_annotator is None:
            self._box_annotator = BoxAnnotator(
                color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1
            )
        return self._box_annotator

    # -------------------- Static helpers --------------------
    @staticmethod
    def detections2boxes(detections: Detections) -> np.ndarray:
        return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))

    @staticmethod
    def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
        return np.array([track.tlbr for track in tracks], dtype=float)

    @classmethod
    def match_detections_with_tracks(
        cls, detections: Detections, tracks: List[STrack]
    ) -> List[Optional[int]]:
        if len(detections) == 0 or len(tracks) == 0:
            return []
        tracks_boxes = cls.tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, detections.xyxy)
        track2detection = np.argmax(iou, axis=1)
        tracker_ids: List[Optional[int]] = [None] * len(detections)
        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                tracker_ids[detection_index] = tracks[tracker_index].track_id
        return tracker_ids

    # -------------------- Track info management --------------------
    def update_track_infos(self, detections: Detections) -> None:
        """Update/create TrackInfo for current detections and finalize disappeared ones."""
        current_ids: set[int] = set()
        for xyxy, conf, cid, tid in detections:
            if tid is None:
                continue
            current_ids.add(tid)
            x1, y1, x2, y2 = xyxy.tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if tid not in self._track_infos:
                self._track_infos[tid] = VideoTracker.TrackInfo(
                    last_xyxy=[x1, y1, x2, y2]
                )
            info = self._track_infos[tid]
            info.last_xyxy = [x1, y1, x2, y2]
            info.center_history.append((cx, cy))

        # Move tracks that disappeared this frame to finished
        disappeared = [tid for tid in self._track_infos if tid not in current_ids]
        for tid in disappeared:
            self._finished_track_infos[tid] = self._track_infos.pop(tid)

    def get_active_tracks(self) -> dict[int, "VideoTracker.TrackInfo"]:
        return self._track_infos

    def get_finished_tracks(self) -> dict[int, "VideoTracker.TrackInfo"]:
        return self._finished_track_infos

    # -------------------- Visualization helpers --------------------
    def draw_centers(self, frame: np.ndarray, tail: int = 15) -> None:
        """Draw current centers and short trajectory for each active track.

        tail: number of recent center points to draw per track.
        """
        # Detect if frame is grayscale (single channel)
        is_gray = len(frame.shape) == 2 or (
            len(frame.shape) == 3 and frame.shape[2] == 1
        )
        for tid, info in self._track_infos.items():
            if not info.center_history:
                continue
            # Choose color deterministic by tid (ensure ints)
            b = int((37 * tid) % 255)
            g = int((17 * tid) % 255)
            r = int((97 * tid) % 255)
            color = (b, g, r) if not is_gray else int(0.299 * r + 0.587 * g + 0.114 * b)
            # Draw trajectory
            pts = info.center_history[-tail:]
            for i in range(1, len(pts)):
                x1, y1 = map(int, pts[i - 1])
                x2, y2 = map(int, pts[i])
                thickness = 2
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
            # Draw current center point
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

    def filter_min_height(
        self, detections: Detections, frame: np.ndarray, ratio: float = 0.5
    ) -> None:
        """In-place filter removing detections whose bbox height < ratio * frame_height.

        If all detections removed, function just leaves detections empty.
        """
        if len(detections) == 0:
            return
        frame_h = frame.shape[0]
        xyxy = detections.xyxy
        heights = xyxy[:, 3] - xyxy[:, 1]
        mask = heights >= ratio * frame_h
        detections.filter(mask, inplace=True)

    # -------------------- Secondary object classification --------------------
    def classify_objects_in_tracks(
        self, frame: np.ndarray, detections: Detections
    ) -> None:
        """Run secondary object model on each tracked detection crop and store class names.

        Strategy:
            - For each detection with tracker_id, crop the region (with slight padding)
            - Run the secondary model (batch for efficiency when possible)
            - Collect predicted class names over time in TrackInfo.object_classes

        Notes:
            - If the object model predicts multiple boxes, we add all its class names.
            - We ignore extremely small crops (w or h < 5 px) to avoid wasting inference.
        """
        if len(detections) == 0:
            return
        # Prepare crops and track ids
        crops = []
        track_ids: list[int] = []
        h, w = frame.shape[:2]
        for xyxy, conf, cid, tid in detections:
            if tid is None:
                continue
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            # Add small padding
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = frame[y1:y2, x1:x2]
            crops.append(crop)
            track_ids.append(tid)
        if not crops:
            return
        try:
            results = self.model_object(crops, verbose=False)
        except FileNotFoundError:
            # Already logged earlier; silently skip
            return
        except Exception as e:  # noqa: BLE001
            LOG.warning("Secondary model inference failed: %s", e)
            return
        # Iterate over results aligning with track_ids
        for res, tid in zip(results, track_ids):
            try:
                names = res.names  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                names = None
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            cls_tensor = getattr(boxes, "cls", None)
            if cls_tensor is None:
                continue
            cls_ids = cls_tensor.cpu().numpy().astype(int)
            if tid not in self._track_infos:
                continue
            info = self._track_infos[tid]
            for cid in cls_ids:
                if names and cid in names:
                    info.object_classes.add(str(names[cid]))
                else:
                    info.object_classes.add(str(cid))

    # -------------------- Core logic --------------------
    def run(self) -> None:
        """Execute tracking over the configured video."""
        if not self.cfg.source_video.exists():
            raise FileNotFoundError(f"Source video not found: {self.cfg.source_video}")
        self.cfg.target_video.parent.mkdir(parents=True, exist_ok=True)

        video_info = VideoInfo.from_video_path(str(self.cfg.source_video))
        frames: Iterable[np.ndarray] = get_video_frames_generator(
            str(self.cfg.source_video)
        )
        if self.cfg.progress:
            # tqdm can infer length lazily; omit explicit total to remove extra variable bookkeeping
            frames = tqdm(frames, desc="Tracking")

        with VideoSink(str(self.cfg.target_video), video_info) as sink:
            for frame_index, frame in enumerate(frames):
                if self.cfg.max_frames and frame_index >= self.cfg.max_frames:
                    break

                # Model inference
                result = self.model_body(frame, verbose=False)
                boxes = result[0].boxes
                detections = Detections(
                    xyxy=boxes.xyxy.cpu().numpy(),
                    confidence=boxes.conf.cpu().numpy(),
                    class_id=boxes.cls.cpu().numpy().astype(int),
                )
                mask = np.isin(detections.class_id, list(self.cfg.class_ids))
                detections.filter(mask=mask, inplace=True)
                if len(detections) == 0:
                    sink.write_frame(frame)
                    continue

                # Filter out boxes whose height < 50% of frame height
                self.filter_min_height(detections, frame, ratio=0.5)
                if len(detections) == 0:
                    sink.write_frame(frame)
                    continue
                # Tracking
                tracks = self.byte_tracker.update(
                    output_results=self.detections2boxes(detections),
                    img_info=frame.shape,
                    img_size=frame.shape,
                )
                tracker_id = self.match_detections_with_tracks(detections, tracks)
                detections.tracker_id = np.array(tracker_id)

                # Filter out detections without trackers
                if detections.tracker_id is not None:
                    mask = np.array(
                        [tid is not None for tid in detections.tracker_id], dtype=bool
                    )
                    detections.filter(mask=mask, inplace=True)

                labels = []
                for xyxy, conf, cid, tid in detections:
                    if tid is None:
                        labels.append(f"#? {conf:0.2f}")
                        continue
                    info = self._track_infos.get(tid)
                    if info and info.object_classes:
                        objs = ",".join(sorted(list(info.object_classes))[:3])
                        label = f"#{tid} {conf:0.2f} [{objs}]"
                    else:
                        label = f"#{tid} {conf:0.2f}"
                    labels.append(label)

                # Update track info (centers & history)
                self.update_track_infos(detections)

                # Secondary object detection per track (crop-based)
                self.classify_objects_in_tracks(frame, detections)

                # Annotation
                frame = self.box_annotator.annotate(
                    frame=frame, detections=detections, labels=labels
                )
                # Draw centers & short trajectories
                self.draw_centers(frame)
                sink.write_frame(frame)

        LOG.info(
            "Completed tracking. Active=%d Finished=%d",
            len(self._track_infos),
            len(self._finished_track_infos),
        )


def parse_cli(argv: Optional[Sequence[str]] = None) -> TrackerConfig:
    """Parse raw CLI args and build a TrackerConfig instance."""
    args = raw_parse_args(argv)
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return TrackerConfig(
    model_body=args.model_body,
    model_object=args.model_object,
        source_video=args.source,
        target_video=args.target,
        class_ids=args.class_ids,
        max_frames=max_frames,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        aspect_ratio_thresh=args.aspect_ratio_thresh,
        min_box_area=args.min_box_area,
        mot20=args.mot20,
        progress=not args.no_progress,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_cli(argv)
    LOG.debug("Final config: %s", cfg)
    tracker = VideoTracker(cfg)
    tracker.run()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
