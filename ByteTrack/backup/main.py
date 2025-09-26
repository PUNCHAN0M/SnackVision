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
from tracking_helpers import FrameProcessor, TrackingVisualizer

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
    model_person_path: Path
    model_object_path: Path
    source_video_path: Path
    target_video_path: Path
    class_ids: Sequence[int]
    max_frames: Optional[int]
    track_thresh: float
    track_buffer: int
    match_thresh: float
    aspect_ratio_thresh: float
    min_box_area: float
    mot20: bool
    progress: bool
    model_person_conf_threshold: float = 0.25
    model_object_conf_threshold: float = 0.25

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
    @dataclass
    class TrackInfo:
        last_xyxy: list[float]
        center_history: list[tuple[float, float]] = field(default_factory=list)
        hold_inhand: list[str] = field(default_factory=list)
        # list of entries: {'name': str, 'index': int (1-based within that name), 'count': int}
        object_classes: list[dict] = field(default_factory=list)
        # each: {'name': str, 'xyxy': tuple[int,int,int,int], 'confidence': float}
        object_boxes: list[dict] = field(default_factory=list)

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._model_person: Optional[YOLO] = None
        self._model_object: Optional[YOLO] = None
        self._byte_tracker: Optional[BYTETracker] = None
        self._class_names: Optional[dict] = None
        self._box_annotator: Optional[BoxAnnotator] = None
        self._track_infos: dict[int, VideoTracker.TrackInfo] = {}
        self._finished_track_infos: dict[int, VideoTracker.TrackInfo] = {}
        # helper components
        self._frame_processor = FrameProcessor(self)
        self._visualizer = TrackingVisualizer()

    # -------------------- Lazy properties --------------------
    @property
    def model_person(self) -> YOLO:
        if self._model_person is None:
            LOG.info("Loading person model: %s", self.cfg.model_person_path)
            self._model_person = YOLO(str(self.cfg.model_person_path))
            try:
                self._model_person.fuse()  # speed optimization
            except Exception:
                pass
            self._class_names = self._model_person.model.names
            LOG.debug("Loaded person model with %d classes", len(self._class_names))
        return self._model_person

    @property
    def model_object(self) -> YOLO:
        if self._model_object is None:
            path = self.cfg.model_object_path
            if not path.exists():
                raise FileNotFoundError(f"Secondary object model not found: {path}")
            LOG.info("Loading object model: %s", path)
            self._model_object = YOLO(str(path))
            try:
                self._model_object.fuse()  # speed optimization
            except Exception:
                pass
        return self._model_object

    # ==================== BYTETracker Setup ====================
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
                color=ColorPalette(), thickness=2, text_thickness=1, text_scale=1
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

    # -------------------- Secondary object classification --------------------
    def classify_objects_in_tracks(
        self, frame: np.ndarray, detections: Detections
    ) -> None:
        """Detect objects on full frame and assign to person track with highest area overlap (no ties)."""
        # Reset object boxes
        for info in self._track_infos.values():
            info.object_boxes = []
        if not self._track_infos:
            return
        try:
            obj_results = self.model_object(
                frame, verbose=False, conf=self.cfg.model_object_conf_threshold
            )
        except FileNotFoundError:
            return
        except Exception as e:  # noqa: BLE001
            LOG.warning("Object model inference failed: %s", e)
            return
        if not obj_results:
            return
        res = obj_results[0]
        names = getattr(res, "names", None)
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return
        cls_tensor = getattr(boxes, "cls", None)
        xyxy_tensor = getattr(boxes, "xyxy", None)
        conf_tensor = getattr(boxes, "conf", None)
        if cls_tensor is None or xyxy_tensor is None:
            return
        cls_ids = cls_tensor.cpu().numpy().astype(int)
        all_xyxy = xyxy_tensor.cpu().numpy()
        confidences = (
            conf_tensor.cpu().numpy()
            if conf_tensor is not None
            else np.zeros(len(cls_ids))
        )
        # Map current track boxes
        track_box_map: dict[int, np.ndarray] = {}
        for xyxy, conf, cid, tid in detections:
            if tid is None:
                continue
            track_box_map[tid] = xyxy
        # Assign objects
        for cid, obj_box, conf_val in zip(cls_ids, all_xyxy, confidences):
            x1o, y1o, x2o, y2o = obj_box
            obj_area = max(0.0, (x2o - x1o)) * max(0.0, (y2o - y1o))
            if obj_area <= 0:
                continue
            ratios: list[tuple[int, float]] = []
            for tid, pbox in track_box_map.items():
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
            ratios.sort(key=lambda x: x[1], reverse=True)
            if ratios[0][1] <= 0:
                continue
            # tie check
            if len(ratios) > 1 and abs(ratios[0][1] - ratios[1][1]) < 1e-6:
                continue
            best_tid = ratios[0][0]
            if best_tid not in self._track_infos:
                continue
            info = self._track_infos[best_tid]
            cname = str(names[cid]) if names and cid in names else str(cid)
            # record object box
            info.object_boxes.append(
                {
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
                existing_sorted[0]["count"] += 1

    # Internal pipeline / visualization methods moved to helper classes

    # -------------------- Core logic --------------------
    def run(self) -> None:
        """Execute tracking over the configured video."""
        if not self.cfg.source_video_path.exists():
            raise FileNotFoundError(
                f"Source video not found: {self.cfg.source_video_path}"
            )
        self.cfg.target_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_info = VideoInfo.from_video_path(str(self.cfg.source_video_path))
        frames: Iterable[np.ndarray] = get_video_frames_generator(
            str(self.cfg.source_video_path)
        )
        total_frames = getattr(video_info, "total_frames", None)
        if self.cfg.progress:
            frames = tqdm(frames, desc="Tracking", total=total_frames)

        with VideoSink(str(self.cfg.target_video_path), video_info) as sink:
            for frame_index, frame in enumerate(frames):
                if self.cfg.max_frames and frame_index >= self.cfg.max_frames:
                    break

                detections = self._frame_processor.detect_persons(frame)
                if detections is None:
                    sink.write_frame(frame)
                    continue

                self._frame_processor.assign_track_ids(detections, frame.shape)
                self.update_track_infos(detections)
                self.classify_objects_in_tracks(frame, detections)
                labels = self._frame_processor.build_labels(detections)
                annotated = self._visualizer.annotate_frame(
                    frame, detections, labels, self._track_infos
                )
                sink.write_frame(annotated)

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
        progress=not args.no_progress,
        model_person_conf_threshold=args.model_person_confidence_threshold,
        model_object_conf_threshold=args.model_object_confidence_threshold,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_cli(argv)
    LOG.debug("Final config: %s", cfg)
    tracker = VideoTracker(cfg)
    tracker.run()


if __name__ == "__main__":
    main()
