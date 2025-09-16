"""CLI argument parsing for the tracking application.

Goals of this module:
        * Provide a clean, well grouped help output
        * Centralize default values (minimize magic numbers spread across code)
        * Remain lightweight – transformation to higher-level objects happens in
            `run.py` so that importing this module has no side effects.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import config


@dataclass
class Defaults:
    model: Path = Path(config.MODEL_BODY_PATH)
    source: Path = Path(config.SOURCE_VIDEO_PATH)
    target: Path = Path(config.TARGET_VIDEO_PATH)
    target_metadata: Path = Path(config.TARGET_METADATA_PATH)
    class_ids: Sequence[int] = tuple(config.MODEL_BODY_CLASS_TARGET)
    line_start = (50, 1500)
    line_end = (3840 - 50, 1500)
    max_frames: int = config.MAX_FRAMES
    track_thresh: float = config.TRACK_THRESHOLD
    track_buffer: int = config.TRACK_BUFFER
    match_thresh: float = config.MATCH_THRESH
    aspect_ratio_thresh: float = config.ASPECT_RATIO_THRESHOLD
    min_box_area: float = config.MIN_BOX_AREA
    mot20: bool = config.MOT20
    log_level: str = "INFO"


def build_arg_parser() -> argparse.ArgumentParser:
    d = Defaults()
    parser = argparse.ArgumentParser(
        description="YOLO + BYTETrack video tracking with line counting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core paths & model
    group_core = parser.add_argument_group("Core IO & Model")
    group_core.add_argument(
        "--model", type=Path, default=d.model, help="YOLO model .pt file"
    )
    group_core.add_argument(
        "--source", type=Path, default=d.source, help="Input video path"
    )
    group_core.add_argument(
        "--target", type=Path, default=d.target, help="Output (written) video path"
    )
    group_core.add_argument(
        "--target_metadata",
        type=Path,
        default=d.target_metadata,
        help="Output (written) metadata path",
    )
    group_core.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=list(d.class_ids),
        help="Class IDs to retain after detection (space separated)",
    )

    # Geometry / counting
    group_geo = parser.add_argument_group("Counting Line")
    group_geo.add_argument(
        "--line-start",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=list(d.line_start),
        help="Line start coordinate",
    )
    group_geo.add_argument(
        "--line-end",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=list(d.line_end),
        help="Line end coordinate",
    )

    # Performance / limits
    group_perf = parser.add_argument_group("Performance")
    group_perf.add_argument(
        "--max-frames",
        type=int,
        default=d.max_frames,
        help="Process only first N frames (<=0 for full video)",
    )
    group_perf.add_argument(
        "--no-progress", action="store_true", help="Hide progress bar (tqdm)"
    )

    # Tracker tuning
    group_trk = parser.add_argument_group("BYTETrack Parameters")
    group_trk.add_argument(
        "--track-thresh",
        type=float,
        default=d.track_thresh,
        help="Association confidence threshold",
    )
    group_trk.add_argument(
        "--track-buffer",
        type=int,
        default=d.track_buffer,
        help="Frames to keep lost tracks",
    )
    group_trk.add_argument(
        "--match-thresh", type=float, default=d.match_thresh, help="IoU match threshold"
    )
    group_trk.add_argument(
        "--aspect-ratio-thresh",
        type=float,
        default=d.aspect_ratio_thresh,
        help="Reject boxes with h/w greater than this",
    )
    group_trk.add_argument(
        "--min-box-area",
        type=float,
        default=d.min_box_area,
        help="Reject boxes with pixel area below this",
    )
    group_trk.add_argument(
        "--mot20",
        action="store_true",
        default=d.mot20,
        help="Enable MOT20 specific settings (overrides some heuristics)",
    )

    # Logging
    group_log = parser.add_argument_group("Logging")
    group_log.add_argument(
        "--log-level",
        default=d.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging verbosity",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = build_arg_parser()
    return parser.parse_args(argv)
