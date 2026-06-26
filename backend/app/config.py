"""Central configuration.

Everything tunable lives here and can be overridden with environment variables
(or a .env file). The defaults are chosen to run well on a weak 2-core / 4GB
CPU-only machine while still detecting objects reliably. The detector handles
all 80 COCO classes; ``target_classes`` just selects which to count (default:
``car``).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # .../backend


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    model_dir: Path = BASE_DIR / "models"
    upload_dir: Path = BASE_DIR / "uploads"

    # ------------------------------------------------------------------ model
    # Default: YOLO26n (Jan 2026) exported to ONNX — best accuracy AND fastest on
    # CPU (NMS-free head + ONNX Runtime). run.sh auto-creates models/yolo26n.onnx
    # on first launch; if it's missing we fall back to the auto-downloaded
    # yolo26n.pt (see resolved_model_path). Set "yolo11n.pt"/"yolo11n.onnx" for
    # the older fully-mature model. Requires ultralytics>=8.4.
    model_path: str = "yolo26n.onnx"

    # Which COCO classes to count by default. COCO names include: car, truck,
    # bus, motorcycle, bicycle, person, ... "car" is the primary use case; add
    # more here (or via TARGET_CLASSES) to extend — no code changes needed.
    target_classes: List[str] = ["car"]

    # ------------------------------------------------------------ detection
    conf_threshold: float = 0.35      # min confidence to count a detection
    iou_threshold: float = 0.45       # NMS IoU
    imgsz: int = 416                  # inference size; 320 = faster, 640 = more
                                      # accurate. Must match the exported ONNX
                                      # (run.sh exports at this size).
    max_det: int = 100                # max detections per frame
    min_box_area_ratio: float = 0.0   # ignore boxes smaller than this fraction
                                      # of the frame area (false-positive guard)

    # ------------------------------------------------------------ performance
    # Run detection on every Nth frame; in-between frames reuse the last boxes.
    # On a 2-core CPU, 2-3 keeps things smooth.
    process_every_n: int = 2
    # Jitter buffer for LIVE streams (YouTube/HLS/RTSP). A background reader
    # thread pre-buffers this many seconds of frames so the video doesn't freeze
    # while the next HLS segment is being fetched. Higher = smoother but more
    # latency behind "live" (and more RAM). 0 disables buffering.
    live_buffer_seconds: float = 2.0
    # Downscale incoming frames to at most this width before processing/encoding.
    max_frame_width: int = 960
    # MJPEG delivery cap (frames/sec sent to the browser).
    stream_fps: int = 15
    jpeg_quality: int = 70
    # Max simultaneous detection sessions. A 2-core box really only handles 1
    # heavy stream well; raise with care.
    max_sessions: int = 2
    # Number of OpenCV / ONNX / Torch threads. Keep small on a 2-core machine.
    num_threads: int = 2

    # ------------------------------------------------------------------ server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]

    # How many recent detection-log entries to keep per session.
    log_size: int = 100

    @field_validator("target_classes", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        """Accept comma-separated env-var strings as lists.

        e.g. ``TARGET_CLASSES="car,truck,bus"`` -> ``["car", "truck", "bus"]``.
        Non-string values (already a list) pass through unchanged.
        """
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def resolved_model_path(self) -> str:
        """Resolve the model to load, with a graceful fallback chain:

        1. An absolute path that exists.
        2. models/<name> if present (e.g. an exported models/yolo26n.onnx).
        3. If an .onnx was requested but isn't there yet, fall back to its .pt
           twin (models/<stem>.pt, else the bare downloadable <stem>.pt).
        4. Otherwise the bare name (Ultralytics auto-downloads known weights).
        """
        import logging
        log = logging.getLogger("config")

        p = Path(self.model_path)
        if p.is_absolute() and p.exists():
            return str(p)

        candidate = self.model_dir / p.name
        if candidate.exists():
            return str(candidate)

        if p.suffix == ".onnx":
            pt_local = self.model_dir / (p.stem + ".pt")
            if pt_local.exists():
                log.warning("%s not found; using %s. Run scripts/export_model.py "
                            "for faster ONNX inference.", p.name, pt_local.name)
                return str(pt_local)
            log.warning("%s not found; falling back to downloadable %s.pt "
                        "(slower). Run scripts/export_model.py to speed up.",
                        p.name, p.stem)
            return p.stem + ".pt"

        return self.model_path


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
