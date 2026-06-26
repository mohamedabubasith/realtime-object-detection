"""Pydantic request/response models and the stats payload shape shared with the
frontend over WebSocket."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    upload = "upload"     # a previously uploaded file (source = file_id)
    url = "url"           # YouTube / HLS / any page yt-dlp can resolve
    rtsp = "rtsp"         # rtsp:// live camera (source = full url)
    webcam = "webcam"     # local camera (source = device index, e.g. "0")


class CreateSessionRequest(BaseModel):
    source_type: SourceType
    source: str = Field(..., description="file_id, URL, rtsp url, or webcam index")
    label: Optional[str] = None
    # Optional per-session overrides; fall back to global config when omitted.
    conf_threshold: Optional[float] = None
    target_classes: Optional[List[str]] = None
    loop: bool = False  # loop uploaded video files when they reach the end


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int


class Detection(BaseModel):
    cls: str
    conf: float
    box: List[float]  # [x1, y1, x2, y2] in pixels of the (possibly downscaled) frame
    id: Optional[int] = None  # stable per-object track id (null until tracked)


class LogEntry(BaseModel):
    t: float           # epoch seconds
    message: str


# --- interactive geometry (counting lines / occupancy zones) -----------------
# Coordinates are NORMALIZED floats in [0,1] RELATIVE TO THE VIDEO FRAME (the
# actual video pixels, not the browser element). Sent frontend->backend via
# PATCH /api/sessions/{sid}/settings and echoed back inside SessionStats so the
# overlay can re-sync after a reload.

class Line(BaseModel):
    id: str
    name: str
    points: List[List[float]]  # exactly two points: [[x1,y1],[x2,y2]]


class Zone(BaseModel):
    id: str
    name: str
    points: List[List[float]]  # polygon, >= 3 points: [[x,y],[x,y],...]


class LineCount(BaseModel):
    id: str
    name: str
    in_: int = Field(0, alias="in")        # crossings side<0 -> side>0
    out: int = 0                           # crossings side>0 -> side<0
    total: int = 0                         # in + out
    by_class: Dict[str, int] = {}          # total crossings per class name

    class Config:
        populate_by_name = True            # allow constructing via in_ or "in"


class ZoneOccupancy(BaseModel):
    id: str
    name: str
    count: int = 0                         # tracks currently inside
    peak: int = 0                          # max simultaneous occupancy seen
    avg_dwell_s: float = 0.0               # mean time tracks spent inside


class SessionStats(BaseModel):
    session_id: str
    status: str                       # starting | running | finished | error | stopped
    source_label: str
    source_type: SourceType
    error: Optional[str] = None

    current_count: int = 0            # detected objects in the latest frame
    smoothed_count: float = 0.0       # moving-average count (reduces flicker)
    max_count: int = 0
    total_detections: int = 0         # cumulative across all processed frames

    last_confidence: float = 0.0
    avg_confidence: float = 0.0

    fps: float = 0.0
    processed_frames: int = 0
    elapsed_seconds: float = 0.0

    width: int = 0
    height: int = 0

    conf_threshold: float = 0.0
    target_classes: List[str] = []

    detections: List[Detection] = []  # current frame's boxes

    # --- tracking + interactive counting (additive) ---
    unique_total: int = 0             # cumulative distinct tracks created
    unique_active: int = 0            # currently active tracks
    unique_by_class: Dict[str, int] = {}  # cumulative distinct tracks by class

    line_counts: List[LineCount] = []      # directional in/out per line
    zone_occupancy: List[ZoneOccupancy] = []  # live occupancy + dwell per zone

    lines: List[Line] = []            # echo of configured counting lines
    zones: List[Zone] = []            # echo of configured occupancy zones

    log: List[LogEntry] = []          # recent events (tail)


class SessionInfo(BaseModel):
    session_id: str
    status: str
    source_label: str
    source_type: SourceType
    created_at: float


class UpdateSettingsRequest(BaseModel):
    conf_threshold: Optional[float] = None
    target_classes: Optional[List[str]] = None
    process_every_n: Optional[int] = None
    # Interactive geometry (normalized [0,1] coords relative to the video frame).
    # Omitted (None) leaves the current geometry untouched; an empty list clears.
    lines: Optional[List[Line]] = None
    zones: Optional[List[Zone]] = None


class ConfigResponse(BaseModel):
    default_conf_threshold: float
    default_target_classes: List[str]
    available_classes: List[str]
    imgsz: int
    process_every_n: int
    max_sessions: int
    model_path: str
