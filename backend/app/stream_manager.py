"""Per-session detection worker + a manager that owns all active sessions.

Each session runs ONE background thread that:
  1. reads frames from its FrameSource,
  2. runs detection on every Nth frame (reusing boxes in between),
  3. draws boxes and JPEG-encodes the annotated frame into a latest-frame buffer,
  4. keeps live stats (counts, fps, confidence, a rolling event log).

The HTTP layer just reads the latest JPEG (for MJPEG) and the latest stats (for
the WebSocket) — capture and delivery are fully decoupled, so multiple browser
tabs can watch one session cheaply.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

import cv2

from .config import Settings
from .detector import Detector, Detection
from .schemas import SourceType
from .sources import FrameSource, build_resolved_source
from .tracker import CentroidTracker

logger = logging.getLogger("stream")


def _geom_to_dict(g: Any) -> Dict[str, Any]:
    """Normalize an incoming line/zone (pydantic model OR plain dict) to a plain
    dict with id/name/points, coercing points to lists of [float,float]."""
    if hasattr(g, "model_dump"):       # pydantic v2 model
        g = g.model_dump()
    elif hasattr(g, "dict"):           # pydantic v1 model (defensive)
        g = g.dict()
    pts = [[float(p[0]), float(p[1])] for p in g.get("points", [])]
    return {"id": str(g.get("id")), "name": str(g.get("name", g.get("id"))),
            "points": pts}


def _line_side(ax: float, ay: float, bx: float, by: float,
               px: float, py: float) -> float:
    """Signed side of point P relative to the directed line A->B.

    ``side = sign((B-A) x (P-A))``. Returned as the raw cross product (its sign
    is what matters): >0 on one side, <0 on the other, 0 exactly on the line.
    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _segments_intersect(ax: float, ay: float, bx: float, by: float,
                        cx: float, cy: float, dx: float, dy: float) -> bool:
    """True if segment A-B intersects segment C-D (proper or touching).

    Used to confirm a track's centroid path (C->D) actually crosses the line
    SEGMENT A-B, not merely its infinite extension.
    """
    d1 = _line_side(cx, cy, dx, dy, ax, ay)
    d2 = _line_side(cx, cy, dx, dy, bx, by)
    d3 = _line_side(ax, ay, bx, by, cx, cy)
    d4 = _line_side(ax, ay, bx, by, dx, dy)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    # Collinear/touching edge cases: point lies on the other segment.
    def _on(px, py, qx, qy, rx, ry):  # r on segment p-q (assuming collinear)
        return (min(px, qx) <= rx <= max(px, qx)
                and min(py, qy) <= ry <= max(py, qy))
    if d1 == 0 and _on(cx, cy, dx, dy, ax, ay):
        return True
    if d2 == 0 and _on(cx, cy, dx, dy, bx, by):
        return True
    if d3 == 0 and _on(ax, ay, bx, by, cx, cy):
        return True
    if d4 == 0 and _on(ax, ay, bx, by, dx, dy):
        return True
    return False


def _point_in_polygon(px: float, py: float,
                      poly: List[List[float]]) -> bool:
    """Ray-casting point-in-polygon test (pixel coords). Polygon is a list of
    [x,y] vertices; assumed closed implicitly (last connects to first)."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        # Does a horizontal ray from P cross edge (i, j)?
        if ((yi > py) != (yj > py)):
            x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_cross:
                inside = not inside
        j = i
    return inside


def _summarize_detections(dets: List[Detection]) -> str:
    """Build a human-readable breakdown of a frame's detections by class name.

    e.g. ``[car, car, truck]`` -> ``"3 object(s): 2 car, 1 truck"``.
    """
    counts = Counter(d.cls for d in dets)
    parts = ", ".join(f"{n} {cls}" for cls, n in counts.most_common())
    return f"{len(dets)} object(s): {parts}"


class Session:
    """A single detection job and its live state.

    Owns one background worker thread that reads frames from a
    :class:`~app.sources.FrameSource`, runs detection on every Nth frame,
    annotates and JPEG-encodes the result into a latest-frame buffer, and
    maintains live stats (counts, fps, confidence, a rolling event log). All
    mutable state shared with the HTTP layer is guarded by ``_lock``.
    """

    def __init__(self, sid: str, source_type: SourceType, source: str,
                 detector: Detector, settings: Settings,
                 conf_threshold: Optional[float] = None,
                 target_classes: Optional[List[str]] = None,
                 label: Optional[str] = None, loop: bool = False) -> None:
        self.id = sid
        self.source_type = source_type
        self.source = source
        self.detector = detector
        self.settings = settings
        self.created_at = time.time()

        self.conf_threshold = conf_threshold or settings.conf_threshold
        self.target_classes = target_classes or list(settings.target_classes)
        self.process_every_n = settings.process_every_n
        self.loop = loop

        self.label = label or source
        self.status = "starting"     # starting|running|finished|error|stopped
        self.error: Optional[str] = None

        # --- live state (guarded by _lock) ---
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._detections: List[Detection] = []
        self._log: Deque[dict] = deque(maxlen=settings.log_size)

        self.width = 0
        self.height = 0
        self.current_count = 0
        self.smoothed_count = 0.0
        self.max_count = 0
        self.total_detections = 0
        self.last_confidence = 0.0
        self._conf_sum = 0.0
        self._conf_n = 0
        self.processed_frames = 0
        self.fps = 0.0
        self._start_time = time.time()
        self._fps_window: Deque[float] = deque(maxlen=30)

        # --- tracking + interactive counting (guarded by _lock) ---
        # Stable per-object ids come from this tracker; no extra inference.
        # max_age is in PROCESSED frames (we only ever update on those).
        self._tracker = CentroidTracker(max_age=30)
        # Geometry, stored in NORMALIZED [0,1] coords (relative to video frame),
        # converted to pixels lazily per frame using current width/height.
        self._lines: List[Dict[str, Any]] = []   # {id,name,points:[[x,y],[x,y]]}
        self._zones: List[Dict[str, Any]] = []    # {id,name,points:[[x,y],...]}
        # Per-line tallies: line_id -> {"in":int,"out":int,"by_class":Counter}
        self._line_counts: Dict[str, Dict[str, Any]] = {}
        # Per-zone state: zone_id -> {"count","peak","inside":set(track_ids),
        #                             "enter_times":{tid:t},"dwell_samples":[s,...]}
        self._zone_state: Dict[str, Dict[str, Any]] = {}
        # Guard so each (track_id, line_id) crossing is only counted once.
        self._counted_crossings: set = set()
        # Debounce: track_id -> {line_id: last_side_sign} (only flips after a
        # confirmed move across, suppressing 1-frame jitter on the line).
        self._line_last_side: Dict[int, Dict[str, int]] = {}

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"session-{sid}",
                                        daemon=True)

    # ----------------------------------------------------------------- control
    def start(self) -> None:
        """Launch the worker thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop at the next loop iteration."""
        self._stop.set()

    def join(self, timeout: float = 5.0) -> None:
        """Wait for the worker thread to exit (best-effort, bounded)."""
        self._thread.join(timeout=timeout)

    def update_settings(self, conf_threshold: Optional[float] = None,
                        target_classes: Optional[List[str]] = None,
                        process_every_n: Optional[int] = None,
                        lines: Optional[List[Any]] = None,
                        zones: Optional[List[Any]] = None) -> None:
        """Apply per-session overrides; omitted (None) fields are left as-is.

        ``lines``/``zones`` are geometry in NORMALIZED [0,1] coords (relative to
        the video frame). They may arrive as pydantic models or plain dicts.
        Passing an empty list clears that geometry. Whenever geometry changes,
        the per-geometry counters are reset so stale crossings don't linger.
        """
        with self._lock:
            if conf_threshold is not None:
                self.conf_threshold = float(conf_threshold)
            if target_classes is not None:
                self.target_classes = list(target_classes)
            if process_every_n is not None:
                self.process_every_n = max(1, int(process_every_n))
            if lines is not None:
                self._lines = [_geom_to_dict(g) for g in lines]
                self._reset_line_counters()
            if zones is not None:
                self._zones = [_geom_to_dict(g) for g in zones]
                self._reset_zone_counters()
        self._log_event(f"Settings updated (conf={self.conf_threshold:.2f}, "
                        f"classes={','.join(self.target_classes)})")

    def _reset_line_counters(self) -> None:
        """(Re)initialize per-line tallies for the current ``self._lines``.
        Caller must hold ``self._lock``."""
        self._line_counts = {
            ln["id"]: {"in": 0, "out": 0, "by_class": Counter()}
            for ln in self._lines
        }
        self._counted_crossings = set()
        self._line_last_side = {}

    def _reset_zone_counters(self) -> None:
        """(Re)initialize per-zone state for the current ``self._zones``.
        Caller must hold ``self._lock``."""
        self._zone_state = {
            zn["id"]: {"count": 0, "peak": 0, "inside": set(),
                       "enter_times": {}, "dwell_samples": []}
            for zn in self._zones
        }

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        """Capture -> detect -> annotate -> encode loop; runs in its own thread."""
        try:
            resolved = build_resolved_source(
                self.source_type.value, self.source, self.settings.upload_dir)
            self.label = resolved.label
            src = FrameSource(resolved, max_width=self.settings.max_frame_width,
                              loop_file=self.loop,
                              buffer_seconds=self.settings.live_buffer_seconds)
            src.open()
            self.status = "running"
            self._log_event(f"Started: {self.label}")
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            logger.exception("Failed to start session %s", self.id)
            self._log_event(f"Error: {e}")
            return

        frame_idx = 0
        last_dets: List[Detection] = []
        target_dt = 1.0 / max(1, self.settings.stream_fps)
        last_emit = 0.0

        # For LIVE sources, pace consumption to ~real-time so the jitter buffer
        # keeps a reserve to cover HLS segment-fetch stalls instead of being
        # drained instantly. Files are processed as fast as the CPU allows.
        is_live = resolved.is_live
        frame_interval = (1.0 / (src.fps_hint or 25.0)) if is_live else 0.0

        try:
            while not self._stop.is_set():
                loop_t0 = time.time()
                frame = src.read()
                if frame is None:
                    self.status = "finished"
                    self._log_event("Source finished")
                    break

                self.height, self.width = frame.shape[:2]

                # Detect on every Nth frame; reuse last boxes in between.
                if frame_idx % self.process_every_n == 0:
                    with self._lock:
                        conf = self.conf_threshold
                        classes = list(self.target_classes)
                    class_ids = self.detector.class_ids_for(classes)
                    dets = self.detector.detect(
                        frame, conf=conf, class_ids=class_ids,
                        min_box_area_ratio=self.settings.min_box_area_ratio)
                    last_dets = dets
                    self._record_detection(dets)
                frame_idx += 1

                # Annotate (a copy so we never mutate source buffers).
                annotated = frame.copy()
                Detector.annotate(annotated, last_dets)

                # Throttle JPEG encoding/delivery to stream_fps.
                now = time.time()
                if now - last_emit >= target_dt:
                    ok, buf = cv2.imencode(
                        ".jpg", annotated,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality])
                    if ok:
                        with self._lock:
                            self._latest_jpeg = buf.tobytes()
                            self._detections = last_dets
                    last_emit = now

                # Pace live playback to ~real-time (interruptible by stop()).
                if frame_interval:
                    elapsed = time.time() - loop_t0
                    if elapsed < frame_interval:
                        self._stop.wait(frame_interval - elapsed)
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            logger.exception("Session %s crashed", self.id)
            self._log_event(f"Error: {e}")
        finally:
            src.release()
            if self.status == "running":
                self.status = "stopped"
            self._log_event(f"Session {self.status}")

    def _record_detection(self, dets: List[Detection]) -> None:
        """Fold one processed frame's detections into the running stats + log.

        This is also where tracking + interactive counting happen: we stamp
        stable ids onto the detections via the centroid tracker (no extra
        inference), then run line-crossing and zone-occupancy geometry against
        the tracks' centroid motion.
        """
        count = len(dets)
        self.processed_frames += 1
        self.total_detections += count
        now = time.time()
        self._fps_window.append(now)
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                self.fps = (len(self._fps_window) - 1) / span

        with self._lock:
            # 1) Tracking: stamp .id on each detection, get active tracks.
            active_tracks = self._tracker.update(dets)

            # 2) Geometry against current pixel-space lines/zones.
            self._update_line_crossings(active_tracks)
            self._update_zone_occupancy(active_tracks, now)

            # 3) Existing aggregate stats.
            self.current_count = count
            self.max_count = max(self.max_count, count)
            # exponential moving average to smooth flicker
            alpha = 0.4
            self.smoothed_count = (alpha * count
                                   + (1 - alpha) * self.smoothed_count)
            if dets:
                top = max(d.conf for d in dets)
                self.last_confidence = top
                self._conf_sum += sum(d.conf for d in dets)
                self._conf_n += count

        if count:
            self._log_event(f"{_summarize_detections(dets)} "
                            f"(top {self.last_confidence * 100:.0f}%)")

    # ------------------------------------------------------- geometry counting
    def _update_line_crossings(self, tracks: List[Any]) -> None:
        """Count directional line crossings for this frame.

        Convention (per the data contract): for a directed line A->B,
        ``side(P)=sign((B-A) x (P-A))``. A track whose centroid moves from
        side<0 to side>0 is an "in"; side>0 to side<0 is an "out". Each
        (track,line) pair is counted at most once, the crossing is confirmed
        only if the centroid path actually intersects the line SEGMENT (not its
        infinite extension), and a small side-flip debounce suppresses jitter.

        Caller must hold ``self._lock``. Coordinates are converted from
        normalized [0,1] to pixels using the current frame width/height.
        """
        if not self._lines or not tracks:
            return
        w, h = self.width, self.height
        if w <= 0 or h <= 0:
            return  # don't know the frame size yet; skip until first frame sized

        for tr in tracks:
            # Need real motion (a previous centroid distinct from current).
            px, py = tr.prev_centroid
            cx, cy = tr.centroid
            if px == cx and py == cy:
                continue
            sides = self._line_last_side.setdefault(tr.id, {})
            for ln in self._lines:
                pts = ln["points"]
                if len(pts) < 2:
                    continue
                ax, ay = pts[0][0] * w, pts[0][1] * h
                bx, by = pts[1][0] * w, pts[1][1] * h

                cur = _line_side(ax, ay, bx, by, cx, cy)
                cur_sign = 1 if cur > 0 else (-1 if cur < 0 else 0)
                prev_sign = sides.get(ln["id"], 0)

                # Debounce: only act on a confirmed sign change between two
                # non-zero sides. Record the latest non-zero side either way.
                if cur_sign != 0:
                    sides[ln["id"]] = cur_sign

                if prev_sign == 0 or cur_sign == 0 or cur_sign == prev_sign:
                    continue  # no confirmed flip this frame

                key = (tr.id, ln["id"])
                if key in self._counted_crossings:
                    continue  # already counted this (track,line)

                # Segment-bounded check: the centroid path must actually cross
                # the finite segment A-B, not just its infinite extension.
                if not _segments_intersect(ax, ay, bx, by, px, py, cx, cy):
                    continue

                tally = self._line_counts.get(ln["id"])
                if tally is None:
                    continue
                if prev_sign < 0 and cur_sign > 0:
                    tally["in"] += 1
                else:  # prev_sign > 0 and cur_sign < 0
                    tally["out"] += 1
                tally["by_class"][tr.cls] += 1
                self._counted_crossings.add(key)

    def _update_zone_occupancy(self, tracks: List[Any], now: float) -> None:
        """Update live occupancy + dwell for each zone.

        A track is "inside" a zone when its centroid passes the ray-casting
        point-in-polygon test. We track enter/exit transitions per (track,zone):
        on enter we stamp the time; on exit we record the dwell duration. Peak
        occupancy is the max simultaneous count ever seen.

        Caller must hold ``self._lock``. Coords are normalized [0,1] -> pixels.
        """
        if not self._zones:
            return
        w, h = self.width, self.height
        if w <= 0 or h <= 0:
            return

        # Centroid (pixel) for every active track this frame.
        active_centroids = {tr.id: (tr.centroid[0], tr.centroid[1])
                            for tr in tracks}
        active_ids = set(active_centroids.keys())

        for zn in self._zones:
            state = self._zone_state.get(zn["id"])
            if state is None:
                continue
            poly_px = [[p[0] * w, p[1] * h] for p in zn["points"]]
            inside_now: set = set()
            for tid, (cx, cy) in active_centroids.items():
                if _point_in_polygon(cx, cy, poly_px):
                    inside_now.add(tid)

            prev_inside: set = state["inside"]
            # Enters: stamp the time we first saw them inside.
            for tid in inside_now - prev_inside:
                state["enter_times"][tid] = now
            # Exits: either left the zone or vanished from tracking entirely.
            gone = prev_inside - inside_now
            for tid in gone:
                t0 = state["enter_times"].pop(tid, None)
                if t0 is not None:
                    state["dwell_samples"].append(now - t0)

            state["inside"] = inside_now
            state["count"] = len(inside_now)
            state["peak"] = max(state["peak"], state["count"])

        # Prune enter_times for tracks the tracker has fully dropped, so dwell
        # for objects that vanished mid-zone is still recorded once.
        for zn in self._zones:
            state = self._zone_state.get(zn["id"])
            if state is None:
                continue
            stale = [tid for tid in list(state["enter_times"].keys())
                     if tid not in active_ids]
            for tid in stale:
                t0 = state["enter_times"].pop(tid, None)
                if t0 is not None and tid in state["inside"]:
                    state["dwell_samples"].append(now - t0)
                    state["inside"].discard(tid)
            state["count"] = len(state["inside"])

    def _log_event(self, message: str) -> None:
        """Append a timestamped message to the bounded event log."""
        with self._lock:
            self._log.append({"t": time.time(), "message": message})

    # ------------------------------------------------------------------ reads
    def get_jpeg(self) -> Optional[bytes]:
        """Return the most recently encoded annotated JPEG frame, if any."""
        with self._lock:
            return self._latest_jpeg

    def stats(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot of the live stats (frontend payload)."""
        with self._lock:
            avg_conf = (self._conf_sum / self._conf_n) if self._conf_n else 0.0
            return {
                "session_id": self.id,
                "status": self.status,
                "source_label": self.label,
                "source_type": self.source_type,
                "error": self.error,
                "current_count": self.current_count,
                "smoothed_count": round(self.smoothed_count, 2),
                "max_count": self.max_count,
                "total_detections": self.total_detections,
                "last_confidence": round(self.last_confidence, 3),
                "avg_confidence": round(avg_conf, 3),
                "fps": round(self.fps, 1),
                "processed_frames": self.processed_frames,
                "elapsed_seconds": round(time.time() - self._start_time, 1),
                "width": self.width,
                "height": self.height,
                "conf_threshold": round(self.conf_threshold, 2),
                "target_classes": list(self.target_classes),
                "detections": [d.as_dict() for d in self._detections],

                # --- tracking + interactive counting (additive) ---
                "unique_total": self._tracker.unique_total(),
                "unique_active": self._tracker.unique_active(),
                "unique_by_class": self._tracker.unique_by_class(),
                "line_counts": self._line_counts_snapshot(),
                "zone_occupancy": self._zone_occupancy_snapshot(),
                "lines": [dict(ln) for ln in self._lines],
                "zones": [dict(zn) for zn in self._zones],

                "log": list(self._log)[-25:],
            }

    def _line_counts_snapshot(self) -> List[Dict[str, Any]]:
        """Build the per-line counts payload. Caller must hold ``self._lock``."""
        out: List[Dict[str, Any]] = []
        for ln in self._lines:
            tally = self._line_counts.get(
                ln["id"], {"in": 0, "out": 0, "by_class": Counter()})
            ins = int(tally["in"])
            outs = int(tally["out"])
            out.append({
                "id": ln["id"],
                "name": ln["name"],
                "in": ins,
                "out": outs,
                "total": ins + outs,
                "by_class": dict(tally["by_class"]),
            })
        return out

    def _zone_occupancy_snapshot(self) -> List[Dict[str, Any]]:
        """Build the per-zone occupancy payload. Caller must hold ``self._lock``."""
        out: List[Dict[str, Any]] = []
        for zn in self._zones:
            state = self._zone_state.get(zn["id"])
            if state is None:
                out.append({"id": zn["id"], "name": zn["name"], "count": 0,
                            "peak": 0, "avg_dwell_s": 0.0})
                continue
            samples = state["dwell_samples"]
            avg_dwell = (sum(samples) / len(samples)) if samples else 0.0
            out.append({
                "id": zn["id"],
                "name": zn["name"],
                "count": int(state["count"]),
                "peak": int(state["peak"]),
                "avg_dwell_s": round(float(avg_dwell), 2),
            })
        return out

    def info(self) -> Dict[str, Any]:
        """Return the lightweight summary used by the session-list endpoints."""
        return {
            "session_id": self.id,
            "status": self.status,
            "source_label": self.label,
            "source_type": self.source_type,
            "created_at": self.created_at,
        }


class SessionManager:
    """Owns all detection sessions and enforces the concurrency cap."""

    def __init__(self, detector: Detector, settings: Settings) -> None:
        self.detector = detector
        self.settings = settings
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, source_type: SourceType, source: str,
               conf_threshold: Optional[float] = None,
               target_classes: Optional[List[str]] = None,
               label: Optional[str] = None, loop: bool = False) -> Session:
        """Create and start a new session.

        Raises ``RuntimeError`` if the active-session cap is reached.
        """
        with self._lock:
            active = [s for s in self._sessions.values()
                      if s.status in ("starting", "running")]
            if len(active) >= self.settings.max_sessions:
                raise RuntimeError(
                    f"Max {self.settings.max_sessions} active session(s) reached. "
                    "Stop one before starting another.")
            sid = uuid.uuid4().hex[:12]
            session = Session(sid, source_type, source, self.detector,
                              self.settings, conf_threshold, target_classes,
                              label, loop)
            self._sessions[sid] = session
        session.start()
        return session

    def get(self, sid: str) -> Optional[Session]:
        """Return a session by id, or None if it doesn't exist."""
        return self._sessions.get(sid)

    def list(self) -> List[Dict[str, Any]]:
        """Return summary info for every known session."""
        return [s.info() for s in self._sessions.values()]

    def stop(self, sid: str) -> bool:
        """Signal a session to stop. Returns False if it doesn't exist."""
        s = self._sessions.get(sid)
        if not s:
            return False
        s.stop()
        return True

    def remove(self, sid: str) -> bool:
        """Stop and forget a session. Returns False if it doesn't exist."""
        s = self._sessions.pop(sid, None)
        if not s:
            return False
        s.stop()
        s.join(timeout=3.0)
        return True

    def shutdown(self) -> None:
        """Stop and join all sessions (called at application shutdown)."""
        for s in list(self._sessions.values()):
            s.stop()
        for s in list(self._sessions.values()):
            s.join(timeout=3.0)
        self._sessions.clear()
