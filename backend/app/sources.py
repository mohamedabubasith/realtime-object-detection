"""Turn any requested source into a stream of frames.

Supported sources:
  - upload  : a local file previously uploaded (path under uploads/)
  - url     : YouTube / HLS (.m3u8) / any page yt-dlp can resolve to a media URL
  - rtsp    : rtsp:// live camera
  - webcam  : a local camera device index

`FrameSource` wraps cv2.VideoCapture and adds:
  - yt-dlp resolution for page URLs
  - RTSP-over-TCP (more reliable than the default UDP)
  - auto-reconnect for live sources
  - frame downscaling to a max width (big CPU saver)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("sources")

# FFmpeg options applied to every ffmpeg-backed VideoCapture. Must be set before
# the capture is constructed.
#   rtsp_transport;tcp  -> RTSP over TCP (far more reliable than default UDP)
#   timeout;<usec>      -> socket timeout in MICROSECONDS (modern name for the
#                          old "stimeout"); beats OpenCV's ~30s hang on dead links
#   reconnect*          -> let ffmpeg auto-reconnect HTTP/HLS streams
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000"
    "|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5",
)


@dataclass
class ResolvedSource:
    """A source resolved to something ``cv2.VideoCapture`` can open."""

    target: Union[str, int]   # path/url string, or int device index
    is_live: bool             # True for streams/cameras (no natural end)
    label: str                # human-readable name for the UI


def resolve_stream_url(page_url: str, max_height: int = 720) -> Tuple[str, bool, str]:
    """Use yt-dlp to turn a YouTube/stream page URL into a direct media URL that
    OpenCV/ffmpeg can read. Returns (media_url, is_live, title)."""
    import yt_dlp

    cookies = os.environ.get("YTDLP_COOKIES", "").strip()
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Resilience against flaky networks / rate-limiting.
        "retries": 3,
        "extractor_retries": 3,
        "socket_timeout": 20,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
        # IMPORTANT: pick a SINGLE already-muxed (progressive) stream. OpenCV /
        # ffmpeg via VideoCapture cannot mux a separate video+audio pair, so we
        # must never select "bestvideo+bestaudio". The bare "best" selector and
        # the filters below all resolve to one playable URL. We also cap the
        # height so a weak CPU isn't decoding 4K, and prefer HTTP progressive.
        "format": (
            f"best[height<=?{max_height}][protocol^=http][ext=mp4]/"
            f"best[height<=?{max_height}][protocol^=http]/"
            f"best[height<=?{max_height}]/best"
        ),
    }
    # Optional escape hatch: a Netscape-format cookies.txt lets YouTube work from
    # servers whose datacenter IP it would otherwise block. Set the env var
    # YTDLP_COOKIES=/path/to/cookies.txt.
    if cookies and Path(cookies).exists():
        ydl_opts["cookiefile"] = cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(page_url, download=False)
    except Exception as e:
        raise ValueError(
            "Could not load this URL via yt-dlp. YouTube (and some sites) block "
            "requests from cloud / datacenter servers such as Hugging Face "
            "Spaces. Try uploading the video file, an RTSP camera, or a direct "
            ".m3u8 / .mp4 stream URL instead — or provide cookies via "
            f"YTDLP_COOKIES. (details: {type(e).__name__}: {str(e)[:140]})"
        ) from e

    if info is None:
        raise ValueError("yt-dlp could not resolve the URL")

    # When a single format is selected, the direct URL is on the top-level dict.
    media_url = info.get("url")
    if not media_url:
        # Fall back to the best format entry that carries a URL.
        formats = info.get("formats") or []
        for f in reversed(formats):
            if f.get("url"):
                media_url = f["url"]
                break
    if not media_url:
        raise ValueError("yt-dlp resolved no playable media URL")

    is_live = bool(info.get("is_live"))
    title = info.get("title") or page_url
    return media_url, is_live, title


def build_resolved_source(source_type: str, source: str,
                          upload_dir: Union[str, Path],
                          max_height: int = 720) -> ResolvedSource:
    """Resolve a (source_type, source) pair into a :class:`ResolvedSource`."""
    st = source_type
    if st == "upload":
        path = Path(upload_dir) / source
        if not path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {source}")
        return ResolvedSource(str(path), is_live=False, label=path.name)

    if st == "webcam":
        try:
            idx = int(source)
        except ValueError:
            raise ValueError("webcam source must be an integer device index")
        return ResolvedSource(idx, is_live=True, label=f"Webcam {idx}")

    if st == "rtsp":
        return ResolvedSource(source, is_live=True, label=source)

    if st == "url":
        low = source.lower()
        # Direct media URLs (HLS/mp4) can be opened straight away.
        if low.endswith(".m3u8") or low.endswith(".mp4") or low.startswith("rtsp://"):
            return ResolvedSource(source, is_live=low.endswith(".m3u8"),
                                  label=source)
        media_url, is_live, title = resolve_stream_url(source, max_height)
        return ResolvedSource(media_url, is_live=is_live, label=title)

    raise ValueError(f"Unknown source_type: {st}")


class FrameSource:
    """Opens a ResolvedSource and yields downscaled BGR frames.

    For LIVE sources we run a background reader thread that pre-buffers frames
    into a bounded jitter buffer. This is what keeps the video from freezing
    every few seconds while ffmpeg fetches the next HLS segment: the reader
    absorbs the burst of frames a segment delivers, and the consumer drains the
    buffer smoothly (paced to real-time by the worker). The buffer drops the
    oldest frames if detection can't keep up, so we stay near-live with bounded
    memory.
    """

    def __init__(self, resolved: ResolvedSource, max_width: int = 960,
                 loop_file: bool = False, reconnect: bool = True,
                 buffer_seconds: float = 0.0) -> None:
        self.resolved = resolved
        self.max_width = max_width
        self.loop_file = loop_file and not resolved.is_live
        self.reconnect = reconnect and resolved.is_live
        self.buffer_seconds = buffer_seconds if resolved.is_live else 0.0
        self.cap: Optional[cv2.VideoCapture] = None
        self._scale: Optional[float] = None
        # reader-thread / jitter-buffer state (only used when buffer_seconds > 0)
        self._buf: Optional[Deque[np.ndarray]] = None
        self._reader: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._ended = False

    # ------------------------------------------------------------- capture i/o
    def _open_capture(self) -> None:
        """Open (or re-open) the underlying ``cv2.VideoCapture``."""
        target = self.resolved.target
        self.cap = (cv2.VideoCapture(target, cv2.CAP_FFMPEG)
                    if isinstance(target, str)
                    else cv2.VideoCapture(target))
        # A tiny buffer only helps low-latency RTSP cameras. For HLS we do our
        # own read-ahead, so leaving the default buffer lets ffmpeg prefetch.
        if isinstance(target, str) and target.lower().startswith("rtsp"):
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.resolved.label}")

    def _release_capture(self) -> None:
        """Release the capture if open, swallowing any errors."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def open(self) -> None:
        """Open the capture and, for live sources, start the reader thread."""
        self._open_capture()
        if self.buffer_seconds > 0:
            fps = self.fps_hint or 25.0
            maxlen = int(self.buffer_seconds * fps)
            maxlen = max(8, min(maxlen, 120))  # clamp for bounded memory
            self._buf = deque(maxlen=maxlen)
            self._ended = False
            self._reader_stop.clear()
            self._reader = threading.Thread(
                target=self._reader_loop, name="frame-reader", daemon=True)
            self._reader.start()
            logger.info("Live jitter buffer: %d frames (~%.1fs @ %.0ffps)",
                        maxlen, maxlen / fps, fps)

    @property
    def fps_hint(self) -> float:
        if self.cap is None:
            return 0.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and 0 < fps <= 120 else 0.0

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        if self._scale is None:
            w = frame.shape[1]
            self._scale = (self.max_width / w) if w > self.max_width else 1.0
        if self._scale != 1.0:
            frame = cv2.resize(frame, None, fx=self._scale, fy=self._scale,
                               interpolation=cv2.INTER_AREA)
        return frame

    def _raw_read(self) -> Optional[np.ndarray]:
        """Blocking read straight from the capture (with loop/reconnect)."""
        if self.cap is None:
            self._open_capture()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if self.loop_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    return None
            elif self.reconnect:
                logger.info("Stream dropped, reconnecting: %s",
                            self.resolved.label)
                self._reconnect()
                if self.cap is None or not self.cap.isOpened():
                    return None
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    return None
            else:
                return None
        return self._downscale(frame)

    def _reader_loop(self) -> None:
        """Continuously pull frames into the jitter buffer (live sources)."""
        while not self._reader_stop.is_set():
            frame = self._raw_read()
            if frame is None:
                self._ended = True
                break
            self._buf.append(frame)  # deque(maxlen) drops the oldest if full

    def read(self) -> Optional[np.ndarray]:
        """Return the next frame, or None when the source is exhausted.

        Buffered (live) mode: pops the oldest buffered frame, waiting briefly if
        the buffer is momentarily empty. Direct mode (files/webcam): reads the
        capture directly.
        """
        if self._buf is None:
            return self._raw_read()
        while True:
            if self._buf:
                return self._buf.popleft()
            if self._ended or self._reader_stop.is_set():
                return None
            time.sleep(0.005)

    def _reconnect(self, attempts: int = 5, delay: float = 1.5) -> None:
        """Re-open only the capture (live sources); the reader thread keeps
        running. Leaves ``self.cap`` None if every attempt fails."""
        self._release_capture()
        for i in range(attempts):
            time.sleep(delay)
            try:
                self._open_capture()
                logger.info("Reconnected to %s", self.resolved.label)
                return
            except Exception as e:
                logger.warning("Reconnect %d/%d failed: %s", i + 1, attempts, e)
        self.cap = None

    def release(self) -> None:
        """Stop the reader thread (if any) and release the capture."""
        self._reader_stop.set()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        self._release_capture()
