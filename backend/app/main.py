"""FastAPI application for the Object Detection API.

Exposes the HTTP/WebSocket surface that the React frontend talks to:
upload, session lifecycle, the annotated MJPEG video feed, and a live-stats
WebSocket. The heavy lifting (frame capture, detection, encoding) lives in the
per-session workers owned by :class:`~app.stream_manager.SessionManager`; this
module is just the transport layer.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (FastAPI, File, HTTPException, Request, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .detector import Detector
from .schemas import (ConfigResponse, CreateSessionRequest, SessionInfo,
                      UpdateSettingsRequest, UploadResponse)
from .stream_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# Video container extensions accepted by the upload endpoint.
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg",
                     ".mpeg", ".flv", ".wmv"}

# Process-wide singletons populated by the lifespan handler at startup:
# "settings" (Settings), "detector" (Detector), "manager" (SessionManager).
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the detector once at startup and tear down active sessions on exit."""
    settings = get_settings()
    logger.info("Loading detector (%s)...", settings.resolved_model_path())
    detector = Detector(
        model_path=settings.resolved_model_path(),
        imgsz=settings.imgsz,
        iou=settings.iou_threshold,
        max_det=settings.max_det,
        num_threads=settings.num_threads,
    )
    logger.info("Model classes available: %d", len(detector.available_classes))
    state["settings"] = settings
    state["detector"] = detector
    state["manager"] = SessionManager(detector, settings)
    yield
    state["manager"].shutdown()


app = FastAPI(title="Object Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def manager() -> SessionManager:
    """Return the process-wide session manager created at startup."""
    return state["manager"]


# --------------------------------------------------------------------- health
@app.get("/api/health")
async def health() -> Dict[str, str]:
    """Liveness probe; also reports the configured model path."""
    return {"status": "ok", "model": state["settings"].model_path}


@app.get("/api/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    """Expose default detection settings and the model's available classes."""
    s = state["settings"]
    d: Detector = state["detector"]
    return ConfigResponse(
        default_conf_threshold=s.conf_threshold,
        default_target_classes=s.target_classes,
        available_classes=d.available_classes,
        imgsz=s.imgsz,
        process_every_n=s.process_every_n,
        max_sessions=s.max_sessions,
        model_path=s.model_path,
    )


# --------------------------------------------------------------------- upload
@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Stream an uploaded video file to disk and return its generated id."""
    s = state["settings"]
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. "
                                 f"Allowed: {sorted(ALLOWED_VIDEO_EXT)}")
    file_id = f"{uuid.uuid4().hex[:12]}{ext}"
    dest = s.upload_dir / file_id
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
            size += len(chunk)
    logger.info("Uploaded %s (%d bytes) -> %s", file.filename, size, file_id)
    return UploadResponse(file_id=file_id, filename=file.filename or file_id,
                          size_bytes=size)


# ------------------------------------------------------------------- sessions
@app.post("/api/sessions", response_model=SessionInfo)
async def create_session(req: CreateSessionRequest) -> SessionInfo:
    """Start a detection session for the requested source."""
    try:
        session = manager().create(
            source_type=req.source_type,
            source=req.source,
            conf_threshold=req.conf_threshold,
            target_classes=req.target_classes,
            label=req.label,
            loop=req.loop,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))
    return SessionInfo(**session.info())


@app.get("/api/sessions")
async def list_sessions() -> List[Dict[str, Any]]:
    """List all known sessions (active and finished)."""
    return manager().list()


@app.get("/api/sessions/{sid}", response_model=SessionInfo)
async def get_session(sid: str) -> SessionInfo:
    """Return summary info for a single session."""
    s = manager().get(sid)
    if not s:
        raise HTTPException(404, "Session not found")
    return SessionInfo(**s.info())


@app.get("/api/sessions/{sid}/stats")
async def get_stats(sid: str) -> Dict[str, Any]:
    """Return the latest live-stats snapshot for a session."""
    s = manager().get(sid)
    if not s:
        raise HTTPException(404, "Session not found")
    return s.stats()


@app.patch("/api/sessions/{sid}/settings")
async def update_settings(sid: str, req: UpdateSettingsRequest) -> Dict[str, bool]:
    """Apply per-session overrides (confidence, classes, frame stride)."""
    s = manager().get(sid)
    if not s:
        raise HTTPException(404, "Session not found")
    s.update_settings(req.conf_threshold, req.target_classes,
                      req.process_every_n, req.lines, req.zones)
    return {"ok": True}


@app.post("/api/sessions/{sid}/stop")
async def stop_session(sid: str) -> Dict[str, bool]:
    """Signal a session's worker to stop (keeps it listed with its final stats)."""
    if not manager().stop(sid):
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str) -> Dict[str, bool]:
    """Stop a session and remove it from the manager entirely."""
    if not manager().remove(sid):
        raise HTTPException(404, "Session not found")
    return {"ok": True}


# ----------------------------------------------------------- MJPEG video feed
async def _mjpeg_generator(sid: str, request: Request) -> AsyncIterator[bytes]:
    """Yield ``multipart/x-mixed-replace`` JPEG frames from a session's
    latest-frame buffer.

    Async (not sync) on purpose: a sync streaming generator occupies one AnyIO
    threadpool slot for the entire lifetime of the connection — a few viewers
    would exhaust the pool. With ``await asyncio.sleep(...)`` we yield control
    back to the event loop between frames and scale to many viewers cheaply.
    """
    s = manager().get(sid)
    settings = state["settings"]
    boundary = b"--frame"
    interval = 1.0 / max(1, settings.stream_fps)
    last_sent: Optional[bytes] = None
    no_frame_deadline = time.time() + 15  # wait up to 15s for the first frame
    while s is not None:
        if await request.is_disconnected():
            break
        frame = s.get_jpeg()
        if frame is None:
            # Terminal session that never produced a frame (e.g. the source
            # failed to open): stop promptly so the <img> request ends and the
            # UI shows the error instead of spinning forever.
            if s.status in ("error", "finished", "stopped"):
                break
            if time.time() > no_frame_deadline:
                break
            await asyncio.sleep(0.05)
            continue
        # Avoid re-sending an identical frame (e.g. paused/finished source).
        if frame is not last_sent:
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                   + frame + b"\r\n")
            last_sent = frame
        if s.status in ("error", "finished", "stopped"):
            break
        await asyncio.sleep(interval)


@app.get("/api/sessions/{sid}/video")
async def video_feed(sid: str, request: Request) -> StreamingResponse:
    """Serve the annotated MJPEG stream for a session."""
    if not manager().get(sid):
        raise HTTPException(404, "Session not found")
    return StreamingResponse(
        _mjpeg_generator(sid, request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


# ----------------------------------------------------------- WS live stats
@app.websocket("/api/sessions/{sid}/ws")
async def stats_ws(websocket: WebSocket, sid: str) -> None:
    """Push a live-stats snapshot every 0.5s until the session ends or the
    client disconnects, then send one final snapshot and close cleanly."""
    await websocket.accept()
    s = manager().get(sid)
    if not s:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    try:
        while True:
            await websocket.send_json(s.stats())
            if s.status in ("error", "finished", "stopped"):
                # send a final snapshot then close cleanly
                await asyncio.sleep(0.5)
                await websocket.send_json(s.stats())
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WS error for %s: %s", sid, e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# --------------------------------------------------- static frontend (1 port)
# When the built React app is present (backend/static), serve it from the SAME
# origin as the API. This powers the single-port Docker / Hugging Face deploy.
# Mounted LAST so it only catches paths the /api routes above didn't. In local
# split-stack dev the dir is absent (frontend runs on Vite), so this is skipped.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True),
              name="frontend")
    logger.info("Serving frontend from %s", _STATIC_DIR)
else:
    logger.info("No static frontend at %s (dev mode); API only.", _STATIC_DIR)


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)
