# 🎯 Object Detection

> CPU-optimized real-time object detection — video, streams & live cameras.

Detect objects in **uploaded videos, YouTube / HLS streaming links, RTSP
cameras, and local webcams** — with live bounding boxes and live stats —
running on a modest **2-core / 4 GB CPU-only** machine (no GPU required).

It detects **any of the 80 COCO classes** out of the box (car, truck, bus,
person, bicycle, dog, …). `car` is just the *default* selected class — change it
in config or toggle classes live in the UI.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Node" src="https://img.shields.io/badge/Node-18%2B-339933?logo=node.js&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="React" src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black">
  <img alt="Vite" src="https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white">
  <img alt="Ultralytics YOLO" src="https://img.shields.io/badge/Ultralytics-YOLO26n-0B5394">
  <img alt="ONNX Runtime" src="https://img.shields.io/badge/ONNX%20Runtime-CPU-005CED?logo=onnx&logoColor=white">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

> **A note on accuracy:** No vision model is literally 100% accurate. This app
> uses a strong, lightweight model and gives you a confidence slider, temporal
> smoothing, and class filtering to push *practical* accuracy as high as possible
> on your footage. Expect very good results on clear video; tune the threshold
> for difficult scenes.

---

## Features

- 🎥 **Four source types** — uploaded video files, YouTube / HLS / generic
  stream URLs, RTSP cameras, and local webcams.
- 🟩 **Live annotated video** — bounding boxes drawn server-side and streamed to
  the browser as MJPEG (a plain `<img>`, no heavy player).
- 📊 **Live stats over WebSocket** — current / smoothed / peak counts, total
  detections, FPS, confidence, elapsed time, resolution, and a per-class
  breakdown of the current frame.
- 🧰 **80 COCO classes, filterable** — pick which classes to count from the UI or
  via config; defaults to `car`.
- 🐌 **Built for weak CPUs** — ONNX Runtime inference, configurable inference
  size, frame striding (`detect every Nth frame`), frame downscaling, and a
  live-stream jitter buffer so YouTube/HLS playback doesn't freeze between
  segments.
- 🎚️ **Live tuning** — change confidence threshold, target classes, and
  detection cadence mid-session without restarting.
- 📝 **Activity log** — a rolling, human-readable feed of what was detected.
- 🔌 **Self-contained frontend** — no external CDNs/fonts; builds and runs
  offline.

---

## Tech stack

**Backend:** FastAPI · Uvicorn · [Ultralytics YOLO](https://docs.ultralytics.com/)
(YOLO26n) · ONNX Runtime · OpenCV (headless) · yt-dlp ·
**Frontend:** React 18 · Vite 5 (annotated MJPEG `<img>` + WebSocket stats).

---

## Architecture

```mermaid
flowchart LR
    subgraph Browser["React UI (Vite, :5173)"]
        IMG["&lt;img&gt; MJPEG feed"]
        WS["WebSocket stats"]
        CTL["Controls / source picker"]
    end

    subgraph API["FastAPI backend (:8000)"]
        ROUTES["REST routes\nupload · sessions · settings"]
        SM["SessionManager\n(per-session worker threads)"]
        DET["YOLO detector\n(ONNX Runtime, CPU)"]
        SRC["Source ingestion\nfile · YouTube/HLS · RTSP · webcam"]
    end

    CTL -->|REST| ROUTES
    ROUTES --> SM
    SRC --> SM
    SM --> DET
    SM -->|annotated JPEG| IMG
    SM -->|live stats| WS
```

<details>
<summary>ASCII version</summary>

```
┌──────────────┐   MJPEG video (annotated)   ┌────────────────────────────┐
│              │ ◀─────────────────────────── │  FastAPI backend (:8000)   │
│   React UI   │   WebSocket (live stats)     │   ├─ YOLO detector (CPU)   │
│  (Vite,:5173)│ ◀─────────────────────────── │   ├─ source ingestion      │
│              │   REST (upload, sessions)    │   │   file/YouTube/RTSP/cam │
└──────────────┘ ──────────────────────────▶ │   └─ per-session worker    │
                                              │      threads + buffers     │
                                              └────────────────────────────┘
```
</details>

Each detection **session** runs one background worker thread that reads frames
(a live reader-thread + jitter buffer for streams), runs YOLO on every Nth
frame, draws boxes, and publishes the latest annotated JPEG + stats. The browser
pulls the JPEG stream into an `<img>` and the stats over a WebSocket — capture
and delivery are decoupled, so it stays light on few cores.

---

## Demo / Screenshots

> 📸 Drop a screenshot or short GIF at **`./docs/screenshot.png`** and it will
> render here. (Create the `docs/` folder first — it isn't tracked until you add
> an image.)

![Object Detection dashboard](./docs/screenshot.png)

---

## Requirements

- **Python 3.10+**
- **Node.js 18+** (Node 22 recommended)
- **ffmpeg** on PATH (used by OpenCV/yt-dlp for streams & RTSP; usually already
  present — `ffmpeg -version` to check)

First launch downloads the YOLO weights and exports an ONNX model, so an
internet connection is needed **once** during setup. After that it runs fully
offline.

---

## Quick start

From the project root:

```bash
./start.sh
```

This launches **both** servers. The first run installs dependencies (the backend
pulls PyTorch/OpenCV, auto-downloads the YOLO weights, and exports an ONNX model
— roughly 1–2 min):

- Frontend → http://localhost:5173
- Backend  → http://localhost:8000  (interactive API docs at `/docs`)

Open http://localhost:5173, pick a source, and hit **Start Detection**.

### Run the servers separately

```bash
# Terminal 1 — backend
cd backend && ./run.sh

# Terminal 2 — frontend
cd frontend && ./run.sh
```

`backend/run.sh` creates a virtualenv, installs `requirements.txt`, exports
`models/yolo26n.onnx` on first run, and starts Uvicorn.
`frontend/run.sh` runs `npm install` (first time) then the Vite dev server.

---

## Using it

| Source | What to enter |
|---|---|
| **Upload Video** | An mp4/mov/mkv/webm/… file (optionally loop it) |
| **Stream / YouTube** | A YouTube URL, a live stream, or a direct `.m3u8` / `.mp4` URL |
| **RTSP Camera** | `rtsp://user:pass@host:554/stream` (IP cameras / NVRs) |
| **Webcam** | A device index (`0` = default) on the **backend** machine |

Live controls: **confidence threshold**, **detect-every-Nth-frame** (CPU vs
responsiveness), and **object classes**. **Stop** ends the session; **New
Source** starts over.

---

## Configuration

All settings live in `backend/app/config.py` and can be overridden via
environment variables or a `backend/.env` file. Copy the template to start:

```bash
cp backend/.env.example backend/.env
```

Key knobs:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `yolo26n.onnx` | Model to load. Resolves to `models/<name>`; falls back to the `.pt` twin (or a downloadable `.pt`) if the ONNX isn't present yet. |
| `TARGET_CLASSES` | `car` | Comma-separated COCO classes to count, e.g. `car,truck,bus,person`. The model still detects all 80; this just selects what to report. |
| `CONF_THRESHOLD` | `0.35` | Minimum confidence (0–1) for a detection to count. |
| `IMGSZ` | `416` | Inference size. `320` = faster, `640` = more accurate. Must match the exported ONNX (`run.sh` exports at this size). |
| `PROCESS_EVERY_N` | `2` | Run detection on every Nth frame; in-between frames reuse the last boxes. Higher = lighter CPU. |
| `MAX_FRAME_WIDTH` | `960` | Downscale incoming frames to at most this width before processing/encoding. |
| `LIVE_BUFFER_SECONDS` | `2.0` | Jitter buffer for live streams. Higher = smoother but more delay behind "live" (and more RAM); `0` disables it. |
| `STREAM_FPS` | `15` | Cap on MJPEG frames/sec sent to the browser. |
| `MAX_SESSIONS` | `2` | Max simultaneous detection sessions (a 2-core box really handles ~1 heavy stream). |
| `NUM_THREADS` | `2` | OpenCV / ONNX / Torch thread count. Match your CPU core count. |

Other available knobs (see `backend/.env.example`): `IOU_THRESHOLD`, `MAX_DET`,
`MIN_BOX_AREA_RATIO`, `JPEG_QUALITY`, `HOST`, `PORT`, `CORS_ORIGINS`,
`LOG_SIZE`.

The frontend talks to `http://localhost:8000` by default; override with
`VITE_API_BASE` in `frontend/.env` (see `frontend/.env.example`).

---

## Performance

The default is **YOLO26n exported to ONNX @ 416** — chosen because ONNX Runtime
plus a smaller inference size is the biggest practical speed lever on a CPU.

Indicative throughput, measured on a **fast dev CPU (Apple M4)** — a real
2-core box is slower in absolute terms, but the **ratios hold**:

| Config | ~FPS (Apple M4) |
|---|---|
| PyTorch `.pt` @ 640 | ~8.5 |
| PyTorch `.pt` @ 416 | ~16 |
| **ONNX @ 416 (default)** | **~30** |
| yolo11n ONNX @ 320 | ~36 |

> ONNX @ 416 is roughly **3.5× faster** than `.pt` @ 640.

The two main speed levers:

1. **Export to ONNX** (done automatically on first run by `run.sh`; the
   fallback is the slower `.pt` model).
2. **Lower `IMGSZ`** — `416 → 320` is a meaningful speedup for a small accuracy
   cost.

Then raise `PROCESS_EVERY_N`, keep to one active session, and lower
`MAX_FRAME_WIDTH` / `STREAM_FPS` if you're still CPU-bound.

### Re-exporting the model

If you change `IMGSZ` or want a different model, re-run the export so the ONNX
input size matches:

```bash
cd backend
source .venv/bin/activate              # created by run.sh
python scripts/export_model.py --format onnx --imgsz 416   # → models/yolo26n.onnx
```

> **YOLO26 + ONNX note:** YOLO26 has an NMS-free head. After exporting, sanity-
> check that boxes aren't duplicated (one per object). If they are, point
> `MODEL_PATH` at the `.pt` model instead. The export script prints a reminder.

---

## Deploy

The backend can serve the **built frontend and the API on a single port**, which
makes it easy to host anywhere. When `backend/static/` exists (a copied frontend
build), FastAPI serves the SPA at `/` and the API at `/api/*` — no separate
frontend server or CORS needed. The frontend talks to its own origin by default
(`VITE_API_BASE=""`).

### Hugging Face Spaces (Docker)

A ready-to-use Docker setup lives in [`deploy/huggingface/`](deploy/huggingface/).
It clones this GitHub repo, builds the frontend, installs the backend (CPU-only
PyTorch), exports the ONNX model, and serves everything on port **7860**.

1. Create a **Docker** Space (Blank).
2. Add a `Dockerfile` with the contents of
   [`deploy/huggingface/Dockerfile`](deploy/huggingface/Dockerfile).
3. Set `REPO_URL` (and `REPO_REF`) near the top to your GitHub repo.
4. Commit — the Space builds and launches.

The free **CPU basic** tier (2 vCPU / 16 GB) runs it comfortably. Full
instructions: [`deploy/huggingface/README.md`](deploy/huggingface/README.md).

### Run the single-port build locally

```bash
cd frontend && npm run build          # -> frontend/dist
cp -r frontend/dist backend/static    # backend now serves the UI
cd ../backend && ./run.sh             # open http://localhost:8000
```

---

## Extending to other objects

The detector is class-agnostic — the bundled model knows all 80 COCO classes.

- **Count more COCO classes:** set `TARGET_CLASSES=car,truck,bus,person` (or
  toggle classes live in the UI). No code changes needed.
- **Detect something not in COCO:** train or fine-tune a YOLO model on your own
  data, export it, and point `MODEL_PATH` at the new weights. Re-export to ONNX
  at your chosen `IMGSZ` for CPU speed.

---

## API reference

Base URL: `http://localhost:8000`. Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness probe + active model path |
| `GET` | `/api/config` | Default settings + the model's available classes |
| `POST` | `/api/upload` | Upload a video file → `{ file_id, filename, size_bytes }` |
| `POST` | `/api/sessions` | Start a detection session for a source |
| `GET` | `/api/sessions` | List all sessions (active + finished) |
| `GET` | `/api/sessions/{id}` | Session info |
| `GET` | `/api/sessions/{id}/stats` | One-shot stats snapshot |
| `PATCH` | `/api/sessions/{id}/settings` | Update conf / classes / cadence live |
| `POST` | `/api/sessions/{id}/stop` | Stop a session (keeps it listed) |
| `DELETE` | `/api/sessions/{id}` | Stop **and** remove a session |
| `GET` | `/api/sessions/{id}/video` | **MJPEG** annotated video stream |
| `WS` | `/api/sessions/{id}/ws` | **WebSocket** live-stats stream |

**Create a session** — `POST /api/sessions`:

```json
{
  "source_type": "url",
  "source": "https://www.youtube.com/watch?v=...",
  "label": "Front gate",
  "conf_threshold": 0.4,
  "target_classes": ["car", "truck"],
  "loop": false
}
```

`source_type` is one of `upload` (source = `file_id`), `url` (YouTube/HLS/any
yt-dlp–resolvable page), `rtsp` (full `rtsp://` URL), or `webcam` (device index
string, e.g. `"0"`).

**Live-stats payload** (the same shape from `GET …/stats` and the WebSocket):

```json
{
  "session_id": "…", "status": "running",
  "source_label": "…", "source_type": "url", "error": null,
  "current_count": 3, "smoothed_count": 2.7, "max_count": 6,
  "total_detections": 142,
  "last_confidence": 0.88, "avg_confidence": 0.81,
  "fps": 14.9, "processed_frames": 430, "elapsed_seconds": 29.1,
  "width": 960, "height": 540,
  "conf_threshold": 0.35, "target_classes": ["car"],
  "detections": [{ "cls": "car", "conf": 0.88, "box": [x1, y1, x2, y2] }],
  "log": [{ "t": 1750000000.0, "message": "3 object(s): 2 car, 1 truck (top 88%)" }]
}
```

`status` is one of `starting | running | finished | error | stopped`.

---

## Project layout

```
yolo/
├── backend/
│   ├── app/
│   │   ├── config.py          # all tunables (env-overridable)
│   │   ├── schemas.py         # API + stats models (shared shape w/ frontend)
│   │   ├── detector.py        # YOLO wrapper: predict + draw, class filter, ONNX
│   │   ├── sources.py         # file / YouTube / HLS / RTSP / webcam → frames
│   │   ├── stream_manager.py  # per-session worker threads, buffers, pacing
│   │   └── main.py            # FastAPI routes (upload, sessions, MJPEG, WS)
│   ├── scripts/
│   │   ├── export_model.py    # export .pt → ONNX/OpenVINO
│   │   ├── benchmark.py       # throughput benchmark
│   │   └── smoke_test.py      # end-to-end sanity check
│   ├── requirements.txt
│   ├── .env.example
│   └── run.sh
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── hooks/useStatsSocket.js
│   │   └── components/{SourceSelector,VideoView,StatsPanel,DetectionLog,Controls}.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── run.sh
├── start.sh                   # run both servers together
├── .gitignore
├── LICENSE
└── README.md
```

---

## Troubleshooting

- **"Backend offline" in the UI** — start the backend (`cd backend && ./run.sh`).
  The frontend talks to `http://localhost:8000`; override with `VITE_API_BASE`
  in `frontend/.env`.
- **YouTube URL fails** — yt-dlp goes stale as sites change. Update it:
  `pip install -U yt-dlp` (inside `backend/.venv`). Some videos are region- or
  login-restricted and can't be resolved.
- **RTSP won't open / stutters** — verify the URL and credentials, and that the
  camera is reachable **from the backend machine**. Raising
  `LIVE_BUFFER_SECONDS` smooths stutter at the cost of more latency.
- **Webcam not found** — the device index refers to a camera on the **backend
  host**, not your laptop (unless they're the same machine). Try `0`, `1`, …
- **Too slow / low FPS** — make sure ONNX is in use (`models/yolo26n.onnx`
  exists; check `/api/health`), drop `IMGSZ` to `320`, raise `PROCESS_EVERY_N`,
  lower `MAX_FRAME_WIDTH` / `STREAM_FPS`, and keep to one active session.
- **ffmpeg not found** — install it (`brew install ffmpeg` /
  `apt install ffmpeg`) and ensure it's on PATH.

---

## Roadmap

- [ ] Multi-object tracking with stable IDs (e.g. ByteTrack) and line-crossing
      counts.
- [ ] Recording / snapshot export of annotated clips.
- [ ] Region-of-interest masks and per-zone counts.
- [ ] OpenVINO / INT8 export path documented for Intel CPUs.
- [ ] Optional persistence of sessions and historical stats.
- [ ] Authentication for non-local deployments.

---

## License

Released under the [MIT License](./LICENSE). Copyright © 2026 Prathiv AR.
Change the copyright holder in `LICENSE` if you fork this.
