# Hugging Face Spaces — Docker deploy for the Object Detection app.
#
# This Dockerfile CLONES your public GitHub repo, builds the React frontend,
# installs the FastAPI backend, exports the YOLO model to ONNX, and serves the
# whole thing (frontend + API) on the single port HF exposes (7860).
#
# HOW TO USE:
#   1. Create a new Space -> SDK: "Docker" -> "Blank".
#   2. Add a file named `Dockerfile` and paste this in.
#   3. Edit REPO_URL below to point at YOUR GitHub repo (and REPO_REF if not main).
#   4. Commit. The Space builds and launches automatically.
#
# Hardware: the free "CPU basic" tier (2 vCPU / 16 GB) is plenty. No GPU needed.

# ──────────────────────────── Stage 1: build the frontend ───────────────────
FROM node:20-slim AS frontend

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG REPO_URL=https://github.com/mohamedabubasith/realtime-object-detection.git
ARG REPO_REF=main

RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" /src

WORKDIR /src/frontend
# Empty API base => the frontend calls the API on its own origin (same port).
ENV VITE_API_BASE=""
RUN npm ci && npm run build          # -> /src/frontend/dist

# ──────────────────────────── Stage 2: Python runtime ───────────────────────
FROM python:3.11-slim AS runtime

# System libraries: ffmpeg (stream/RTSP/HLS decode) + OpenCV runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as a non-root user (uid 1000) with a writable
# home. Create it and install everything under that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app

# Backend source (from the repo cloned in stage 1) + the built frontend.
COPY --from=frontend --chown=user:user /src/backend ./backend
COPY --from=frontend --chown=user:user /src/frontend/dist ./backend/static

# Install the CPU-only PyTorch wheel FIRST so pip doesn't pull the huge CUDA
# build (ultralytics depends on torch). Then the rest of the backend deps.
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r backend/requirements.txt \
    && pip install --user onnx onnxslim          # needed only for the export below

WORKDIR /home/user/app/backend

# Runtime config. Caches/config must live somewhere writable by uid 1000.
ENV YOLO_CONFIG_DIR=/home/user/app/backend/.ultralytics \
    MPLCONFIGDIR=/home/user/app/backend/.mpl \
    MODEL_PATH=models/yolo26n.onnx \
    IMGSZ=416 \
    HOST=0.0.0.0 \
    PORT=7860 \
    MAX_SESSIONS=1

# Pre-export the ONNX model at build time so the first request isn't slow.
# Best-effort: if it fails, the app falls back to yolo26n.pt at runtime.
RUN python scripts/export_model.py --format onnx --imgsz 416 \
    || echo "ONNX export skipped at build; will use yolo26n.pt at runtime."

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
