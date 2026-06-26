#!/usr/bin/env bash
# Start the object-detection backend.
# Creates a venv on first run, installs deps, then launches the API server.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
VENV=.venv

if [ ! -d "$VENV" ]; then
  echo ">> Creating virtual environment ($VENV)..."
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">> Installing dependencies (first run pulls torch/opencv, ~1-2 min)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# One-time: export an ONNX model for fast CPU inference (~3.5x vs .pt@640).
# Best-effort — if it fails (e.g. low RAM), the app falls back to the .pt model.
IMGSZ_EXPORT=${IMGSZ:-416}
if [ ! -f "models/yolo26n.onnx" ]; then
  echo ">> Exporting YOLO26n -> ONNX @${IMGSZ_EXPORT} for faster CPU inference (one-time)..."
  python scripts/export_model.py --format onnx --imgsz "${IMGSZ_EXPORT}" \
    || echo ">> ONNX export failed (will run the slower .pt model instead)."
fi

# Load .env if present so MODEL_PATH etc. are honored when run directly.
PORT_DEFAULT=8000
PORT_RUN=${PORT:-$PORT_DEFAULT}

echo ">> Starting API on http://localhost:${PORT_RUN}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT_RUN}"
