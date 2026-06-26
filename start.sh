#!/usr/bin/env bash
# Launch backend + frontend together for local development.
# Backend:  http://localhost:8000  (API + docs at /docs)
# Frontend: http://localhost:5173
set -euo pipefail
cd "$(dirname "$0")"

cleanup() {
  echo ""
  echo ">> Shutting down..."
  kill 0
}
trap cleanup EXIT INT TERM

echo "=============================================="
echo " Object Detection — starting backend + frontend"
echo "=============================================="

( ./backend/run.sh ) &
( ./frontend/run.sh ) &

wait
