#!/usr/bin/env bash
# Start the frontend dev server.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d node_modules ]; then
  echo ">> Installing frontend dependencies..."
  npm install
fi

echo ">> Starting frontend on http://localhost:5173"
exec npm run dev -- --host
