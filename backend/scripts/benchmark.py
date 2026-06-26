"""Benchmark detection speed across models / formats / input sizes on THIS CPU.

Helps pick the fastest config for your machine. Run in the backend venv:
    python scripts/benchmark.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import ASSETS  # noqa: E402

IMG = cv2.imread(str(ASSETS / "bus.jpg"))
RUNS = 8


def _onnx_path(weights: str, imgsz: int) -> Path:
    stem = Path(weights).stem
    dest = MODELS / f"{stem}-{imgsz}.onnx"
    if dest.exists():
        return dest
    print(f"   exporting {weights} -> {dest.name} (one-time)...")
    out = YOLO(weights).export(format="onnx", imgsz=imgsz, simplify=True,
                               dynamic=False)
    Path(out).replace(dest)
    return dest


def bench(label: str, model_path, imgsz: int, runs: int = RUNS) -> Optional[float]:
    """Time ``runs`` predictions and print median latency/throughput.

    Returns the median latency in milliseconds, or None if the config failed.
    """
    try:
        m = YOLO(str(model_path))
        m.predict(IMG, imgsz=imgsz, conf=0.25, verbose=False, device="cpu")  # warmup
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            r = m.predict(IMG, imgsz=imgsz, conf=0.25, verbose=False, device="cpu")
            ts.append(time.perf_counter() - t)
        ms = statistics.median(ts) * 1000
        n = len(r[0].boxes) if r and r[0].boxes is not None else 0
        print(f"{label:34s} {ms:7.1f} ms   ~{1000 / ms:5.1f} fps   ({n} boxes)")
        return ms
    except Exception as e:
        print(f"{label:34s} FAILED: {e}")
        return None


def main() -> None:
    print(f"Image: {IMG.shape}, runs/config: {RUNS}\n")
    print(f"{'config':34s} {'latency':>9s}   {'throughput':>9s}")
    print("-" * 70)

    # PyTorch (.pt) at several sizes — shows the cost of resolution.
    bench("yolo26n.pt  @640 (current default)", "yolo26n.pt", 640)
    bench("yolo26n.pt  @416", "yolo26n.pt", 416)
    bench("yolo26n.pt  @320", "yolo26n.pt", 320)
    print("-" * 70)

    # ONNX — usually a big CPU win at the same size.
    for sz in (416, 320):
        try:
            bench(f"yolo26n.onnx @{sz}", _onnx_path("yolo26n.pt", sz), sz)
        except Exception as e:
            print(f"yolo26n.onnx @{sz:<3d}                FAILED export: {e}")
    print("-" * 70)

    # YOLO11n ONNX as a proven fallback (no NMS-free export caveat).
    for sz in (416, 320):
        try:
            bench(f"yolo11n.onnx @{sz}", _onnx_path("yolo11n.pt", sz), sz)
        except Exception as e:
            print(f"yolo11n.onnx @{sz:<3d}                FAILED export: {e}")

    print("\nLower ms / higher fps = faster. Pick the fastest that still detects")
    print("your objects well on YOUR footage, then set MODEL_PATH + IMGSZ in "
          "backend/.env")


if __name__ == "__main__":
    main()
