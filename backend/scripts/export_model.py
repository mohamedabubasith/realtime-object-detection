"""Download the YOLO nano weights and (optionally) export them to a CPU-friendly
format for faster inference on a weak machine.

Usage:
    python scripts/export_model.py                 # just download yolo26n.pt
    python scripts/export_model.py --format onnx   # export -> models/yolo26n.onnx
    python scripts/export_model.py --format openvino --int8

ONNX (run via onnxruntime) is the easiest speedup and is well supported.
OpenVINO is fastest on Intel CPUs. INT8 quantization roughly halves size and can
2x throughput at a small accuracy cost.

After exporting, point the backend at it:
    MODEL_PATH=models/yolo26n.onnx   (in backend/.env)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt",
                    help="Base weights to download/export (default: yolo26n.pt)")
    ap.add_argument("--format", choices=["none", "onnx", "openvino"],
                    default="none", help="Export format (default: none)")
    ap.add_argument("--imgsz", type=int, default=416,
                    help="Export/inference size (default 416). Must match IMGSZ "
                         "in backend/.env. 320=faster, 640=more accurate.")
    ap.add_argument("--int8", action="store_true",
                    help="INT8 quantize (smaller/faster, needs a calib dataset "
                         "for best accuracy; uses a default otherwise)")
    args = ap.parse_args()

    from ultralytics import YOLO

    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"Loading/downloading weights: {args.weights}")
    model = YOLO(args.weights)

    # Keep a copy of the .pt under models/ so the backend can find it offline.
    src_pt = Path(args.weights)
    if src_pt.exists():
        shutil.copy(src_pt, MODELS / src_pt.name)
    else:
        # Ultralytics caches downloaded weights; copy from its resolved path.
        ckpt = getattr(model, "ckpt_path", None)
        if ckpt and Path(ckpt).exists():
            shutil.copy(ckpt, MODELS / Path(ckpt).name)

    if args.format == "none":
        print(f"Done. Weights are in {MODELS}. "
              f"Set MODEL_PATH to one of them in backend/.env (the default "
              f"MODEL_PATH=models/yolo26n.onnx expects an ONNX export).")
        return

    print(f"Exporting to {args.format} (imgsz={args.imgsz}, int8={args.int8})...")
    # Static shape (dynamic=False) — on CPU this is ~3x faster than a dynamic
    # ONNX. The model is locked to this imgsz; the backend auto-detects that size
    # at load time, so an IMGSZ mismatch won't crash (it just uses this size).
    # To actually run at a different size, re-export with --imgsz <n>.
    out = model.export(format=args.format, imgsz=args.imgsz, int8=args.int8,
                       simplify=True, dynamic=False)
    # `out` may be a path string or directory (openvino). Move under models/.
    out_path = Path(out)
    dest = MODELS / out_path.name
    if out_path.resolve() != dest.resolve():
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(out_path), str(dest))
    print(f"Exported to: {dest}")
    print(f"Now set MODEL_PATH={dest.relative_to(BASE)} in backend/.env")
    if "yolo26" in args.weights.lower() and args.format == "onnx":
        print("\nNOTE: YOLO26 is NMS-free. After exporting to ONNX, sanity-check "
              "that detections aren't duplicated (one box per object). If you see "
              "duplicate/overlapping boxes, prefer the .pt model or the OpenVINO "
              "export on Intel CPUs.")


if __name__ == "__main__":
    main()
