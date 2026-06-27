"""YOLO detector wrapper.

Uses Ultralytics so we get correct letterbox preprocessing + NMS for free, and
so the SAME code works whether `model_path` points at a PyTorch `.pt`, an
exported `.onnx`, or an OpenVINO model directory. We force CPU and small thread
counts to stay friendly on a 2-core / 4GB box.

The detector is class-agnostic: it filters to whatever class *names* you ask for
(default: ["car"]). Adding "truck", "bus", "person", ... requires no code change.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("detector")

# Stable, readable colors per class (BGR). Falls back to a hash-based color.
_PALETTE = [
    (0, 200, 0), (0, 140, 255), (255, 120, 0), (200, 0, 200),
    (0, 0, 230), (180, 180, 0), (0, 180, 180),
]


def _color_for(name: str) -> Tuple[int, int, int]:
    return _PALETTE[hash(name) % len(_PALETTE)]


class Detection:
    """A single detected object: class name, confidence, pixel box, and an
    optional stable track id (stamped by the tracker, ``None`` until then)."""

    __slots__ = ("cls", "conf", "box", "id")

    def __init__(self, cls: str, conf: float, box: Sequence[float],
                 id: Optional[int] = None) -> None:
        self.cls = cls
        self.conf = float(conf)
        self.box = [float(b) for b in box]  # x1,y1,x2,y2
        self.id = id  # stable per-object track id, or None if untracked

    def as_dict(self) -> Dict[str, object]:
        """Serialize to the JSON shape the frontend expects (rounded)."""
        return {"cls": self.cls, "conf": round(self.conf, 3),
                "box": [round(v, 1) for v in self.box], "id": self.id}


class Detector:
    """Thread-safe wrapper around an Ultralytics YOLO model.

    Loads a ``.pt`` / ``.onnx`` / OpenVINO model on CPU, serializes concurrent
    ``predict()`` calls with a lock, and returns filtered :class:`Detection`
    objects. For static ONNX models the input size is read from the graph so a
    mismatched ``imgsz`` in config can never crash inference.
    """

    def __init__(self, model_path: str, imgsz: int = 640, iou: float = 0.45,
                 max_det: int = 100, num_threads: int = 2) -> None:
        from ultralytics import YOLO  # imported lazily so the module loads fast

        try:
            import torch
            torch.set_num_threads(max(1, num_threads))
        except Exception:  # torch always ships with ultralytics, but be safe
            pass
        cv2.setNumThreads(max(1, num_threads))

        logger.info("Loading model: %s", model_path)
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        # Static ONNX / OpenVINO models are locked to the size they were exported
        # at. Read that size and use it, so a mismatched IMGSZ in config can never
        # crash inference (it just uses the model's real size).
        mp = str(model_path).lower()
        baked = None
        if mp.endswith(".onnx"):
            baked = self._onnx_input_size(model_path)
        elif mp.endswith("_openvino_model") or mp.endswith("_openvino_model/"):
            baked = self._openvino_input_size(model_path)
        if baked:
            if baked != imgsz:
                logger.warning("Model is built for imgsz=%d; using that instead "
                               "of configured %d. Re-export to change.",
                               baked, imgsz)
            self.imgsz = baked
        self.iou = iou
        self.max_det = max_det
        # name -> id, lowercased for forgiving matching
        self.names: Dict[int, str] = dict(self.model.names)
        self._name_to_id = {v.lower(): k for k, v in self.names.items()}
        # Ultralytics models are not guaranteed thread-safe across concurrent
        # predict() calls; on a 2-core box we serialize them with a lock.
        self._lock = threading.Lock()
        # Warm up once so the first real frame isn't slow.
        self._warmup()

    @staticmethod
    def _onnx_input_size(path: str) -> Optional[int]:
        """Read a static ONNX's square input size (H==W) from its graph, or None
        if the model is dynamic / unreadable."""
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape  # [N, C, H, W]
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
                return int(h)
        except Exception as e:
            logger.warning("Could not read ONNX input size: %s", e)
        return None

    @staticmethod
    def _openvino_input_size(path) -> Optional[int]:
        """Read the square input size from an OpenVINO export's metadata.yaml
        (Ultralytics writes imgsz there), or None if unreadable."""
        try:
            import yaml
            meta = Path(path) / "metadata.yaml"
            if meta.exists():
                data = yaml.safe_load(meta.read_text()) or {}
                sz = data.get("imgsz")
                if isinstance(sz, (list, tuple)):
                    sz = sz[0] if sz else None
                if isinstance(sz, int) and sz > 0:
                    return int(sz)
        except Exception as e:
            logger.warning("Could not read OpenVINO input size: %s", e)
        return None

    @property
    def available_classes(self) -> List[str]:
        """All class names the loaded model can predict, sorted."""
        return sorted(self.names.values())

    def class_ids_for(self, class_names: Sequence[str]) -> List[int]:
        """Map class names to their model ids (case-insensitive); skip unknowns."""
        ids: List[int] = []
        for n in class_names:
            cid = self._name_to_id.get(n.strip().lower())
            if cid is not None:
                ids.append(cid)
            else:
                logger.warning("Unknown class name ignored: %s", n)
        return ids

    def _warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            with self._lock:
                self.model.predict(dummy, imgsz=self.imgsz, conf=0.5,
                                   verbose=False, device="cpu")
        except Exception as e:  # don't let warmup kill startup
            logger.warning("Warmup failed (non-fatal): %s", e)

    def detect(self, frame: np.ndarray, conf: float,
               class_ids: Optional[List[int]] = None,
               min_box_area_ratio: float = 0.0) -> List[Detection]:
        """Run detection on a single BGR frame and return filtered detections."""
        with self._lock:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=conf,
                iou=self.iou,
                max_det=self.max_det,
                classes=class_ids if class_ids else None,
                verbose=False,
                device="cpu",
            )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out

        frame_area = float(frame.shape[0] * frame.shape[1]) or 1.0
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, confs, clss):
            if min_box_area_ratio > 0:
                w = max(0.0, box[2] - box[0])
                h = max(0.0, box[3] - box[1])
                if (w * h) / frame_area < min_box_area_ratio:
                    continue
            out.append(Detection(self.names.get(int(k), str(k)), c, box))
        return out

    @staticmethod
    def annotate(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw boxes + labels onto the given frame (mutated in place) and return
        it. Box thickness and font scale adapt to the frame height."""
        h = frame.shape[0]
        thickness = max(1, round(h / 400))
        font_scale = max(0.4, h / 1200)
        for d in detections:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            color = _color_for(d.cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # Show the stable track id (e.g. "car #42 88%") once assigned.
            if d.id is not None:
                label = f"{d.cls} #{d.id} {d.conf * 100:.0f}%"
            else:
                label = f"{d.cls} {d.conf * 100:.0f}%"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            ytop = max(0, y1 - th - baseline - 2)
            cv2.rectangle(frame, (x1, ytop), (x1 + tw + 2, ytop + th + baseline + 2),
                          color, -1)
            cv2.putText(frame, label, (x1 + 1, ytop + th),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return frame
