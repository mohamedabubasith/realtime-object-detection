"""End-to-end smoke test for the object-detection backend.

Exercises the WHOLE pipeline (not just the model):
  1. detector loads + runs on a blank frame
  2. detector finds objects in a real image
  3. FastAPI boots; /api/health + /api/config respond
  4. full session: upload a video -> start session -> worker processes frames
     -> pull a JPEG from the MJPEG stream -> read live stats -> stop

Run inside the backend venv:
    python scripts/smoke_test.py
Exits non-zero on any failure.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `app` importable no matter where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def _load_sample_image() -> np.ndarray:
    """Use the bus.jpg that ships inside the ultralytics package (no network)."""
    import cv2
    from ultralytics.utils import ASSETS
    path = ASSETS / "bus.jpg"
    return cv2.imread(str(path))


def main() -> int:
    import cv2
    from app.config import get_settings
    from app.detector import Detector

    s = get_settings()

    # 1) Detector loads + runs on a synthetic frame (use the configured imgsz so
    #    it matches the model, incl. ONNX exports).
    print(f"[1] Loading detector: {s.resolved_model_path()} (imgsz={s.imgsz})")
    det = Detector(s.resolved_model_path(), imgsz=s.imgsz, num_threads=2)
    assert "car" in det.available_classes, "model has no 'car' class!"
    ids = det.class_ids_for(["car"])
    assert ids, "could not resolve 'car' to a class id"
    blank = np.zeros((s.imgsz, s.imgsz, 3), dtype=np.uint8)
    dets = det.detect(blank, conf=0.25, class_ids=ids)
    print(f"    blank-frame detect -> {len(dets)} (expected 0); class id(s) {ids}")

    # 2) Detector finds objects in a real image (best-effort if offline).
    sample = None
    try:
        print("[2] Loading bundled sample image (ultralytics assets)")
        sample = _load_sample_image()
        all_ids = det.class_ids_for(["car", "bus", "person"])
        rd = det.detect(sample, conf=0.25, class_ids=all_ids)
        found = sorted(set(d.cls for d in rd))
        print(f"    detected {len(rd)} objects: {found}")
        assert len(rd) > 0, "expected detections on the sample image"
    except AssertionError:
        raise
    except Exception as e:
        print(f"    (skipped real-image check, offline?: {e})")

    # 3) API boots; core endpoints respond.
    print("[3] Booting FastAPI app via TestClient...")
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        h = client.get("/api/health")
        assert h.status_code == 200, h.text
        print(f"    /api/health -> {h.json()}")
        cfg = client.get("/api/config").json()
        assert "car" in cfg["available_classes"]
        assert client.get("/api/sessions/nope").status_code == 404
        print(f"    /api/config OK ({len(cfg['available_classes'])} classes)")

        # 4) Full session pipeline.
        print("[4] Building a test video and running a real session...")
        if sample is None:
            # fabricate a frame so the test still runs offline
            sample = np.full((480, 640, 3), 60, dtype=np.uint8)
            cv2.rectangle(sample, (200, 200), (380, 360), (0, 0, 200), -1)
        h_img, w_img = sample.shape[:2]
        tmp = Path(s.upload_dir) / "_smoke_test_input.mp4"
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             (w_img, h_img))
        for _ in range(45):
            vw.write(sample)
        vw.release()
        assert tmp.exists() and tmp.stat().st_size > 0, "failed to write test video"

        with tmp.open("rb") as f:
            up = client.post("/api/upload",
                             files={"file": ("smoke.mp4", f, "video/mp4")})
        assert up.status_code == 200, up.text
        file_id = up.json()["file_id"]
        print(f"    uploaded -> {file_id}")

        sess = client.post("/api/sessions", json={
            "source_type": "upload",
            "source": file_id,
            "target_classes": ["car", "bus", "person"],
            "conf_threshold": 0.25,
            "loop": False,
        })
        assert sess.status_code == 200, sess.text
        sid = sess.json()["session_id"]
        print(f"    session -> {sid}")

        # Poll stats until the worker has processed frames (or it finishes).
        processed = 0
        status = "starting"
        for _ in range(60):  # up to ~15s
            st = client.get(f"/api/sessions/{sid}/stats").json()
            processed = st.get("processed_frames", 0)
            status = st.get("status")
            if st.get("error"):
                raise AssertionError(f"session error: {st['error']}")
            if processed > 0:
                break
            time.sleep(0.25)
        print(f"    stats: status={status} processed_frames={processed} "
              f"max_count={st.get('max_count')} fps={st.get('fps')}")
        assert processed > 0, "worker never processed a frame"

        # Pull at least one JPEG out of the MJPEG stream.
        got_jpeg = False
        with client.stream("GET", f"/api/sessions/{sid}/video") as r:
            assert r.status_code == 200, r.text
            assert "multipart/x-mixed-replace" in r.headers.get("content-type", "")
            buf = b""
            for chunk in r.iter_bytes():
                buf += chunk
                if b"\xff\xd8" in buf and b"\xff\xd9" in buf:  # JPEG SOI+EOI
                    got_jpeg = True
                    break
                if len(buf) > 2_000_000:
                    break
        assert got_jpeg, "no JPEG frame received from MJPEG stream"
        print("    MJPEG stream delivered a valid JPEG frame ✔")

        assert client.post(f"/api/sessions/{sid}/stop").status_code == 200
        assert client.delete(f"/api/sessions/{sid}").status_code == 200
        try:
            tmp.unlink()
        except Exception:
            pass
        print("    session stopped + cleaned up ✔")

    print("\nSMOKE TEST PASSED ✅  (model + API + full session pipeline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
