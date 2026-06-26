"""Self-contained tests for the tracker + interactive geometry counting.

No server, no model, no network — just the pure logic:
  (a) stable ids across frames as objects move;
  (b) a single object crossing a directed line counts in==1/out==0, and the
      reverse direction counts out==1/in==0;
  (c) a point inside a square zone registers occupancy 1 (and dwell on exit).

Run inside the backend venv:
    .venv/bin/python scripts/test_tracker.py
Prints PASS/FAIL per check and exits nonzero on any failure.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `app` importable regardless of launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detector import Detection                       # noqa: E402
from app.tracker import CentroidTracker                  # noqa: E402
from app.stream_manager import (                         # noqa: E402
    _line_side, _segments_intersect, _point_in_polygon)


_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"  -> {detail}"
    print(line)
    if not ok:
        _failures += 1


def _box_at(cx: float, cy: float, half: float = 10.0):
    """A square box of side 2*half centered at (cx, cy)."""
    return [cx - half, cy - half, cx + half, cy + half]


# ----------------------------------------------------------------------------
# (a) Stable ids across frames
# ----------------------------------------------------------------------------
def test_stable_ids() -> None:
    tracker = CentroidTracker(max_age=30)

    # Two objects moving steadily to the right; ids must stay constant.
    a_id = None
    b_id = None
    first_a = first_b = None
    for step in range(8):
        a = Detection("car", 0.9, _box_at(50 + step * 8, 100))
        b = Detection("car", 0.8, _box_at(200 + step * 8, 300))
        tracker.update([a, b])
        if step == 0:
            first_a, first_b = a.id, b.id
        a_id, b_id = a.id, b.id

    stable = (a_id == first_a and b_id == first_b
              and first_a is not None and first_b is not None
              and first_a != first_b)
    check("stable ids: two objects keep their ids across 8 frames",
          stable, f"a={first_a}->{a_id}, b={first_b}->{b_id}")

    # unique_total should be exactly 2 (no spurious new tracks created).
    check("stable ids: unique_total == 2 (no id churn)",
          tracker.unique_total() == 2, f"unique_total={tracker.unique_total()}")

    # Both seen this frame -> 2 active.
    check("stable ids: unique_active == 2",
          tracker.unique_active() == 2, f"unique_active={tracker.unique_active()}")

    # by-class accounting.
    check("stable ids: unique_by_class == {'car': 2}",
          tracker.unique_by_class() == {"car": 2},
          f"{tracker.unique_by_class()}")


def test_new_id_for_new_object() -> None:
    tracker = CentroidTracker(max_age=30)
    d1 = Detection("car", 0.9, _box_at(50, 50))
    tracker.update([d1])
    # A second, far-away object on the next frame should get a fresh id.
    d1b = Detection("car", 0.9, _box_at(58, 50))
    d2 = Detection("car", 0.9, _box_at(400, 400))
    tracker.update([d1b, d2])
    check("new object gets a fresh distinct id",
          d1b.id == d1.id and d2.id is not None and d2.id != d1.id,
          f"obj1={d1.id}->{d1b.id}, obj2={d2.id}")
    check("unique_total == 2 after a new object appears",
          tracker.unique_total() == 2, f"{tracker.unique_total()}")


def test_eviction() -> None:
    tracker = CentroidTracker(max_age=3)
    d = Detection("car", 0.9, _box_at(50, 50))
    tracker.update([d])
    first = d.id
    # Feed empty frames; after max_age missed updates the track is evicted.
    for _ in range(5):
        tracker.update([])
    check("track evicted after max_age missed updates",
          len(tracker.tracks) == 0, f"tracks alive={len(tracker.tracks)}")
    # A reappearing object should get a NEW id (old one was evicted).
    d2 = Detection("car", 0.9, _box_at(50, 50))
    tracker.update([d2])
    check("reappearing object after eviction gets a new id",
          d2.id is not None and d2.id != first, f"old={first}, new={d2.id}")


# ----------------------------------------------------------------------------
# geometry primitives sanity (cheap, deterministic)
# ----------------------------------------------------------------------------
def test_geometry_primitives() -> None:
    # Vertical line A=(50,0) -> B=(50,100). side = (B-A) x (P-A).
    # For B-A=(0,100): point to the right (x>50) has cross<0; left has cross>0.
    left = _line_side(50, 0, 50, 100, 10, 50)
    right = _line_side(50, 0, 50, 100, 90, 50)
    check("line_side: left/right of a vertical line have opposite signs",
          (left > 0) != (right > 0), f"left={left}, right={right}")

    # Segment intersection: a horizontal path from x=10 to x=90 at y=50 crosses
    # the vertical segment (50,0)-(50,100).
    crosses = _segments_intersect(50, 0, 50, 100, 10, 50, 90, 50)
    check("segments_intersect: crossing path detected", crosses)

    # But a path entirely left of the segment must NOT intersect it.
    no_cross = _segments_intersect(50, 0, 50, 100, 10, 50, 40, 50)
    check("segments_intersect: non-crossing path rejected", not no_cross)

    # Segment-bounded: a path crossing the infinite line ABOVE the segment
    # (y=-50, outside the 0..100 span) must NOT count.
    above = _segments_intersect(50, 0, 50, 100, 10, -50, 90, -50)
    check("segments_intersect: crossing only the infinite extension rejected",
          not above)

    # Point in polygon: unit-ish square.
    square = [[0, 0], [100, 0], [100, 100], [0, 100]]
    check("point_in_polygon: inside point", _point_in_polygon(50, 50, square))
    check("point_in_polygon: outside point",
          not _point_in_polygon(150, 50, square))


# ----------------------------------------------------------------------------
# (b) line crossing in/out, via a real Session (no worker thread started)
# ----------------------------------------------------------------------------
def _make_session():
    """Build a Session object WITHOUT starting its worker thread, with a fixed
    frame size so normalized geometry maps to known pixels."""
    from app.config import Settings
    from app.stream_manager import Session
    from app.schemas import SourceType

    settings = Settings()  # defaults are fine; we never run the worker
    # detector=None is safe: we call _record_detection directly, never detect().
    sess = Session("test", SourceType.upload, "x", detector=None,
                   settings=settings)
    sess.width = 100
    sess.height = 100
    return sess


def test_line_crossing_in() -> None:
    sess = _make_session()
    # Vertical line down the middle, A=(0.5,0)->B=(0.5,1) in normalized coords.
    # With B-A pointing +y, a point on the LEFT (x<0.5) has side>0 and the
    # RIGHT (x>0.5) has side<0. So moving RIGHT->LEFT is side<0 -> side>0 == IN.
    sess.update_settings(lines=[{"id": "L1", "name": "mid",
                                 "points": [[0.5, 0.0], [0.5, 1.0]]}])

    # Object starts on the right (x=80) and moves left across to x=20.
    xs = [80, 65, 45, 20]
    for x in xs:
        d = Detection("car", 0.9, _box_at(x, 50))
        sess._record_detection([d])

    lc = sess.stats()["line_counts"][0]
    check("line crossing IN: right->left counts in==1, out==0",
          lc["in"] == 1 and lc["out"] == 0,
          f"in={lc['in']}, out={lc['out']}, total={lc['total']}")
    check("line crossing IN: by_class records the car",
          lc["by_class"].get("car") == 1, f"by_class={lc['by_class']}")


def test_line_crossing_out() -> None:
    sess = _make_session()
    sess.update_settings(lines=[{"id": "L1", "name": "mid",
                                 "points": [[0.5, 0.0], [0.5, 1.0]]}])
    # Reverse direction: left (x=20) -> right (x=80) is side>0 -> side<0 == OUT.
    xs = [20, 35, 55, 80]
    for x in xs:
        d = Detection("car", 0.9, _box_at(x, 50))
        sess._record_detection([d])

    lc = sess.stats()["line_counts"][0]
    check("line crossing OUT: left->right counts out==1, in==0",
          lc["out"] == 1 and lc["in"] == 0,
          f"in={lc['in']}, out={lc['out']}, total={lc['total']}")


def test_line_crossing_counted_once() -> None:
    sess = _make_session()
    sess.update_settings(lines=[{"id": "L1", "name": "mid",
                                 "points": [[0.5, 0.0], [0.5, 1.0]]}])
    # Object crosses then keeps going (extra frames on the far side): still 1.
    xs = [80, 60, 40, 20, 15, 12, 10]
    for x in xs:
        sess._record_detection([Detection("car", 0.9, _box_at(x, 50))])
    lc = sess.stats()["line_counts"][0]
    check("line crossing counted once per (track,line)",
          lc["total"] == 1, f"total={lc['total']}")


def test_line_segment_bounded() -> None:
    sess = _make_session()
    # Short line covering only the TOP region: y in [0.0, 0.3]. An object
    # crossing the vertical x=0.5 at y=50 (mid frame) is BELOW the segment, so
    # it must NOT count (crossing only the infinite extension).
    sess.update_settings(lines=[{"id": "L1", "name": "top",
                                 "points": [[0.5, 0.0], [0.5, 0.3]]}])
    for x in [80, 50, 20]:
        sess._record_detection([Detection("car", 0.9, _box_at(x, 50))])
    lc = sess.stats()["line_counts"][0]
    check("segment-bounded: crossing below the segment is not counted",
          lc["total"] == 0, f"total={lc['total']}")


# ----------------------------------------------------------------------------
# (c) zone occupancy + dwell
# ----------------------------------------------------------------------------
def test_zone_occupancy() -> None:
    sess = _make_session()
    # Square zone covering the center: x,y in [0.4, 0.6] -> pixels [40,60].
    sess.update_settings(zones=[{"id": "Z1", "name": "box",
                                 "points": [[0.4, 0.4], [0.6, 0.4],
                                            [0.6, 0.6], [0.4, 0.6]]}])
    # Object sits at center (50,50) -> inside the zone.
    sess._record_detection([Detection("car", 0.9, _box_at(50, 50))])
    zo = sess.stats()["zone_occupancy"][0]
    check("zone occupancy: point inside square counts as 1",
          zo["count"] == 1, f"count={zo['count']}")
    check("zone occupancy: peak == 1", zo["peak"] == 1, f"peak={zo['peak']}")

    # Object outside the zone -> occupancy drops to 0, peak stays 1.
    sess._record_detection([Detection("car", 0.9, _box_at(90, 90))])
    zo = sess.stats()["zone_occupancy"][0]
    check("zone occupancy: leaving drops count to 0, peak stays 1",
          zo["count"] == 0 and zo["peak"] == 1,
          f"count={zo['count']}, peak={zo['peak']}")
    check("zone occupancy: dwell recorded on exit (avg_dwell_s >= 0)",
          zo["avg_dwell_s"] >= 0.0, f"avg_dwell_s={zo['avg_dwell_s']}")


def test_geometry_echoed() -> None:
    sess = _make_session()
    sess.update_settings(
        lines=[{"id": "L1", "name": "mid", "points": [[0.5, 0.0], [0.5, 1.0]]}],
        zones=[{"id": "Z1", "name": "box",
                "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]}])
    st = sess.stats()
    ok = (len(st["lines"]) == 1 and st["lines"][0]["id"] == "L1"
          and len(st["zones"]) == 1 and st["zones"][0]["id"] == "Z1")
    check("geometry echoed back in stats for overlay re-sync", ok,
          f"lines={st['lines']}, zones={st['zones']}")


def main() -> int:
    print("=== tracker + geometry tests ===")
    test_stable_ids()
    test_new_id_for_new_object()
    test_eviction()
    test_geometry_primitives()
    test_line_crossing_in()
    test_line_crossing_out()
    test_line_crossing_counted_once()
    test_line_segment_bounded()
    test_zone_occupancy()
    test_geometry_echoed()
    print("=" * 33)
    if _failures:
        print(f"RESULT: {_failures} check(s) FAILED")
        return 1
    print("RESULT: all checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
