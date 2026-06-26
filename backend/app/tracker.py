"""A tiny, dependency-light multi-object tracker (CPU-cheap).

Why this exists
---------------
We want STABLE per-object ids so the geometry layer (line crossing, zone
occupancy/dwell) can reason about "the same car over time" — WITHOUT paying for a
second inference pass and WITHOUT switching Ultralytics to its ``.track()`` API
(which forces the slow PyTorch path and breaks our ONNX-CPU optimization).

So this tracker works ONLY on the :class:`~app.detector.Detection` boxes the
detector already produced, on the frames we already process. It is a classic
centroid/IoU tracker:

  * Greedy IoU matching between the current frame's boxes and the active tracks.
  * Fallback to nearest-centroid matching within a distance gate for the
    leftovers (handles fast motion where boxes stop overlapping).
  * Unmatched detections spawn new tracks with a fresh incrementing id.
  * Tracks that go unmatched for ``max_age`` updates are evicted.

Cost is O(tracks x dets) per processed frame — for the dozens of objects we ever
see this is negligible next to the detector itself.

Each :class:`Track` also carries a ``bookkeeping`` dict that the owning Session
uses to remember per-(line/zone) state (last side of a line, whether a
(track,line) crossing was already counted, which zones it currently sits in,
when it entered a zone). The tracker never looks inside that dict; it just keeps
it alive for as long as the track lives.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection


def _iou(a: List[float], b: List[float]) -> float:
    """Intersection-over-union of two ``[x1,y1,x2,y2]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _centroid(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _diag(box: List[float]) -> float:
    """Box diagonal length — used to scale the nearest-centroid distance gate so
    a big object is allowed to move farther between frames than a small one."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return float(np.hypot(w, h))


class Track:
    """One tracked object's live state.

    ``bookkeeping`` is an opaque per-track scratch dict owned by the Session for
    line/zone state; the tracker only keeps it alive across updates.
    """

    __slots__ = ("id", "cls", "box", "centroid", "prev_centroid",
                 "hits", "misses", "bookkeeping")

    def __init__(self, tid: int, det: Detection) -> None:
        self.id = tid
        self.cls = det.cls
        self.box = list(det.box)
        c = _centroid(self.box)
        self.centroid: Tuple[float, float] = c
        self.prev_centroid: Tuple[float, float] = c
        self.hits = 1
        self.misses = 0
        self.bookkeeping: Dict[str, object] = {}

    def update(self, det: Detection) -> None:
        """Fold a matched detection into this track (advances centroid history)."""
        self.prev_centroid = self.centroid
        self.box = list(det.box)
        self.centroid = _centroid(self.box)
        self.cls = det.cls
        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        self.misses += 1


class CentroidTracker:
    """Greedy IoU + nearest-centroid multi-object tracker.

    Parameters
    ----------
    max_age:
        Evict a track after this many consecutive missed updates.
    iou_threshold:
        Minimum IoU for a box<->track match in the first (greedy IoU) pass.
    max_centroid_distance_ratio:
        Distance gate for the fallback nearest-centroid pass, expressed as a
        multiple of the track box diagonal. Keeps a far-away detection from
        being glued onto an unrelated track.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3,
                 max_centroid_distance_ratio: float = 1.5) -> None:
        self.max_age = int(max_age)
        self.iou_threshold = float(iou_threshold)
        self.max_centroid_distance_ratio = float(max_centroid_distance_ratio)
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}
        # cumulative distinct tracks ever created (the "unique total")
        self.total_created = 0
        # cumulative distinct tracks ever created, per class name
        self.created_by_class: Dict[str, int] = {}

    @property
    def tracks(self) -> List[Track]:
        """All currently-alive tracks (including ones missed this frame)."""
        return list(self._tracks.values())

    def _new_track(self, det: Detection) -> Track:
        tid = self._next_id
        self._next_id += 1
        tr = Track(tid, det)
        self._tracks[tid] = tr
        self.total_created += 1
        self.created_by_class[det.cls] = self.created_by_class.get(det.cls, 0) + 1
        return tr

    def update(self, detections: List[Detection]) -> List[Track]:
        """Match detections to tracks, stamp each Detection's ``.id``, and return
        the tracks that are *active this frame* (matched or freshly created).

        Mutates the passed ``Detection`` objects in place (sets ``.id``).
        """
        track_ids = list(self._tracks.keys())

        # No live tracks: every detection becomes a new track.
        if not track_ids:
            active: List[Track] = []
            for det in detections:
                tr = self._new_track(det)
                det.id = tr.id
                active.append(tr)
            # nothing to age out (no pre-existing tracks)
            return active

        # No detections: age every track, evict the stale, return nothing active.
        if not detections:
            for tr in self._tracks.values():
                tr.mark_missed()
            self._evict()
            return []

        n_t = len(track_ids)
        n_d = len(detections)

        # ---- Pass 1: greedy IoU matching -----------------------------------
        # Build an IoU matrix (tracks x dets), then repeatedly take the global
        # best remaining pair above threshold. Greedy is plenty for our scale.
        iou_mat = np.zeros((n_t, n_d), dtype=np.float32)
        for ti, tid in enumerate(track_ids):
            tbox = self._tracks[tid].box
            for di, det in enumerate(detections):
                iou_mat[ti, di] = _iou(tbox, det.box)

        matched_tracks: Dict[int, int] = {}   # track-row -> det-col
        used_tracks = [False] * n_t
        used_dets = [False] * n_d

        # Greedy: pick the largest IoU pair each iteration.
        while True:
            ti, di = np.unravel_index(int(np.argmax(iou_mat)), iou_mat.shape)
            best = iou_mat[ti, di]
            if best < self.iou_threshold:
                break
            matched_tracks[ti] = di
            used_tracks[ti] = True
            used_dets[di] = True
            iou_mat[ti, :] = -1.0   # remove this track row
            iou_mat[:, di] = -1.0   # remove this det col

        # ---- Pass 2: nearest-centroid fallback for leftovers ---------------
        # For any still-unmatched track, find the closest still-unmatched
        # detection within a box-scaled distance gate.
        for ti, tid in enumerate(track_ids):
            if used_tracks[ti]:
                continue
            tr = self._tracks[tid]
            tcx, tcy = tr.centroid
            gate = max(1.0, _diag(tr.box) * self.max_centroid_distance_ratio)
            best_di = -1
            best_dist = gate
            for di, det in enumerate(detections):
                if used_dets[di]:
                    continue
                dcx, dcy = _centroid(det.box)
                dist = float(np.hypot(tcx - dcx, tcy - dcy))
                if dist < best_dist:
                    best_dist = dist
                    best_di = di
            if best_di >= 0:
                matched_tracks[ti] = best_di
                used_tracks[ti] = True
                used_dets[best_di] = True

        # ---- Apply matches, spawn new tracks, age the unmatched ------------
        active: List[Track] = []
        for ti, tid in enumerate(track_ids):
            tr = self._tracks[tid]
            if ti in matched_tracks:
                det = detections[matched_tracks[ti]]
                tr.update(det)
                det.id = tr.id
                active.append(tr)
            else:
                tr.mark_missed()

        for di, det in enumerate(detections):
            if used_dets[di]:
                continue
            tr = self._new_track(det)
            det.id = tr.id
            active.append(tr)

        self._evict()
        return active

    def _evict(self) -> None:
        """Drop tracks that have been missed for more than ``max_age`` frames."""
        stale = [tid for tid, tr in self._tracks.items()
                 if tr.misses > self.max_age]
        for tid in stale:
            del self._tracks[tid]

    # ------------------------------------------------------------- stats helpers
    def unique_total(self) -> int:
        """Cumulative distinct tracks ever created."""
        return self.total_created

    def unique_active(self) -> int:
        """Tracks that were matched/seen in the most recent processed frame."""
        return sum(1 for tr in self._tracks.values() if tr.misses == 0)

    def unique_by_class(self) -> Dict[str, int]:
        """Cumulative distinct tracks ever created, keyed by class name."""
        return dict(self.created_by_class)

    def reset(self) -> None:
        """Forget all tracks and counters (rarely needed; new sessions get a
        fresh tracker instead)."""
        self._tracks.clear()
        self._next_id = 1
        self.total_created = 0
        self.created_by_class.clear()
