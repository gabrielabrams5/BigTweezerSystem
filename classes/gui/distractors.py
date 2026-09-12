"""distractors — low-rate global blob tracking of everything that isn't
the selected robot.

You cannot exclude a neighbour you aren't modelling. The per-object
tracker only ever looks inside its own search window, so it has no
representation of the other beads on the slide; when one drifts into the
window it becomes an equally-good correlation peak with nothing to
distinguish it.

This module runs the *same* mask the operator already tuned over the
**whole** frame at a low rate (5–10 Hz is plenty — beads are slow), keeps
a lightweight constant-velocity track per blob, and hands out predicted
positions so the tracker can suppress them on its correlation surface.

Two appearance features come along for free because the pipeline already
computes them per-blob:

* **area** — bead diameter CV is a few percent, so a per-track running
  mean separates populations even though any single frame is noise.
* **Laplacian variance** — beads at different z have genuinely different
  focus. Also a running mean.

Both are used as association tie-breakers, not as hard gates.

Threading: ``DistractorWorker`` owns a daemon thread with drop-old frame
semantics, identical in spirit to ``path_follow._TrackerAsync``. The
global pass is ~10 ms on a multi-megapixel frame using
``cv2.connectedComponentsWithStats`` (C++, single pass) — scipy's
``ndimage.label`` is several times slower and is deliberately not used
here, unlike the small-window helpers in ``tracker.py``.

Run with:  python3 -m pytest classes/gui/test_distractors.py -v
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from classes.gui.motion_filter import ConstantVelocityKF
from classes.gui.tracker import MaskParams, _apply_mask


@dataclass
class Blob:
    pos: Tuple[float, float]
    area: float
    blur: float = 0.0


@dataclass
class DistractorTrack:
    """One non-selected object, tracked loosely."""
    kf: ConstantVelocityKF
    area_mean: float
    blur_mean: float
    last_t: float
    hits: int = 1
    misses: int = 0
    tid: int = 0

    def predict(self, t: float) -> Tuple[Tuple[float, float], float]:
        return self.kf.project(max(0.0, t - self.last_t))


def find_blobs(gray: np.ndarray,
               params: MaskParams,
               max_blobs: int = 64,
               with_blur: bool = True) -> List[Blob]:
    """Global mask → connected components → blob list, largest first.

    Uses ``cv2.connectedComponentsWithStats`` rather than
    ``scipy.ndimage.label``: on a full multi-megapixel frame the scipy
    path costs tens of milliseconds, which is too much even at 5 Hz.
    """
    mask = _apply_mask(gray, params)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    out: List[Blob] = []
    for i in range(1, n):                      # 0 is background
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < params.min_blob_area_px:
            continue
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        blur = 0.0
        if with_blur:
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            patch = gray[y:y + h, x:x + w]
            if patch.size >= 9:
                blur = float(cv2.Laplacian(patch, cv2.CV_64F).var())
        out.append(Blob(pos=(cx, cy), area=area, blur=blur))
    out.sort(key=lambda b: -b.area)
    return out[:max_blobs]


class DistractorField:
    """Maintains constant-velocity tracks over every masked blob.

    Pure logic — feed it grayscale frames and timestamps. Association is
    nearest-neighbour inside ``gate_px``, with area/blur similarity used
    to break ties between candidates that are both in gate.
    """

    def __init__(self,
                 mask_params: MaskParams,
                 gate_px: float = 40.0,
                 max_misses: int = 5,
                 max_blobs: int = 64,
                 accel_noise_px_s2: float = 400.0,
                 feature_weight: float = 0.35):
        self.mask_params = mask_params
        self.gate_px = float(gate_px)
        self.max_misses = int(max_misses)
        self.max_blobs = int(max_blobs)
        self.accel_noise = float(accel_noise_px_s2)
        self.feature_weight = float(feature_weight)
        self.tracks: List[DistractorTrack] = []
        self._next_tid = 1

    def set_mask_params(self, params: MaskParams) -> None:
        self.mask_params = params

    # ---- association --------------------------------------------------

    def _cost(self, track: DistractorTrack, blob: Blob,
              pred: Tuple[float, float]) -> float:
        """Gated distance cost; lower is better, inf means out of gate."""
        d = float(np.hypot(blob.pos[0] - pred[0], blob.pos[1] - pred[1]))
        if d > self.gate_px:
            return float("inf")
        # Normalize appearance mismatch to a 0..1-ish scale and fold it in
        # as a fraction of the gate, so it only ever decides ties.
        denom = max(track.area_mean, 1.0)
        area_err = abs(blob.area - track.area_mean) / denom
        if track.blur_mean > 1e-6:
            blur_err = abs(blob.blur - track.blur_mean) / track.blur_mean
        else:
            blur_err = 0.0
        feat = min(1.0, 0.5 * area_err + 0.5 * blur_err)
        return d + self.feature_weight * self.gate_px * feat

    def update(self, gray: np.ndarray, t: float) -> None:
        """Run one global pass and advance every track."""
        blobs = find_blobs(gray, self.mask_params, self.max_blobs)
        preds = [tr.predict(t)[0] for tr in self.tracks]

        # Greedy nearest-neighbour over all (track, blob) pairs. N is
        # small (tens), so an O(N·M) sort beats pulling in an assignment
        # solver.
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for bi, b in enumerate(blobs):
                c = self._cost(tr, b, preds[ti])
                if c < float("inf"):
                    pairs.append((c, ti, bi))
        pairs.sort(key=lambda p: p[0])

        used_t, used_b = set(), set()
        for _c, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            tr, b = self.tracks[ti], blobs[bi]
            dt = max(0.0, t - tr.last_t)
            tr.kf.step(dt, None, b.pos)
            tr.area_mean += 0.2 * (b.area - tr.area_mean)
            tr.blur_mean += 0.2 * (b.blur - tr.blur_mean)
            tr.last_t = t
            tr.hits += 1
            tr.misses = 0

        for ti, tr in enumerate(self.tracks):
            if ti in used_t:
                continue
            dt = max(0.0, t - tr.last_t)
            tr.kf.step(dt, None, None)
            tr.last_t = t
            tr.misses += 1

        for bi, b in enumerate(blobs):
            if bi in used_b:
                continue
            self.tracks.append(DistractorTrack(
                kf=ConstantVelocityKF(
                    b.pos, accel_noise_px_s2=self.accel_noise,
                    meas_noise_px=3.0),
                area_mean=b.area, blur_mean=b.blur, last_t=t,
                tid=self._next_tid))
            self._next_tid += 1

        self.tracks = [tr for tr in self.tracks
                       if tr.misses <= self.max_misses]

    # ---- output -------------------------------------------------------

    def predicted(self, t: float,
                  exclude_xy: Optional[Tuple[float, float]] = None,
                  exclude_radius_px: float = 25.0,
                  min_hits: int = 2) -> List[Tuple[float, float, float]]:
        """Predicted ``(x, y, sigma)`` for every confirmed track.

        ``exclude_xy`` drops the track that *is* the selected robot — the
        global pass has no idea which blob the operator picked, so the
        caller passes the robot's own predicted position and anything
        within ``exclude_radius_px`` is filtered out.
        """
        out = []
        for tr in self.tracks:
            if tr.hits < min_hits:
                continue
            (px, py), sigma = tr.predict(t)
            if exclude_xy is not None:
                d = np.hypot(px - exclude_xy[0], py - exclude_xy[1])
                if d < exclude_radius_px:
                    continue
            out.append((float(px), float(py), float(sigma)))
        return out


class DistractorWorker:
    """Runs a :class:`DistractorField` on its own thread at a capped rate.

    Drop-old: ``submit`` replaces any queued frame, so a slow global pass
    can never build a backlog. ``snapshot`` is safe to call from the Qt
    thread at any time.
    """

    def __init__(self,
                 field: DistractorField,
                 rate_hz: float = 8.0,
                 log_fn=print):
        self.field = field
        self.min_interval = 1.0 / max(0.1, float(rate_hz))
        self._log = log_fn
        self._cond = threading.Condition()
        self._pending: Optional[Tuple[np.ndarray, float]] = None
        self._stopped = False
        self._last_run_t = -1e9
        self._snapshot: List[Tuple[float, float, float]] = []
        self._snapshot_t = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, gray: np.ndarray, t: float) -> None:
        with self._cond:
            if t - self._last_run_t < self.min_interval:
                return
            self._pending = (gray, t)
            self._cond.notify()

    def snapshot(self) -> Tuple[List[Tuple[float, float, float]], float]:
        with self._cond:
            return list(self._snapshot), self._snapshot_t

    def set_mask_params(self, params: MaskParams) -> None:
        self.field.set_mask_params(params)

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()
        self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                gray, t = self._pending
                self._pending = None
                self._last_run_t = t
            try:
                self.field.update(gray, t)
                snap = self.field.predicted(t)
            except Exception as e:                      # pragma: no cover
                self._log(f"DistractorWorker: {e}")
                continue
            with self._cond:
                self._snapshot = snap
                self._snapshot_t = t
