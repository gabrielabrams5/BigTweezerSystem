"""Per-object tracker: mask-based initial detection + template-match per-frame.

The legacy stack does grayscale-threshold mask + connected components on the
whole frame every tick. That's simple but drifts under crowded / low-contrast
conditions. Here we use the mask *once at click time* to find what the
operator selected, extract a small grayscale template from that blob, and
then per-frame run ``cv2.matchTemplate`` (normalized cross-correlation) in a
small search window around the last known position. The mask stays as a
fallback for when correlation drops below a confidence gate.

Template matching is guaranteed in base opencv-python, runs in microseconds
on a ~120×120 search × ~40×40 template, and gives sub-pixel precision via a
3-point parabolic peak refinement. It's the "lightweight vision model" you
want for a microrobot control loop.

**Identical neighbours.** Plain ``minMaxLoc`` is a maximum-likelihood pick:
with two near-identical beads in the window it produces two nearly-equal
peaks and returns whichever is marginally higher, so sensor noise decides
which bead you are tracking. Two mechanisms defend against that:

* A caller-supplied :class:`MotionPrior` (from a Kalman filter) weights the
  correlation surface before the argmax, making it a MAP estimate — the
  peak consistent with where the bead was heading wins. Known distractor
  positions are suppressed on the same surface.
* :func:`peak_ambiguity` measures the runner-up/winner ratio on the
  *unweighted* surface. This is the signal ``min_match_confidence``
  structurally cannot provide: in a contested frame both peaks score high,
  so an absolute-confidence gate sees nothing wrong. While contested the
  tracker **freezes its template** — neither the periodic refresh nor the
  mask fallback re-cuts — because a refresh taken while the match has
  half-slid onto a neighbour makes the drift permanent.

All time-based behaviour takes an explicit capture timestamp rather than
counting frames: the caller's worker drops frames under load, so a frame
counter fires at an unpredictable wall-clock rate.

Pure functions + a small stateful ``ObjectTracker`` class. Zero Qt / hardware
imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------ params

@dataclass
class MaskParams:
    lower: int = 0
    upper: int = 128
    blur: int = 0
    dilation: int = 0
    crop_length: int = 40
    invert: bool = False
    min_blob_area_px: int = 20


@dataclass
class MotionPrior:
    """Where the object is *expected* to be on this frame.

    Supplied by the caller's Kalman filter. ``pred_xy`` centres both the
    search window and the Gaussian that weights the correlation surface;
    ``sigma_px`` is that Gaussian's width, floored by the tracker so a
    confident filter can never over-commit and lock out a real
    manoeuvre. ``distractors`` are predicted positions of *other* objects
    whose correlation peaks get suppressed before the argmax.
    """
    pred_xy: Tuple[float, float]
    sigma_px: float
    distractors: Tuple[Tuple[float, float, float], ...] = ()   # (x, y, sigma)


@dataclass
class Detection:
    pos: Tuple[float, float]           # (x, y) in original-frame px, sub-pixel float
    area_px: float
    blur: float                        # cv2.Laplacian variance
    cropped_bgr: np.ndarray
    cropped_mask: np.ndarray
    confidence: float                  # 0..1 for template; blob score for mask
    source: str = "template"           # "template" | "mask" | "mask_init"
    # Ratio of the runner-up correlation peak to the winner, measured on
    # the *unweighted* surface. ~0 = unambiguous, →1 = two equally good
    # answers (a neighbouring bead). Absolute confidence cannot see this:
    # in a contested frame both peaks are high.
    ambiguity: float = 0.0
    # Position of that runner-up peak, for post-run audit of near-swaps.
    second_pos: Optional[Tuple[float, float]] = None
    contested: bool = False            # ambiguity above the enter threshold
    t_capture: float = 0.0             # frame capture time (monotonic s)


# ------------------------------------------------------------------ helpers

def _clip_slice(cx: int, cy: int, half: int,
                shape_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) clipped to the frame."""
    h, w = shape_hw
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(w, int(round(cx + half)))
    y1 = min(h, int(round(cy + half)))
    return x0, y0, x1, y1


def _apply_mask(gray: np.ndarray, params: MaskParams) -> np.ndarray:
    """Grayscale → optional blur → inRange → optional invert → dilate."""
    if params.blur > 0:
        k = max(1, int(params.blur))
        gray = cv2.blur(gray, (k, k))
    mask = cv2.inRange(gray, params.lower, params.upper)
    if params.invert:
        mask = cv2.bitwise_not(mask)
    if params.dilation > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=int(params.dilation))
    return mask


def _find_largest_blob(mask: np.ndarray,
                       min_area_px: int) -> Optional[Tuple[Tuple[float, float], float]]:
    """Return ((cx, cy), area_px) for the largest connected component with
    area ≥ min_area_px, or None if nothing passes."""
    labelled, n = ndimage.label(mask > 0)
    if n == 0:
        return None
    sizes = ndimage.sum(mask > 0, labelled, index=np.arange(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    area = float(sizes[largest - 1])
    if area < min_area_px:
        return None
    cy, cx = ndimage.center_of_mass(mask > 0, labelled, largest)
    return (float(cx), float(cy)), area


def _find_nearest_blob(mask: np.ndarray,
                       target_xy: Tuple[float, float],
                       min_area_px: int) -> Optional[Tuple[Tuple[float, float], float]]:
    """Return the blob whose centroid is closest to ``target_xy``."""
    labelled, n = ndimage.label(mask > 0)
    if n == 0:
        return None
    sizes = ndimage.sum(mask > 0, labelled, index=np.arange(1, n + 1))
    centres = ndimage.center_of_mass(mask > 0, labelled, np.arange(1, n + 1))
    best = None
    best_d2 = float("inf")
    tx, ty = target_xy
    for i, (cy, cx) in enumerate(centres):
        if sizes[i] < min_area_px:
            continue
        d2 = (cx - tx) ** 2 + (cy - ty) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = ((float(cx), float(cy)), float(sizes[i]))
    return best


def _parabolic_subpixel(response: np.ndarray,
                        peak_yx: Tuple[int, int]) -> Tuple[float, float]:
    """3-point parabolic peak fit around (py, px). Returns (dx, dy) offset."""
    py, px = peak_yx
    h, w = response.shape[:2]
    if px <= 0 or px >= w - 1 or py <= 0 or py >= h - 1:
        return 0.0, 0.0
    r = response
    dx_num = float(r[py, px - 1] - r[py, px + 1])
    dx_den = float(r[py, px - 1] - 2 * r[py, px] + r[py, px + 1])
    dy_num = float(r[py - 1, px] - r[py + 1, px])
    dy_den = float(r[py - 1, px] - 2 * r[py, px] + r[py + 1, px])
    dx = 0.5 * dx_num / dx_den if abs(dx_den) > 1e-9 else 0.0
    dy = 0.5 * dy_num / dy_den if abs(dy_den) > 1e-9 else 0.0
    return dx, dy


# ---- correlation-surface geometry ------------------------------------
#
# ``cv2.matchTemplate`` returns a surface of shape (wh-th+1, ww-tw+1) whose
# entry (py, px) is the score for the template's *top-left* sitting at
# (px, py) within the search window. Object centres therefore sit half a
# template further on. Everything below converts between that surface and
# original-frame pixels so priors expressed in frame coordinates land in
# the right place.


def _response_to_global(rx: float, ry: float, x0: int, y0: int,
                        tw: int, th: int) -> Tuple[float, float]:
    return x0 + rx + tw / 2.0, y0 + ry + th / 2.0


def _global_to_response(gx: float, gy: float, x0: int, y0: int,
                        tw: int, th: int) -> Tuple[float, float]:
    return gx - x0 - tw / 2.0, gy - y0 - th / 2.0


def gaussian_surface(shape: Tuple[int, int], cx: float, cy: float,
                     sigma: float) -> np.ndarray:
    """Unit-height Gaussian over a response-map-shaped grid."""
    h, w = shape
    s = max(1e-3, float(sigma))
    ys = np.arange(h, dtype=np.float32).reshape(-1, 1)
    xs = np.arange(w, dtype=np.float32).reshape(1, -1)
    d2 = (xs - float(cx)) ** 2 + (ys - float(cy)) ** 2
    return np.exp(-0.5 * d2 / (s * s)).astype(np.float32)


def peak_ambiguity(response: np.ndarray,
                   peak_xy: Tuple[int, int],
                   suppress_radius_px: float
                   ) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Runner-up / winner ratio on the **unweighted** correlation surface.

    Suppresses a disc around the winner and re-runs ``minMaxLoc``. A ratio
    near 1 means two equally plausible answers — the signature of an
    identical neighbouring bead inside the search window, which absolute
    confidence is structurally blind to because *both* peaks score high.

    Returns ``(ratio, second_peak_xy)``; ratio 0 when there's no credible
    runner-up (including the case where the suppression disc swallows the
    whole surface, which fails safe rather than inventing ambiguity).
    """
    px, py = int(peak_xy[0]), int(peak_xy[1])
    if not (0 <= py < response.shape[0] and 0 <= px < response.shape[1]):
        return 0.0, None
    peak1 = float(response[py, px])
    if peak1 <= 1e-6:
        return 0.0, None
    r = response.copy()
    cv2.circle(r, (px, py), int(max(1, round(suppress_radius_px))),
               -1.0, -1)
    _, peak2, _, loc2 = cv2.minMaxLoc(r)
    if peak2 <= 0.0:
        return 0.0, None
    return float(peak2 / peak1), (int(loc2[0]), int(loc2[1]))


# ------------------------------------------------------------------ tracker

class ObjectTracker:
    """One instance per tracked object (robot or cell).

    ``mask_params`` may be updated live (operator dragging spinboxes) — the
    same instance keeps its template but next frame's fallback path uses the
    new mask settings.
    """

    def __init__(self,
                 mask_params: MaskParams,
                 min_match_confidence: float = 0.5,
                 template_refresh_s: float = 1.5,
                 ambiguity_enter: float = 0.85,
                 ambiguity_exit: float = 0.65,
                 search_half_px: int = 0,
                 prior_sigma_floor_px: float = 3.0,
                 distractor_sigma_floor_px: float = 6.0,
                 merge_area_ratio: float = 1.6,
                 contested_min_dwell_s: float = 0.5,
                 contested_max_s: float = 1.5):
        self.mask_params = mask_params
        self.min_match_confidence = float(min_match_confidence)
        # Wall-clock, not frames: the async worker drops frames under load,
        # so a frame counter refreshes at an unpredictable real-world rate.
        self.template_refresh_s = float(template_refresh_s)
        self.ambiguity_enter = float(ambiguity_enter)
        self.ambiguity_exit = float(ambiguity_exit)
        self.search_half_px = int(search_half_px)
        self.prior_sigma_floor_px = float(prior_sigma_floor_px)
        self.distractor_sigma_floor_px = float(distractor_sigma_floor_px)
        self.merge_area_ratio = float(merge_area_ratio)
        self.contested_min_dwell_s = float(contested_min_dwell_s)
        # Upper bound on how long the latch may hold. Coasting is a bridge,
        # never a resting state.
        self.contested_max_s = float(contested_max_s)
        self.template_gray: Optional[np.ndarray] = None
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_area: float = 0.0
        # Running clean-frame area, used to spot a merge. Two beads that
        # overlap produce ONE blob and ONE correlation peak, so the
        # peak-ratio metric *drops* at the closest approach — the moment
        # of maximum danger reads as safe. Area inflation catches it.
        self.area_ref: float = 0.0
        self._last_template_t: float = 0.0
        # Latched by hysteresis: enter above ambiguity_enter, leave below
        # ambiguity_exit. While contested the template is frozen — a
        # refresh taken mid-contest is how a half-slid tracker makes the
        # drift permanent.
        self.contested: bool = False
        self.last_ambiguity: float = 0.0
        self._contested_since: float = 0.0

    # ---- init ---------------------------------------------------

    def initialize(self,
                   frame_bgr: np.ndarray,
                   click_xy: Tuple[int, int],
                   t: Optional[float] = None) -> Optional[Detection]:
        """Called on operator left-click. Runs mask around the click; finds
        the blob nearest the click; extracts the template."""
        t = time.monotonic() if t is None else float(t)
        cx, cy = click_xy
        half = max(self.mask_params.crop_length, 20)
        x0, y0, x1, y1 = _clip_slice(cx, cy, half, frame_bgr.shape[:2])
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = _apply_mask(gray, self.mask_params)

        found = _find_nearest_blob(mask, (cx - x0, cy - y0),
                                   self.mask_params.min_blob_area_px)
        source = "mask_init"
        if found is None:
            # No blob passed. Just seed at the click coordinate itself and
            # try to grab whatever's under the cursor as a template — the
            # operator can iterate mask params after.
            #
            # Reported distinctly, because the consequences are severe and
            # otherwise invisible: the template is then whatever pixels sat
            # under the cursor, quite possibly background. A background
            # template correlates weakly and almost equally everywhere, so
            # the box wanders or sits still. If a selection logs
            # "click_seed", fix the mask before trusting anything downstream.
            found = ((float(cx - x0), float(cy - y0)),
                     float(self.mask_params.crop_length ** 2))
            source = "click_seed"

        (lx, ly), area = found
        gx, gy = float(lx + x0), float(ly + y0)
        self.last_pos = (gx, gy)
        self.last_area = area

        # Extract template around the found blob.
        self._cut_template(frame_bgr, (gx, gy), t)
        self.contested = False
        self.last_ambiguity = 0.0
        self._contested_since = 0.0
        self.area_ref = float(area)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return Detection(pos=(gx, gy), area_px=area, blur=blur,
                         cropped_bgr=crop, cropped_mask=mask,
                         confidence=1.0, source=source, t_capture=t)

    def _cut_template(self, frame_bgr: np.ndarray,
                      centre_xy: Tuple[float, float],
                      t: float) -> None:
        cx, cy = centre_xy
        half = max(8, int(self.mask_params.crop_length * 0.7))
        x0, y0, x1, y1 = _clip_slice(int(cx), int(cy), half,
                                     frame_bgr.shape[:2])
        if x1 - x0 < 8 or y1 - y0 < 8:
            self.template_gray = None
            return
        patch = frame_bgr[y0:y1, x0:x1]
        self.template_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self._last_template_t = float(t)

    def set_position(self, xy: Tuple[float, float]) -> None:
        """Override the tracker's notion of where the object is.

        Used when the caller coasts on its filter through a contested
        frame: without this the search window would stay centred on the
        contested peak and walk onto the neighbour over the next few
        frames even though the reported position was correct.
        """
        self.last_pos = (float(xy[0]), float(xy[1]))

    def _bead_radius_px(self) -> float:
        """Radius implied by the last blob area, floored to something sane."""
        if self.last_area > 1.0:
            return max(3.0, float(np.sqrt(self.last_area / np.pi)))
        return max(3.0, self.mask_params.crop_length * 0.25)

    # ---- per-frame update ---------------------------------------

    def update(self, frame_bgr: np.ndarray,
               t: Optional[float] = None,
               prior: Optional[MotionPrior] = None) -> Optional[Detection]:
        """Locate the object on this frame.

        With a ``prior`` the correlation surface is multiplied by a
        Gaussian centred on the predicted position and by a suppression
        term around each known distractor, and the argmax is taken on that
        *posterior* surface — a MAP estimate rather than the plain
        maximum-likelihood pick. Ambiguity is always measured on the
        unweighted surface so the prior can't mask a genuine contest.
        """
        t = time.monotonic() if t is None else float(t)
        if self.last_pos is None:
            return None
        if self.template_gray is None:
            return self._fallback_mask(frame_bgr, t=t)

        # Centre the search on the *prediction* when we have one. Centring
        # on the last known position is what lets a fast bead sit near the
        # window edge while a stationary neighbour sits dead centre.
        if prior is not None:
            cx, cy = prior.pred_xy
        else:
            cx, cy = self.last_pos
        search_half = (self.search_half_px if self.search_half_px > 0
                       else int(self.mask_params.crop_length * 1.5))
        x0, y0, x1, y1 = _clip_slice(int(round(cx)), int(round(cy)),
                                     search_half, frame_bgr.shape[:2])
        if x1 - x0 < self.template_gray.shape[1] + 2 or \
           y1 - y0 < self.template_gray.shape[0] + 2:
            return self._fallback_mask(frame_bgr, t=t)

        crop = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        try:
            response = cv2.matchTemplate(
                gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return self._fallback_mask(frame_bgr, t=t)

        th, tw = self.template_gray.shape[:2]

        # --- ambiguity, on the raw surface -------------------------
        # Clamp the suppression disc: the surface is only
        # (window − template + 1) across, so a 2·radius disc can swallow
        # it whole on small crop_lengths and silently report "no
        # runner-up". Keeping an annulus alive is what makes the metric
        # mean anything.
        _, raw_peak, _, raw_loc = cv2.minMaxLoc(response)
        suppress_r = min(2.0 * self._bead_radius_px(),
                         0.35 * min(response.shape[0], response.shape[1]))
        ambiguity, second_loc = peak_ambiguity(response, raw_loc, suppress_r)
        second_pos = None
        if second_loc is not None:
            second_pos = _response_to_global(
                second_loc[0], second_loc[1], x0, y0, tw, th)
        self.last_ambiguity = ambiguity
        # Hysteresis so a surface hovering near the threshold doesn't
        # chatter the state machine on and off every frame.
        was_contested = self.contested
        if was_contested:
            still = ambiguity > self.ambiguity_exit
            # Minimum dwell: an approach reads contested, the merge dips
            # below threshold, then separation reads contested again.
            # Without a dwell floor the tracker would unfreeze and re-cut
            # its template exactly during the overlap.
            if not still and (t - self._contested_since
                              < self.contested_min_dwell_s):
                still = True
            # Hard ceiling. Contested means "coast on the filter and stop
            # believing the image", which is only ever safe as a brief
            # bridge — held indefinitely the box stops following the bead
            # and just drifts on stale velocity. Whatever the surface
            # says, give up the latch and go back to trusting the
            # detection after this long.
            if t - self._contested_since > self.contested_max_s:
                still = False
            self.contested = still
        else:
            # Entering is decided by the peak ratio ALONE. Blob-area
            # inflation can only HOLD the latch (below) — as an entry
            # trigger it fires on ordinary mask flicker, and because
            # ``area_ref`` freezes while contested, that would latch
            # permanently and freeze the box.
            self.contested = ambiguity >= self.ambiguity_enter
        if self.contested and not was_contested:
            self._contested_since = t

        # --- posterior surface -------------------------------------
        posterior = response
        if prior is not None:
            # Clamp negatives first. TM_CCOEFF_NORMED runs [-1, 1], and
            # multiplying a NEGATIVE score by a smaller prior weight makes
            # it LARGER — the weighting would invert exactly where the
            # template anti-correlates, handing the argmax to whatever sits
            # furthest from the prediction. Only positive correlation is
            # evidence of the target anyway.
            likelihood = np.maximum(response, 0.0)
            sigma = max(self.prior_sigma_floor_px, float(prior.sigma_px))
            prx, pry = _global_to_response(
                prior.pred_xy[0], prior.pred_xy[1], x0, y0, tw, th)
            posterior = likelihood * gaussian_surface(
                response.shape, prx, pry, sigma)
            for dx_g, dy_g, dsig in prior.distractors:
                drx, dry = _global_to_response(dx_g, dy_g, x0, y0, tw, th)
                if not (-tw <= drx <= response.shape[1] + tw and
                        -th <= dry <= response.shape[0] + th):
                    continue        # nowhere near this window
                ds = max(self.distractor_sigma_floor_px, float(dsig))
                posterior = posterior * (
                    1.0 - gaussian_surface(response.shape, drx, dry, ds))

        _, post_peak, _, max_loc = cv2.minMaxLoc(posterior)
        if post_peak <= 0.0:
            # The prior and distractor suppression between them wiped the
            # surface out — every candidate got zeroed. Picking the argmax
            # of an all-zero map returns an arbitrary corner, which shows
            # up as the box teleporting. Fall back to the raw peak and let
            # the confidence gate below judge it on its own merits.
            posterior = response
            max_loc = raw_loc
        px, py = max_loc
        # Report confidence from the raw surface — the prior is a search
        # aid, not evidence, and folding it into the number the operator
        # reads (and the loss gate tests) would be self-congratulatory.
        max_val = float(response[py, px])
        if max_val < self.min_match_confidence:
            return self._fallback_mask(frame_bgr, gray=gray, crop_bgr=crop,
                                       x0=x0, y0=y0, t=t,
                                       ambiguity=ambiguity,
                                       second_pos=second_pos)

        dx, dy = _parabolic_subpixel(posterior, (py, px))
        # Match position in original frame coords = template top-left + subpix
        # + template half-size to give centre.
        centre_local_x = px + dx + tw / 2.0
        centre_local_y = py + dy + th / 2.0
        gx = x0 + centre_local_x
        gy = y0 + centre_local_y

        # Compute mask + area + blur at the new position for the panel.
        mask = _apply_mask(gray, self.mask_params)
        # Area = largest blob near the centre; if none, fall back to prior area.
        blob = _find_nearest_blob(mask, (centre_local_x, centre_local_y),
                                  self.mask_params.min_blob_area_px)
        area = blob[1] if blob is not None else self.last_area
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Merge check — see the note on ``area_ref``. Only extends an
        # existing contest (the ambiguity dip at closest approach); it
        # never starts one, and it cannot defeat the hard ceiling above.
        merged = (self.area_ref > 1.0
                  and area > self.merge_area_ratio * self.area_ref)
        if (merged and was_contested and not self.contested
                and t - self._contested_since <= self.contested_max_s):
            self.contested = True

        self.last_pos = (gx, gy)
        self.last_area = area
        if not self.contested:
            # Only clean frames update the reference, otherwise a slow
            # merge would drag the baseline up behind it and the detector
            # would go blind.
            self.area_ref = (area if self.area_ref <= 1.0
                             else self.area_ref + 0.1 * (area - self.area_ref))
        # Freeze the template while contested. A refresh taken while the
        # match has half-slid onto a neighbour bakes the drift in
        # permanently — the tracker can never recover because its own
        # reference now *is* the wrong bead.
        if (not self.contested
                and t - self._last_template_t >= self.template_refresh_s):
            self._cut_template(frame_bgr, (gx, gy), t)

        return Detection(pos=(gx, gy), area_px=area, blur=blur,
                         cropped_bgr=crop, cropped_mask=mask,
                         confidence=max_val, source="template",
                         ambiguity=ambiguity, second_pos=second_pos,
                         contested=self.contested, t_capture=t)

    # ---- fallback ------------------------------------------------

    def _fallback_mask(self,
                       frame_bgr: np.ndarray,
                       gray: Optional[np.ndarray] = None,
                       crop_bgr: Optional[np.ndarray] = None,
                       x0: int = 0, y0: int = 0,
                       t: Optional[float] = None,
                       ambiguity: float = 0.0,
                       second_pos: Optional[Tuple[float, float]] = None
                       ) -> Optional[Detection]:
        """Template confidence too low — relocalize via mask.

        Note what this path does *not* do any more: re-cut the template
        while contested. Low confidence during a close approach usually
        means partial occlusion or two beads merging, which is precisely
        the moment to coast rather than re-seed — re-cutting there is how
        a transient overlap becomes a permanent identity swap.
        """
        t = time.monotonic() if t is None else float(t)
        if self.last_pos is None:
            return None
        if crop_bgr is None or gray is None:
            cx, cy = self.last_pos
            search_half = (self.search_half_px if self.search_half_px > 0
                           else int(self.mask_params.crop_length * 1.5))
            x0, y0, x1, y1 = _clip_slice(int(cx), int(cy), search_half,
                                         frame_bgr.shape[:2])
            crop_bgr = frame_bgr[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        mask = _apply_mask(gray, self.mask_params)
        cx, cy = self.last_pos
        blob = _find_nearest_blob(mask, (cx - x0, cy - y0),
                                  self.mask_params.min_blob_area_px)
        if blob is None:
            return None
        (lx, ly), area = blob
        gx, gy = float(lx + x0), float(ly + y0)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.last_pos = (gx, gy)
        self.last_area = area
        if not self.contested:
            # Safe to re-seed: one credible candidate, the old template
            # had simply drifted out of usefulness.
            self._cut_template(frame_bgr, (gx, gy), t)
        return Detection(pos=(gx, gy), area_px=area, blur=blur,
                         cropped_bgr=crop_bgr, cropped_mask=mask,
                         confidence=0.3, source="mask",
                         ambiguity=ambiguity, second_pos=second_pos,
                         contested=self.contested, t_capture=t)

    # ---- upkeep ---------------------------------------------------

    def set_mask_params(self, params: MaskParams) -> None:
        self.mask_params = params
