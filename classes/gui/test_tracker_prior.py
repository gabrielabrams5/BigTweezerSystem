"""Tests for the motion prior + ambiguity detection in classes.gui.tracker.

The headline test is ``test_prior_prevents_identity_swap``: a synthetic
scene of two identical beads, one moving past a stationary one. Without a
prior the tracker ends up glued to the wrong bead — that is the exact
field failure being fixed — and with a prior it stays on the right one.
A regression here means the swap is back.

Run with:  python3 -m pytest classes/gui/test_tracker_prior.py -v
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from classes.gui.motion_filter import ConstantVelocityKF
from classes.gui.tracker import (
    MaskParams,
    MotionPrior,
    ObjectTracker,
    _global_to_response,
    _response_to_global,
    gaussian_surface,
    peak_ambiguity,
)


H, W = 300, 420
BEAD_R = 9.0
DT = 0.05
MASK = MaskParams(lower=0, upper=120, crop_length=40, min_blob_area_px=20)


def render(centres, seed=0, noise=3.0):
    """Dark Gaussian beads on a bright field — the microscope's polarity."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 210.0, dtype=np.float64)
    ys = np.arange(H).reshape(-1, 1)
    xs = np.arange(W).reshape(1, -1)
    for (cx, cy) in centres:
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        img -= 170.0 * np.exp(-0.5 * d2 / (BEAD_R * BEAD_R))
    img += rng.normal(0.0, noise, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)


# ---- coordinate helpers ------------------------------------------------


def test_response_global_round_trip():
    for gx, gy in ((100.0, 50.0), (0.0, 0.0), (37.5, 12.25)):
        rx, ry = _global_to_response(gx, gy, 10, 20, 56, 56)
        assert _response_to_global(rx, ry, 10, 20, 56, 56) == \
            pytest.approx((gx, gy))


def test_gaussian_surface_peaks_at_centre():
    g = gaussian_surface((41, 41), 20.0, 20.0, 5.0)
    assert g.shape == (41, 41)
    assert g[20, 20] == pytest.approx(1.0)
    assert g[0, 0] < 0.01
    # Monotone falloff along a row.
    row = g[20, 20:]
    assert np.all(np.diff(row) <= 1e-9)


# ---- ambiguity ---------------------------------------------------------


def test_ambiguity_zero_on_single_peak():
    resp = gaussian_surface((61, 61), 30.0, 30.0, 4.0)
    ratio, second = peak_ambiguity(resp, (30, 30), 10.0)
    assert ratio < 0.2


def test_ambiguity_high_on_twin_peaks():
    a = gaussian_surface((61, 61), 20.0, 30.0, 4.0)
    b = gaussian_surface((61, 61), 42.0, 30.0, 4.0)
    resp = np.maximum(a, b * 0.98)
    ratio, second = peak_ambiguity(resp, (20, 30), 10.0)
    assert ratio > 0.85
    assert second is not None
    assert abs(second[0] - 42) <= 2


def test_ambiguity_fails_safe_when_disc_covers_surface():
    """A suppression disc larger than the map must report "no runner-up"
    rather than inventing a value from whatever survives at the corners."""
    resp = gaussian_surface((21, 21), 10.0, 10.0, 3.0)
    ratio, second = peak_ambiguity(resp, (10, 10), 500.0)
    assert ratio == 0.0
    assert second is None


def test_ambiguity_rises_as_neighbour_closes_in():
    """End-to-end through the tracker: far neighbour is uncontested, a
    neighbour ~30 px away latches the contested state."""
    robot = (150.0, 150.0)
    seen = {}
    for gap in (140, 45, 30):
        tr = ObjectTracker(MASK, template_refresh_s=1e9)
        tr.initialize(render([robot, (robot[0] + 300, robot[1])]),
                      (int(robot[0]), int(robot[1])), t=0.0)
        det = tr.update(render([robot, (robot[0] + gap, robot[1])]), t=DT)
        seen[gap] = det
    assert not seen[140].contested
    assert seen[140].ambiguity < 0.3
    assert seen[30].contested
    assert seen[30].ambiguity > 0.85
    # And it's monotone-ish on the way in.
    assert seen[45].ambiguity > seen[140].ambiguity


# ---- the swap ----------------------------------------------------------


def _run_pass(use_prior, n=44, seed=7):
    """Bead crosses a stationary identical twin. Returns final tracker pos."""
    def truth(k):
        return (90.0 + 5.0 * k, 150.0)
    distractor = (200.0, 150.0)

    tr = ObjectTracker(MASK, template_refresh_s=1.5)
    det = tr.initialize(render([truth(0), distractor], seed=seed),
                        (int(truth(0)[0]), int(truth(0)[1])), t=0.0)
    kf = ConstantVelocityKF(det.pos, accel_noise_px_s2=120.0,
                            meas_noise_px=1.5)
    max_amb = 0.0
    for k in range(1, n):
        t = k * DT
        frame = render([truth(k), distractor], seed=seed + k)
        prior = None
        if use_prior:
            pred, sigma = kf.project(DT)
            prior = MotionPrior(pred_xy=pred, sigma_px=sigma)
        d = tr.update(frame, t=t, prior=prior)
        if d is None:
            continue
        max_amb = max(max_amb, d.ambiguity)
        if use_prior:
            kf.step(DT, None, d.pos)
    return tr.last_pos, truth(n - 1), distractor, max_amb


def test_swap_happens_without_a_prior():
    """Pins the failure mode. If this ever passes-by-tracking-correctly the
    scenario has stopped exercising the bug and needs to get harder."""
    pos, robot, distractor, _amb = _run_pass(use_prior=False)
    d_robot = np.hypot(pos[0] - robot[0], pos[1] - robot[1])
    d_dist = np.hypot(pos[0] - distractor[0], pos[1] - distractor[1])
    assert d_dist < d_robot
    assert d_dist < 10.0


def test_prior_prevents_identity_swap():
    pos, robot, distractor, max_amb = _run_pass(use_prior=True)
    d_robot = np.hypot(pos[0] - robot[0], pos[1] - robot[1])
    d_dist = np.hypot(pos[0] - distractor[0], pos[1] - distractor[1])
    assert d_robot < 10.0
    assert d_robot < d_dist
    # The pass-by must have been *noticed*, not silently survived.
    assert max_amb > 0.85


# ---- template freezing -------------------------------------------------


def test_template_frozen_while_contested():
    tr = ObjectTracker(MASK, template_refresh_s=0.0)   # refresh every frame
    robot = (150.0, 150.0)
    tr.initialize(render([robot, (robot[0] + 300, robot[1])]),
                  (int(robot[0]), int(robot[1])), t=0.0)
    stamp_clean = tr._last_template_t
    tr.update(render([robot, (robot[0] + 300, robot[1])]), t=1.0)
    assert tr._last_template_t > stamp_clean          # refreshes when clean

    stamp = tr._last_template_t
    det = tr.update(render([robot, (robot[0] + 30, robot[1])]), t=2.0)
    assert det.contested
    # Frozen: a refresh here would bake a half-slid match in permanently.
    assert tr._last_template_t == stamp


def test_merge_keeps_contested_latched():
    """Two beads that overlap make ONE blob and ONE peak, so the peak-ratio
    metric collapses exactly at closest approach. Area inflation plus the
    dwell floor must hold the latch through that dip."""
    tr = ObjectTracker(MASK, template_refresh_s=1e9,
                       contested_min_dwell_s=0.5)
    robot = (150.0, 150.0)
    tr.initialize(render([robot, (robot[0] + 300, robot[1])]),
                  (int(robot[0]), int(robot[1])), t=0.0)
    tr.update(render([robot, (robot[0] + 200, robot[1])]), t=0.1)
    assert not tr.contested
    assert tr.update(render([robot, (robot[0] + 30, robot[1])]),
                     t=0.2).contested
    merged = tr.update(render([robot, (robot[0] + 14, robot[1])]), t=0.3)
    assert merged.contested


def test_area_flicker_alone_never_latches_contested():
    """Blob area on real data jitters well past 1.6× frame to frame. Area
    inflation must not be able to *enter* the contested state: it would
    latch, `area_ref` would freeze (it only updates when clean), and the
    latch would become permanent — the box stops following and drifts."""
    tr = ObjectTracker(MASK, template_refresh_s=1e9)
    robot = (150.0, 150.0)
    tr.initialize(render([robot]), (int(robot[0]), int(robot[1])), t=0.0)
    tr.area_ref = 10.0                     # pretend the seed area was tiny
    det = tr.update(render([robot]), t=0.5)
    assert det.area_px > 1.6 * 10.0        # inflation is present…
    assert not det.contested               # …but must not latch on its own


def test_contested_latch_always_releases():
    """A permanently ambiguous scene must not pin the tracker in coast
    mode forever — the ceiling has to win over the merge/dwell holds."""
    tr = ObjectTracker(MASK, template_refresh_s=1e9, contested_max_s=0.4)
    robot = (150.0, 150.0)
    tr.initialize(render([robot, (robot[0] + 300, robot[1])]),
                  (int(robot[0]), int(robot[1])), t=0.0)
    twins = [robot, (robot[0] + 30, robot[1])]
    assert tr.update(render(twins), t=0.1).contested
    # Same contested scene, held well past the ceiling.
    last = tr.update(render(twins), t=1.5)
    assert not last.contested
    assert not tr.contested


def test_wiped_surface_falls_back_to_the_raw_peak():
    """If the prior and distractor suppression zero the whole surface, the
    argmax of an all-zero map is an arbitrary corner — the box teleports.
    Fall back to the raw peak instead."""
    tr = ObjectTracker(MASK, template_refresh_s=1e9)
    robot = (150.0, 150.0)
    tr.initialize(render([robot]), (int(robot[0]), int(robot[1])), t=0.0)
    # A prior pointing at the bead but with the bead ALSO listed as a
    # distractor: suppression cancels the only real candidate.
    prior = MotionPrior(pred_xy=robot, sigma_px=4.0,
                        distractors=((robot[0], robot[1], 40.0),))
    det = tr.update(render([robot]), t=0.1, prior=prior)
    assert det is not None
    assert np.hypot(det.pos[0] - robot[0], det.pos[1] - robot[1]) < 8.0


def test_set_position_moves_the_search_window():
    tr = ObjectTracker(MASK)
    tr.initialize(render([(150.0, 150.0)]), (150, 150), t=0.0)
    tr.set_position((77.0, 88.0))
    assert tr.last_pos == (77.0, 88.0)
