"""Tests for classes.gui.distractors — global blob tracking.

Run with:  python3 -m pytest classes/gui/test_distractors.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from classes.gui.distractors import Blob, DistractorField, find_blobs
from classes.gui.tracker import MaskParams


H, W = 240, 320
BEAD_R = 8.0
MASK = MaskParams(lower=0, upper=120, crop_length=40, min_blob_area_px=20)


def render_gray(centres, seed=0, noise=2.0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 210.0, dtype=np.float64)
    ys = np.arange(H).reshape(-1, 1)
    xs = np.arange(W).reshape(1, -1)
    for (cx, cy) in centres:
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        img -= 170.0 * np.exp(-0.5 * d2 / (BEAD_R * BEAD_R))
    img += rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_find_blobs_counts_and_locates():
    pts = [(60.0, 60.0), (160.0, 90.0), (240.0, 180.0)]
    blobs = find_blobs(render_gray(pts), MASK)
    assert len(blobs) == 3
    found = sorted(b.pos[0] for b in blobs)
    for got, want in zip(found, sorted(p[0] for p in pts)):
        assert got == pytest.approx(want, abs=2.0)
    assert all(b.area > 20 for b in blobs)


def test_find_blobs_respects_max():
    # Spacing well above 2·BEAD_R so these stay eight distinct blobs
    # rather than merging into one streak.
    pts = [(40.0 + 60.0 * (i % 4), 50.0 + 70.0 * (i // 4)) for i in range(8)]
    assert len(find_blobs(render_gray(pts), MASK)) == 8
    assert len(find_blobs(render_gray(pts), MASK, max_blobs=3)) == 3


def test_field_maintains_stable_track_ids():
    field = DistractorField(MASK)
    for k in range(6):
        field.update(render_gray([(60.0 + 4 * k, 60.0), (200.0, 150.0)],
                                 seed=k), t=k * 0.125)
    assert len(field.tracks) == 2
    # No churn: both tracks were created on the first pass and kept.
    assert {tr.tid for tr in field.tracks} == {1, 2}
    assert all(tr.hits >= 5 for tr in field.tracks)


def test_field_tracks_a_moving_blob():
    field = DistractorField(MASK)
    for k in range(8):
        field.update(render_gray([(50.0 + 10.0 * k, 100.0)], seed=k),
                     t=k * 0.125)
    assert len(field.tracks) == 1
    tr = field.tracks[0]
    assert tr.kf.pos[0] == pytest.approx(50.0 + 10.0 * 7, abs=6.0)
    assert tr.kf.vel[0] > 20.0        # ~80 px/s


def test_predicted_excludes_the_selected_robot():
    field = DistractorField(MASK)
    robot = (60.0, 60.0)
    other = (200.0, 150.0)
    for k in range(4):
        field.update(render_gray([robot, other], seed=k), t=k * 0.125)
    out = field.predicted(0.5, exclude_xy=robot, exclude_radius_px=30.0)
    # The global pass has no idea which blob was selected; the caller
    # filters its own. Suppressing the robot's own peak would push the
    # tracker onto a neighbour, i.e. cause the bug we're fixing.
    assert len(out) == 1
    assert out[0][0] == pytest.approx(other[0], abs=6.0)


def test_stale_tracks_are_pruned():
    field = DistractorField(MASK, max_misses=2)
    for k in range(3):
        field.update(render_gray([(60.0, 60.0), (200.0, 150.0)], seed=k),
                     t=k * 0.125)
    assert len(field.tracks) == 2
    for k in range(3, 9):                       # one blob disappears
        field.update(render_gray([(60.0, 60.0)], seed=k), t=k * 0.125)
    assert len(field.tracks) == 1


def test_area_feature_breaks_a_tie():
    """Two candidates in gate; the one matching the track's running area
    should win. Sizes differ enough to be separable, positions do not."""
    field = DistractorField(MASK, gate_px=60.0, feature_weight=1.0)
    field.update(render_gray([(100.0, 100.0)], seed=1), t=0.0)
    tr = field.tracks[0]
    tr.area_mean = 400.0
    near_wrong_size = Blob(pos=(104.0, 100.0), area=1600.0, blur=0.0)
    far_right_size = Blob(pos=(118.0, 100.0), area=400.0, blur=0.0)
    c_near = field._cost(tr, near_wrong_size, (100.0, 100.0))
    c_far = field._cost(tr, far_right_size, (100.0, 100.0))
    assert c_far < c_near
