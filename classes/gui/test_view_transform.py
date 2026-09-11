"""Tests for classes.gui.view_transform — display-rotation coordinate math.

Run with:  python3 -m pytest classes/gui/test_view_transform.py -v
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from classes.gui.view_transform import (
    display_size,
    display_to_frame,
    frame_to_display,
    stick_display_to_raw,
)


ROTATIONS = (0, 90, 180, 270)
_ROT_CV = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def test_display_size():
    assert display_size(640, 480, 0) == (640, 480)
    assert display_size(640, 480, 90) == (480, 640)
    assert display_size(640, 480, 180) == (640, 480)
    assert display_size(640, 480, 270) == (480, 640)


@pytest.mark.parametrize("rot", ROTATIONS)
def test_round_trip(rot):
    w, h = 640, 480
    points = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
              (123, 456), (7, 200)]
    for x, y in points:
        dx, dy = frame_to_display(x, y, w, h, rot)
        assert display_to_frame(dx, dy, w, h, rot) == (x, y)
        # Displayed point must land inside the rotated canvas.
        dw, dh = display_size(w, h, rot)
        assert 0 <= dx < dw and 0 <= dy < dh


@pytest.mark.parametrize("rot", (90, 180, 270))
def test_matches_cv2_raster(rot):
    """Mark one pixel, cv2.rotate the raster, and confirm the marked pixel
    lands exactly where frame_to_display says it should — nails the ±1
    pixel-center convention against cv2 for every rotation."""
    w, h = 64, 48
    for x, y in ((0, 0), (w - 1, h - 1), (10, 33), (50, 5)):
        img = np.zeros((h, w), dtype=np.uint8)
        img[y, x] = 255
        rotated = cv2.rotate(img, _ROT_CV[rot])
        ys, xs = np.nonzero(rotated)
        assert (int(xs[0]), int(ys[0])) == frame_to_display(x, y, w, h, rot)


@pytest.mark.parametrize("rot", ROTATIONS)
def test_stick_matrix_is_rotation(rot):
    R = np.asarray(stick_display_to_raw(rot), dtype=float)
    # Proper rotation: orthogonal with det +1 (no reflection).
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_stick_matrix_90_case():
    # View rotated 90° CW: display-up corresponds to raw-left.
    R = np.asarray(stick_display_to_raw(90), dtype=float)
    assert np.allclose(R @ np.array([0.0, 1.0]), np.array([-1.0, 0.0]))
    # And display-right corresponds to raw-up.
    assert np.allclose(R @ np.array([1.0, 0.0]), np.array([0.0, 1.0]))
