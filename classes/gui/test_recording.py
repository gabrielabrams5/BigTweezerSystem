"""Tests for classes.gui.recording — pure helpers + worker lifecycle.

All headless: QImage works without a QApplication, and the worker's slots
are called directly (no thread), same style as the bridge tests.

Run with:  python3 -m pytest classes/gui/test_recording.py -v
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PyQt5 import QtGui

from classes.config_loader import REPO_ROOT
from classes.gui.recording import (
    VideoRecorderWorker,
    fit_to_size,
    qimage_to_bgr,
    resolve_output_dir,
)
from classes.gui.test_joystick_bridge import FakeConfig


# ---- qimage_to_bgr -----------------------------------------------------


def test_qimage_to_bgr_known_pixels():
    # Odd width 21 exercises the bytesPerLine stride-padding path.
    w, h = 21, 7
    img = QtGui.QImage(w, h, QtGui.QImage.Format_RGB888)
    img.fill(QtGui.QColor(0, 0, 0))
    img.setPixelColor(0, 0, QtGui.QColor(255, 0, 0))     # red
    img.setPixelColor(20, 6, QtGui.QColor(0, 0, 255))    # blue
    img.setPixelColor(5, 3, QtGui.QColor(10, 20, 30))
    arr = qimage_to_bgr(img)
    assert arr.shape == (h, w, 3)
    assert arr.dtype == np.uint8
    assert arr.flags["C_CONTIGUOUS"]
    assert tuple(arr[0, 0]) == (0, 0, 255)       # red in BGR
    assert tuple(arr[6, 20]) == (255, 0, 0)      # blue in BGR
    assert tuple(arr[3, 5]) == (30, 20, 10)


def test_qimage_to_bgr_from_rgb32():
    # Format_RGB32 is what render_video_image produces.
    img = QtGui.QImage(9, 4, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.QColor(1, 2, 3))
    arr = qimage_to_bgr(img)
    assert arr.shape == (4, 9, 3)
    assert tuple(arr[2, 4]) == (3, 2, 1)


# ---- fit_to_size -------------------------------------------------------


def test_fit_to_size_passthrough():
    f = np.full((48, 64, 3), 7, dtype=np.uint8)
    out = fit_to_size(f, (64, 48))
    assert out is f


def test_fit_to_size_downscale():
    f = np.full((96, 128, 3), 200, dtype=np.uint8)
    out = fit_to_size(f, (64, 48))
    assert out.shape == (48, 64, 3)
    assert out[24, 32].max() > 0


def test_fit_to_size_swapped_dims_pads_centered():
    # 48x64 frame onto a 64x48 canvas: scale = min(64/48, 48/64) = 0.75 →
    # 36x48 picture centered with black side borders.
    f = np.full((64, 48, 3), 255, dtype=np.uint8)
    out = fit_to_size(f, (64, 48))
    assert out.shape == (48, 64, 3)
    assert tuple(out[24, 32]) == (255, 255, 255)   # center is picture
    assert tuple(out[24, 2]) == (0, 0, 0)          # left border black
    assert tuple(out[24, 61]) == (0, 0, 0)         # right border black


# ---- worker lifecycle --------------------------------------------------


def test_writer_lifecycle(tmp_path):
    worker = VideoRecorderWorker()
    results = []
    worker.stopped.connect(lambda p, n: results.append((p, n)))

    path = str(tmp_path / "clip.mp4")
    worker.start(path, (64, 48), 20.0)
    for k in range(10):
        if k % 2 == 0:
            frame = np.full((48, 64, 3), k * 20, dtype=np.uint8)
        else:
            frame = np.full((96, 128, 3), k * 20, dtype=np.uint8)  # fitted
        worker.write(frame)
    worker.stop()

    assert results == [(path, 10)]
    assert os.path.exists(path)
    cap = cv2.VideoCapture(path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0, abs=0.5)
    cap.release()


def test_writer_write_without_start_is_noop():
    worker = VideoRecorderWorker()
    worker.write(np.zeros((10, 10, 3), dtype=np.uint8))   # must not raise
    worker.stop()                                          # must not raise


# ---- resolve_output_dir ------------------------------------------------


def test_resolve_output_dir_override(tmp_path):
    target = str(tmp_path / "my_recordings")
    cfg = FakeConfig({"recording.output_dir": target})
    out = resolve_output_dir(cfg, create=True)
    assert out == target
    assert os.path.isdir(target)


def test_resolve_output_dir_default_under_repo_root():
    out = resolve_output_dir(FakeConfig(), create=False)
    import platform
    if platform.system() != "Windows":
        assert out == os.path.join(REPO_ROOT, "Data")
