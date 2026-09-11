"""Tests for PathFollowController's drive-mode command path and its
calibrated image→world mapping.

Run with:  python3 -m pytest classes/gui/test_path_follow_drive.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from classes.field_solver import FieldSolver, PARAMAGNETIC
from classes.gui.frame_calibration import fit_screen_to_world
from classes.gui.path_follow import PathFollowController
from classes.gui.test_frame_calibration_fit import S_RIG, simulate
from classes.gui.test_joystick_bridge import FakeConfig
from classes.motion_controller import Mode, MotionController, Source
from classes.test_motion_controller import GEOM_BMAP


def make_controller(config_values=None):
    motion = MotionController(FieldSolver(GEOM_BMAP,
                                          robot_type=PARAMAGNETIC))
    cfg = FakeConfig(config_values)
    ctrl = PathFollowController(motion, cfg, log_fn=lambda *_: None)
    return ctrl, motion, cfg


def test_rolling_drive_commands_mode_a():
    ctrl, motion, _ = make_controller({"modes.mode_a.freq_default": 3.0})
    assert ctrl.drive_mode == "rolling"
    ctrl._command_direction((1.0, 0.0))
    assert motion.mode == Mode.A_ROTATING
    assert motion.source == Source.ALGORITHM
    assert np.allclose(motion.direction, [1.0, 0.0, 0.0])
    assert np.allclose(motion.roll_axis, [0.0, 1.0, 0.0])
    assert motion.freq_target_hz == pytest.approx(3.0)


def test_static_drive_commands_mode_b():
    ctrl, motion, _ = make_controller({"path_follow.drive_mode": "static"})
    ctrl._command_direction((0.0, 1.0))
    assert motion.mode == Mode.B_STATIC
    assert np.allclose(motion.direction, [0.0, 1.0, 0.0])


def test_drive_mode_setter_validates():
    ctrl, _, _ = make_controller()
    ctrl.set_drive_mode("static")
    assert ctrl.drive_mode == "static"
    ctrl.set_drive_mode("bogus")
    assert ctrl.drive_mode == "static"
    ctrl.set_drive_mode("rolling")
    assert ctrl.drive_mode == "rolling"


def test_shared_freq_and_mag_read_live():
    """Mid-run Joystick-tab changes apply on the next command with no
    controller restart or listener plumbing."""
    ctrl, motion, cfg = make_controller({
        "modes.mode_a.freq_default": 3.0,
        "modes.mode_a.magnitude_default": 1.0,
    })
    ctrl._command_direction((1.0, 0.0))
    assert motion.freq_target_hz == pytest.approx(3.0)
    assert motion.magnitude == pytest.approx(1.0)
    cfg.set("modes.mode_a.freq_default", 7.5)
    cfg.set("modes.mode_a.magnitude_default", 0.4)
    ctrl._command_direction((1.0, 0.0))
    assert motion.freq_target_hz == pytest.approx(7.5)
    assert motion.magnitude == pytest.approx(0.4)


# ---- image → world mapping ---------------------------------------------


def test_image_to_world_defaults_to_flip_only():
    # No saved matrix → identity → behavioral parity with the old
    # image_flip_y default: (dx, dy) → (dx, -dy).
    ctrl, _, _ = make_controller()
    assert ctrl._image_to_world(3.0, 4.0) == (3.0, -4.0)


def test_image_to_world_uses_calibrated_matrix():
    ctrl, _, _ = make_controller({
        "calibration.screen_to_world_2x2": [[0.0, -1.0], [1.0, 0.0]],
    })
    # (dx, dy) = (3, 4) → up-frame (3, -4) → M rotates 90° CCW → (4, 3).
    wx, wy = ctrl._image_to_world(3.0, 4.0)
    assert (wx, wy) == pytest.approx((4.0, 3.0))


def test_frame_matrix_live_reload():
    ctrl, _, cfg = make_controller()
    assert ctrl._image_to_world(1.0, 0.0) == (1.0, 0.0)
    cfg.set("calibration.screen_to_world_2x2", [[-1.0, 0.0], [0.0, -1.0]])
    assert ctrl._image_to_world(1.0, 0.0) == (-1.0, 0.0)


def test_follower_drives_toward_target_on_mirrored_rig():
    """The reported bug: on a mirrored/skewed rig the follower drove the
    bead AWAY from the target because it ignored the calibrated matrix.
    With the fitted matrix, the rig's screen response to the commanded
    direction must point AT the target."""
    M_sw, q = fit_screen_to_world(simulate(S_RIG))
    assert q["ok"]
    ctrl, _, _ = make_controller({
        "calibration.screen_to_world_2x2": M_sw.tolist(),
    })
    for err_img in ((40.0, 0.0), (0.0, -35.0), (25.0, 25.0), (-10.0, 30.0)):
        d = ctrl._push_direction(err_img)
        assert d is not None
        # Rig response in screen-up frame, mapped back to image coords
        # (y down) to compare against the image-space error vector.
        s_up = S_RIG @ np.array(d)
        s_img = np.array([s_up[0], -s_up[1]])
        e = np.array(err_img)
        cross = s_img[0] * e[1] - s_img[1] * e[0]
        assert abs(cross) < 1e-6 * np.linalg.norm(s_img) * np.linalg.norm(e)
        assert float(np.dot(s_img, e)) > 0.0, f"drives away for {err_img}"

    # And pin the old behavior as wrong: the plain (dx, -dy) mapping (no
    # matrix) points the response in a genuinely different direction on
    # this rig for at least one error vector.
    ctrl_old, _, _ = make_controller()   # identity matrix = old default
    bad = 0
    for err_img in ((40.0, 0.0), (0.0, -35.0), (25.0, 25.0), (-10.0, 30.0)):
        d = ctrl_old._push_direction(err_img)
        s_up = S_RIG @ np.array(d)
        s_img = np.array([s_up[0], -s_up[1]])
        e = np.array(err_img)
        cos = float(np.dot(s_img, e)) / (
            np.linalg.norm(s_img) * np.linalg.norm(e))
        if cos < math.cos(math.radians(20)):
            bad += 1
    assert bad > 0
