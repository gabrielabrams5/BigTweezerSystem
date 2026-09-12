"""Tests for classes.gui.joystick_bridge — axis → motion translation.

The bridge is exercised by calling ``_handle_axes`` directly; no pygame or
QApplication is needed (the poll timer never starts).

Run with:  python3 -m pytest classes/gui/test_joystick_bridge.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from classes.field_solver import FieldSolver, PARAMAGNETIC
from classes.gui.joystick_bridge import JoystickBridge
from classes.motion_controller import Mode, MotionController, Source
from classes.test_motion_controller import GEOM_BMAP


class FakeConfig:
    """Dict-backed stand-in for config_loader.Config with the real class's
    prefix-matching on_change semantics."""

    def __init__(self, values=None):
        self._values = dict(values or {})
        self._listeners = []

    def get(self, path, default=None):
        return self._values.get(path, default)

    def set(self, path, value):
        self._values[path] = value
        for prefix, cb in list(self._listeners):
            if path == prefix or path.startswith(prefix + ".") or prefix == "":
                cb(path, value)

    def on_change(self, prefix, cb):
        self._listeners.append((prefix, cb))

    def save(self):
        pass


class FakeSupervisor:
    def __init__(self):
        self.estopped = False

    def estop(self):
        self.estopped = True

    def reset_estop(self):
        self.estopped = False


def make_bridge(config_values=None):
    motion = MotionController(FieldSolver(GEOM_BMAP,
                                          robot_type=PARAMAGNETIC))
    cfg = FakeConfig(config_values)
    bridge = JoystickBridge(motion, FakeSupervisor(), cfg,
                            log_fn=lambda *_: None)
    return bridge, motion, cfg


# ---- roll axis follows the stick --------------------------------------


def test_stick_right_sets_perpendicular_roll_axis():
    bridge, motion, _ = make_bridge()
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(motion.roll_axis, [0.0, 1.0, 0.0], atol=1e-9)
    assert motion.mode == Mode.A_ROTATING
    assert motion.source == Source.JOYSTICK


def test_stick_up_sets_perpendicular_roll_axis():
    # pygame ly = -1 means stick pushed up → by = +1 → roll axis = -x̂.
    bridge, motion, _ = make_bridge()
    bridge._handle_axes(lx=0.0, ly=-1.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(motion.roll_axis, [-1.0, 0.0, 0.0], atol=1e-9)


def test_screen_to_world_matrix_applied_before_perpendicular():
    # 90° CCW rotation: screen +X → world +Y.
    bridge, motion, _ = make_bridge({
        "calibration.screen_to_world_2x2": [[0.0, -1.0], [1.0, 0.0]],
    })
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(motion.roll_axis, [-1.0, 0.0, 0.0], atol=1e-9)


def test_triggers_only_gives_static_vertical_pull():
    bridge, motion, _ = make_bridge()
    bridge._handle_axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=1.0)
    assert np.allclose(motion.direction, [0.0, 0.0, 1.0], atol=1e-9)
    # Zero roll axis → rotation collapses to identity → static pull.
    assert np.allclose(motion.roll_axis, [0.0, 0.0, 0.0])


# ---- signed frequency mapping -----------------------------------------


def test_frequency_mapping():
    bridge, motion, _ = make_bridge({
        "modes.mode_a.freq_default": 1.0,
        "modes.mode_a.freq_max_hz": 20.0,
    })

    def freq_for(rx):
        bridge._handle_axes(lx=1.0, ly=0.0, rx=rx, ry=0.0, lt=0.0, rt=0.0)
        return motion.freq_target_hz

    assert freq_for(0.0) == pytest.approx(1.0)      # centered → base
    assert freq_for(1.0) == pytest.approx(20.0)     # full right → max
    assert freq_for(0.5) == pytest.approx(1.0 + 0.5 * 19.0)
    assert freq_for(-1.0) == pytest.approx(-20.0)   # full left → reversed


# ---- mode engagement ---------------------------------------------------


def test_idle_drops_to_off_only_when_joystick_owns_mode():
    bridge, motion, _ = make_bridge()
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion.mode == Mode.A_ROTATING
    bridge._handle_axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion.mode == Mode.OFF

    # GUI-owned mode survives stick release.
    motion.set_mode(Mode.B_STATIC)
    motion.source = Source.GUI
    bridge._handle_axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion.mode == Mode.B_STATIC


def test_stick_mode_static_engages_mode_b():
    bridge, motion, _ = make_bridge({"joystick.stick_mode": "static"})
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion.mode == Mode.B_STATIC


def test_stick_mode_live_switch_via_config():
    bridge, motion, _cfg = make_bridge()
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion.mode == Mode.A_ROTATING

    # Release, switch the config live (as the Joystick tab does), re-grab.
    bridge._handle_axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    _cfg.set("joystick.stick_mode", "static")
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion.mode == Mode.B_STATIC


def test_existing_mode_respected_on_stick_grab():
    bridge, motion, _ = make_bridge()
    motion.set_mode(Mode.B_STATIC)
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion.mode == Mode.B_STATIC


# ---- display view-rotation composition --------------------------------


def test_view_rotation_90_stick_up_means_displayed_up():
    # View rotated 90° CW: what appears as "up" on screen is raw-left, so
    # stick-up must command world -X (with identity calibration matrix).
    bridge, motion, _ = make_bridge({"camera.view_rotation_deg": 90})
    bridge._handle_axes(lx=0.0, ly=-1.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [-1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(motion.roll_axis, [0.0, -1.0, 0.0], atol=1e-9)
    # Stick-right under 90° CW view → raw-up (world +Y).
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [0.0, 1.0, 0.0], atol=1e-9)


def test_view_rotation_live_change_applies():
    bridge, motion, cfg = make_bridge()
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [1.0, 0.0, 0.0], atol=1e-9)
    cfg.set("camera.view_rotation_deg", 180)   # fires on_change listener
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [-1.0, 0.0, 0.0], atol=1e-9)


def test_view_rotation_composes_before_calibration_matrix():
    # 90° CCW calibration matrix on top of a 90° view rotation: the view
    # rotation must apply first (world = M @ R @ stick).
    bridge, motion, _ = make_bridge({
        "camera.view_rotation_deg": 90,
        "calibration.screen_to_world_2x2": [[0.0, -1.0], [1.0, 0.0]],
    })
    # Stick-up → R gives raw (-1, 0) → M rotates to (0, -1).
    bridge._handle_axes(lx=0.0, ly=-1.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [0.0, -1.0, 0.0], atol=1e-9)


# ---- axis-invert toggles -----------------------------------------------


def test_invert_x_flips_stick_right():
    bridge, motion, _ = make_bridge({"joystick.invert_x": True})
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [-1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(motion.roll_axis, [0.0, -1.0, 0.0], atol=1e-9)


def test_invert_y_flips_stick_up():
    bridge, motion, _ = make_bridge({"joystick.invert_y": True})
    bridge._handle_axes(lx=0.0, ly=-1.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [0.0, -1.0, 0.0], atol=1e-9)


def test_invert_applies_before_view_rotation():
    # Uninverted stick-right under 90° CW view → world +Y (covered above);
    # inverted display intent goes through R afterwards → world -Y.
    bridge, motion, _ = make_bridge({"joystick.invert_x": True,
                                     "camera.view_rotation_deg": 90})
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [0.0, -1.0, 0.0], atol=1e-9)


def test_invert_live_change_via_config():
    bridge, motion, cfg = make_bridge()
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [1.0, 0.0, 0.0], atol=1e-9)
    cfg.set("joystick.invert_x", True)   # fires on_change listener
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert np.allclose(motion.direction, [-1.0, 0.0, 0.0], atol=1e-9)


# ---- calibrated-matrix + shared-magnitude integration ------------------


def test_calibrated_matrix_makes_stick_direction_true():
    """End-to-end: fit a skewed mirrored rig, feed the matrix to the bridge,
    and confirm the rig's screen response to the commanded direction is
    parallel to the stick for several directions."""
    from classes.gui.test_frame_calibration_fit import S_RIG, simulate
    from classes.gui.frame_calibration import fit_screen_to_world

    M_sw, q = fit_screen_to_world(simulate(S_RIG))
    assert q["ok"]
    bridge, motion, _ = make_bridge({
        "calibration.screen_to_world_2x2": M_sw.tolist(),
    })
    for lx, ly in ((1.0, 0.0), (0.0, -1.0), (0.7, -0.7), (-0.5, 0.86)):
        bridge._handle_axes(lx=lx, ly=ly, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
        u = np.array([lx, -ly])
        s = S_RIG @ motion.direction[:2]
        cross = s[0] * u[1] - s[1] * u[0]
        assert abs(cross) < 1e-9 * max(1.0, np.linalg.norm(s))
        assert float(np.dot(s, u)) > 0.0


def test_anisotropic_matrix_is_direction_only():
    """The calibration matrix must set direction only — its column
    magnitudes encode measured rolling distances, and letting them scale
    drive strength starves the weak screen axis (the reported 'rolling
    works up/down but not left/right, current draw very low' bug). This is
    the operator's actual fitted matrix: stick-right column norm ≈ 0.29."""
    M = [[-0.07756620693459323, 0.9917994928730074],
         [0.2776757542711887, 0.12780362254977456]]
    bridge, motion, _ = make_bridge({
        "calibration.screen_to_world_2x2": M,
    })
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    # Full deflection → full magnitude, regardless of the weak column.
    assert motion.magnitude == pytest.approx(1.0)
    # Direction is still the matrix's column, normalized.
    col = np.array([M[0][0], M[1][0]])
    col = col / np.linalg.norm(col)
    assert np.allclose(motion.direction[:2], col, atol=1e-9)
    assert motion.direction[2] == pytest.approx(0.0)


def test_shared_magnitude_scales_stick_drive():
    bridge, motion, _ = make_bridge({"modes.mode_a.magnitude_default": 0.5})
    bridge._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion.magnitude == pytest.approx(0.5)
    # Default 1.0 reproduces the old behavior.
    bridge2, motion2, _ = make_bridge()
    bridge2._handle_axes(lx=1.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0)
    assert motion2.magnitude == pytest.approx(1.0)


def test_mag_buttons_write_shared_config():
    bridge, motion, cfg = make_bridge({"modes.mode_a.magnitude_default": 0.5})
    pressed = [False] * 8
    pressed[bridge.btn_mag_up] = True
    bridge._handle_buttons(pressed)
    assert cfg.get("modes.mode_a.magnitude_default") == pytest.approx(0.6)
    assert bridge.mag_base == pytest.approx(0.6)   # via on_change listener
    assert motion.magnitude == pytest.approx(0.6)
    pressed = [False] * 8
    pressed[bridge.btn_mag_down] = True
    bridge._handle_buttons(pressed)
    assert cfg.get("modes.mode_a.magnitude_default") == pytest.approx(0.5)
