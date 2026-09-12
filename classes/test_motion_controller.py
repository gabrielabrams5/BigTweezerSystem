"""Tests for classes.motion_controller.

Run with:  python3 -m pytest classes/test_motion_controller.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from classes.field_solver import FieldSolver, PARAMAGNETIC
from classes.motion_controller import (
    Mode,
    MotionController,
    Source,
    _rotation_matrix,
    roll_axis_for,
)


_S = np.sqrt(2) / 2
_AZ = np.deg2rad([0.0, 120.0, 240.0])
GEOM_BMAP = np.column_stack([
    *(np.array([_S * np.cos(a), _S * np.sin(a),  _S]) for a in _AZ),
    *(np.array([_S * np.cos(a), _S * np.sin(a), -_S]) for a in _AZ),
])


def make_controller() -> MotionController:
    solver = FieldSolver(GEOM_BMAP, robot_type=PARAMAGNETIC)
    mc = MotionController(solver)
    return mc


# ---- Rodrigues helper ------------------------------------------------


def test_rotation_matrix_identity_on_zero_axis():
    R = _rotation_matrix([0.0, 0.0, 0.0], 1.234)
    assert np.allclose(R, np.eye(3))


def test_rotation_matrix_z_axis_ninety_deg():
    R = _rotation_matrix([0.0, 0.0, 1.0], np.pi / 2)
    # +X should map to +Y.
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]),
                       np.array([0.0, 1.0, 0.0]), atol=1e-9)


def test_rotation_matrix_normalizes_axis():
    R1 = _rotation_matrix([0.0, 0.0, 5.0], np.pi / 2)
    R2 = _rotation_matrix([0.0, 0.0, 1.0], np.pi / 2)
    assert np.allclose(R1, R2, atol=1e-12)


# ---- Mode A rolling formula --------------------------------------------


class TestModeARolling:
    """B_des(t) = |B| · R(roll_axis, 2πft) @ direction — legacy semantics."""

    def test_static_direction_when_freq_zero(self):
        mc = make_controller()
        mc.direction = np.array([1.0, 0.0, 0.0])
        mc.roll_axis = np.array([0.0, 0.0, 1.0])
        mc.magnitude = 0.7
        mc.freq_hz = mc.freq_target_hz = 0.0
        mc.set_mode(Mode.A_ROTATING)
        # Phase stays at 0 across ticks (freq = 0), so B_des = 0.7 · [1,0,0].
        for _ in range(5):
            mc.tick(0.01)
        expected_B = 0.7 * np.array([1.0, 0.0, 0.0])
        # Reproduce solver.solve on expected_B to compare currents.
        assert np.allclose(mc.solver.solve(expected_B).i,
                           mc.solver.solve(0.7 * (
                               _rotation_matrix(mc.roll_axis, mc.phase)
                               @ mc.direction)).i, atol=1e-9)

    def test_rotates_direction_around_z_at_freq(self):
        """Step through one full period; commanded B_des should trace a
        circle in the XY plane when direction=+X, roll_axis=+Z."""
        mc = make_controller()
        mc.direction = np.array([1.0, 0.0, 0.0])
        mc.roll_axis = np.array([0.0, 0.0, 1.0])
        mc.magnitude = 1.0
        mc.freq_hz = mc.freq_target_hz = 1.0   # 1 Hz
        mc.set_mode(Mode.A_ROTATING)

        # Small dt; sample at t = 0.25 s → phase = π/2 → +X should rotate to +Y.
        dt = 0.0005
        steps = int(0.25 / dt)
        for _ in range(steps):
            mc.tick(dt)
        # Reconstruct the B_des the tick just computed from the internal state.
        R = _rotation_matrix(mc.roll_axis, mc.phase)
        B_des = mc.magnitude * (R @ mc.direction)
        assert np.allclose(B_des, np.array([0.0, 1.0, 0.0]), atol=1e-3)

    def test_rolling_locomotion_vertical_circle(self):
        """Horizontal roll axis ⊥ travel → B_des sweeps the vertical plane
        containing the travel vector (x–z here): +X → −Z → −X over half a
        period at 1 Hz. This is the joystick rolling-locomotion geometry."""
        mc = make_controller()
        mc.direction = np.array([1.0, 0.0, 0.0])   # travel +X
        mc.roll_axis = np.array([0.0, 1.0, 0.0])   # ẑ × x̂ = +Y
        mc.magnitude = 1.0
        mc.freq_hz = mc.freq_target_hz = 1.0
        mc.set_mode(Mode.A_ROTATING)

        dt = 0.0005
        for _ in range(int(0.25 / dt)):            # quarter period
            mc.tick(dt)
        R = _rotation_matrix(mc.roll_axis, mc.phase)
        assert np.allclose(mc.magnitude * (R @ mc.direction),
                           np.array([0.0, 0.0, -1.0]), atol=1e-3)

        for _ in range(int(0.25 / dt)):            # half period total
            mc.tick(dt)
        R = _rotation_matrix(mc.roll_axis, mc.phase)
        assert np.allclose(mc.magnitude * (R @ mc.direction),
                           np.array([-1.0, 0.0, 0.0]), atol=1e-3)

    def test_zero_roll_axis_collapses_to_static(self):
        """Zero roll axis (joystick triggers-only case) → Rodrigues identity
        → B_des stays |B|·direction despite nonzero frequency."""
        mc = make_controller()
        mc.direction = np.array([0.0, 0.0, 1.0])
        mc.roll_axis = np.zeros(3)
        mc.magnitude = 0.8
        mc.freq_hz = mc.freq_target_hz = 5.0
        mc.set_mode(Mode.A_ROTATING)
        expected = mc.solver.solve(0.8 * np.array([0.0, 0.0, 1.0])).i
        for _ in range(20):
            result = mc.tick(0.01)
            assert np.allclose(result.i, expected, atol=1e-9)

    def test_negative_frequency_reverses_phase(self):
        mc = make_controller()
        mc.direction = np.array([1.0, 0.0, 0.0])
        mc.roll_axis = np.array([0.0, 1.0, 0.0])
        mc.freq_hz = mc.freq_target_hz = -1.0
        mc.set_mode(Mode.A_ROTATING)
        for _ in range(10):
            mc.tick(0.01)
        assert mc.phase < 0.0

    def test_off_mode_emits_zeros(self):
        mc = make_controller()
        mc.set_mode(Mode.OFF)
        result = mc.tick(0.01)
        assert np.allclose(result.i, np.zeros(6))
        assert not result.saturated


# ---- roll_axis_for ----------------------------------------------------


def test_roll_axis_for_perpendicular_unit():
    for tx, ty in ((1.0, 0.0), (0.0, 1.0), (-3.0, 4.0), (0.7, -0.7)):
        a = roll_axis_for(tx, ty)
        assert a[2] == 0.0
        assert np.isclose(np.hypot(a[0], a[1]), 1.0)
        # Perpendicular to travel.
        assert np.isclose(a[0] * tx + a[1] * ty, 0.0)
        # ẑ × t̂ handedness: for +X travel the axis is +Y.
    assert np.allclose(roll_axis_for(1.0, 0.0), [0.0, 1.0, 0.0])


def test_roll_axis_for_zero_and_threshold():
    assert roll_axis_for(0.0, 0.0) == [0.0, 0.0, 0.0]
    assert roll_axis_for(0.01, 0.02, min_norm=0.05) == [0.0, 0.0, 0.0]
    assert roll_axis_for(0.06, 0.0, min_norm=0.05) == [0.0, 1.0, 0.0]


# ---- set_roll_axis normalization --------------------------------------


def test_set_roll_axis_normalizes():
    mc = make_controller()
    mc.set_roll_axis([0.0, 0.0, 3.0], Source.GUI)
    assert np.allclose(mc.roll_axis, np.array([0.0, 0.0, 1.0]))


def test_set_roll_axis_keeps_zero():
    mc = make_controller()
    mc.set_roll_axis([0.0, 0.0, 0.0], Source.GUI)
    # Kept as-is; the tick's Rodrigues helper collapses to identity.
    assert np.allclose(mc.roll_axis, np.zeros(3))
