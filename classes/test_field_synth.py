"""Unit tests for classes/field_synth.py.

Run with:  pytest classes/test_field_synth.py -v
"""

import json
import os
import tempfile

import numpy as np
import pytest

from classes import field_synth as fs


# ---------------------------------------------------------------- geometry

def test_coil_axes_shape_and_norm():
    """All six axis vectors are unit length."""
    assert fs.COIL_AXES.shape == (3, 6)
    for i in range(6):
        assert np.linalg.norm(fs.COIL_AXES[:, i]) == pytest.approx(1.0, abs=1e-9)


def test_coil_axes_symmetries():
    """Top and bottom ring differ only in Z sign; azimuths match."""
    top = fs.COIL_AXES[:, :3]
    bot = fs.COIL_AXES[:, 3:]
    # XY components identical
    assert np.allclose(top[:2, :], bot[:2, :])
    # Z components opposite
    assert np.allclose(top[2, :], -bot[2, :])
    # Each top Z is +sqrt(2)/2
    assert np.allclose(top[2, :], np.sqrt(2) / 2)


def test_a_atrans_is_diagonal_3_2_3_2_3():
    """The A A^T = diag(3/2, 3/2, 3) identity that makes the closed-form work."""
    AAt = fs.COIL_AXES @ fs.COIL_AXES.T
    expected = np.diag([1.5, 1.5, 3.0])
    assert np.allclose(AAt, expected, atol=1e-9)


# ---------------------------------------------------------------- uniform

def test_uniform_currents_pure_x():
    I = fs.uniform_currents([1.0, 0.0, 0.0])
    # Closed form: I1 = I4 = 0.472, I2 = I3 = I5 = I6 = -0.236
    expected = np.array([0.4714, -0.2357, -0.2357, 0.4714, -0.2357, -0.2357])
    assert np.allclose(I, expected, atol=1e-3)


def test_uniform_currents_pure_y():
    I = fs.uniform_currents([0.0, 1.0, 0.0])
    # I1 = I4 = 0; C2/C5 positive Y contribution, C3/C6 negative
    assert I[0] == pytest.approx(0.0, abs=1e-9)
    assert I[3] == pytest.approx(0.0, abs=1e-9)
    assert I[1] > 0 and I[4] > 0
    assert I[2] < 0 and I[5] < 0
    # top/bottom symmetry for By: I1 = I4, I2 = I5, I3 = I6
    assert I[1] == pytest.approx(I[4], abs=1e-9)
    assert I[2] == pytest.approx(I[5], abs=1e-9)


def test_uniform_currents_pure_z():
    I = fs.uniform_currents([0.0, 0.0, 1.0])
    # Top ring all +0.2357, bottom ring all -0.2357
    assert np.allclose(I[:3],  0.2357, atol=1e-3)
    assert np.allclose(I[3:], -0.2357, atol=1e-3)


def test_uniform_currents_recover_field():
    """A @ I should return exactly the target field for any B_target."""
    for B in [(1, 0, 0), (0, 1, 0), (0, 0, 1),
              (0.5, -0.5, 0.3), (-1, 1, -1)]:
        I = fs.uniform_currents(B)
        B_actual = fs.COIL_AXES @ I
        assert np.allclose(B_actual, B, atol=1e-9), f"B={B} -> B_actual={B_actual}"


def test_uniform_currents_zero_field_is_zero():
    I = fs.uniform_currents([0, 0, 0])
    assert np.allclose(I, 0.0, atol=1e-12)


def test_uniform_currents_linearity():
    I1 = fs.uniform_currents([1, 0, 0])
    I2 = fs.uniform_currents([0, 1, 0])
    I_sum = fs.uniform_currents([1, 1, 0])
    assert np.allclose(I_sum, I1 + I2, atol=1e-9)


# ---------------------------------------------------------------- gradient

def test_gradient_currents_pure_z():
    I = fs.gradient_currents([0, 0, 1], 1.0)
    # Top ring +sqrt(2)/2, bottom ring -sqrt(2)/2
    assert np.allclose(I[:3],  np.sqrt(2) / 2, atol=1e-9)
    assert np.allclose(I[3:], -np.sqrt(2) / 2, atol=1e-9)


def test_gradient_currents_pure_x():
    I = fs.gradient_currents([1, 0, 0], 1.0)
    # C1 and C4 both have +X in axis, get +sin(45)*cos(0) = sqrt(2)/2
    # C2, C3, C5, C6 have negative X projection = -sqrt(2)/4
    assert I[0] == pytest.approx(np.sqrt(2) / 2, abs=1e-9)
    assert I[3] == pytest.approx(np.sqrt(2) / 2, abs=1e-9)
    for idx in (1, 2, 4, 5):
        assert I[idx] == pytest.approx(-np.sqrt(2) / 4, abs=1e-9)


def test_gradient_currents_zero_direction():
    """Zero-length direction yields zero currents (no NaN)."""
    I = fs.gradient_currents([0, 0, 0], 1.0)
    assert np.allclose(I, 0.0)


def test_gradient_currents_flips_with_sign():
    I_pos = fs.gradient_currents([1, 0, 0], 0.5)
    I_neg = fs.gradient_currents([-1, 0, 0], 0.5)
    assert np.allclose(I_pos, -I_neg, atol=1e-9)


# ---------------------------------------------------------------- roll

def test_roll_zero_freq_is_zero():
    I = fs.roll_currents([1, 0, 0], [0, 0, 1], 0.0, 0.5)
    assert np.allclose(I, 0.0)


def test_roll_at_t_zero_is_zero():
    """At t=0, the rotation is identity -> roll delta is zero."""
    I = fs.roll_currents([1, 0, 0], [0, 0, 1], 5.0, 0.0)
    assert np.allclose(I, 0.0, atol=1e-9)


def test_roll_quarter_period_swaps_axes():
    """After a quarter of a period, +X should rotate to +Y around ẑ."""
    freq = 1.0
    quarter_period = 0.25 / freq
    I = fs.roll_currents([1, 0, 0], [0, 0, 1], freq, quarter_period)
    # Total field = static B(X) + roll_delta = uniform_currents(+X) + roll_delta.
    # roll_delta = uniform_currents(+Y) - uniform_currents(+X).
    # Static + delta = uniform_currents(+Y).
    static = fs.uniform_currents([1, 0, 0])
    expected_total = fs.uniform_currents([0, 1, 0])
    assert np.allclose(static + I, expected_total, atol=1e-9)


def test_roll_zero_axis_returns_zero():
    """A zero axis vector is undefined; return zero rather than NaN."""
    I = fs.roll_currents([1, 0, 0], [0, 0, 0], 1.0, 0.5)
    assert np.allclose(I, 0.0)


# ---------------------------------------------------------------- synthesize

def test_synthesize_uniform_only():
    I = fs.synthesize([0, 0, 1], [0, 0, 0], 0.0, [0, 0, 1], 0.0, t=0.0)
    assert np.allclose(I[:3],  0.2357, atol=1e-3)
    assert np.allclose(I[3:], -0.2357, atol=1e-3)


def test_synthesize_clamps():
    """A huge target gets clamped to [-1, 1] per coil."""
    I = fs.synthesize([0, 0, 100], [0, 0, 0], 0.0, [0, 0, 1], 0.0, t=0.0)
    assert np.all(I <= 1.0 + 1e-12)
    assert np.all(I >= -1.0 - 1e-12)


def test_synthesize_applies_gains():
    """Per-coil gains scale the currents post-superposition."""
    I_ungained = fs.synthesize([0, 0, 1], [0, 0, 0], 0.0, [0, 0, 1], 0.0, t=0.0)
    gains = [0.5, 1.0, 1.0, 1.0, 1.0, -1.0]
    I_gained = fs.synthesize([0, 0, 1], [0, 0, 0], 0.0, [0, 0, 1], 0.0, t=0.0,
                             gains=gains)
    assert I_gained[0] == pytest.approx(0.5 * I_ungained[0])
    assert I_gained[5] == pytest.approx(-1.0 * I_ungained[5])


def test_synthesize_gradient_direction_z_is_asymmetric():
    """+Z gradient asymmetrizes top vs bottom."""
    I = fs.synthesize([0, 0, 0], [0, 0, 1], 0.5, [0, 0, 1], 0.0, t=0.0)
    assert np.all(I[:3] > 0)
    assert np.all(I[3:] < 0)


def test_synthesize_gains_wrong_length_raises():
    with pytest.raises(ValueError):
        fs.synthesize([0, 0, 1], [0, 0, 0], 0.0, [0, 0, 1], 0.0,
                      t=0.0, gains=[1, 1, 1])


# ---------------------------------------------------------------- calibration

def test_load_gains_missing_file(tmp_path):
    assert fs.load_gains(str(tmp_path / "nope.json")) is None


def test_load_gains_wrong_length_returns_none(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"coil_gains": [1, 1, 1]}))
    assert fs.load_gains(str(p)) is None


def test_load_gains_invalid_json_returns_none(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text("{ not valid json")
    assert fs.load_gains(str(p)) is None


def test_load_gains_roundtrip(tmp_path):
    p = str(tmp_path / "cal.json")
    fs.save_gains(p, [0.5, 1.0, 1.5, -1.0, 0.0, 2.0])
    loaded = fs.load_gains(p)
    assert loaded == [0.5, 1.0, 1.5, -1.0, 0.0, 2.0]


def test_save_gains_wrong_length_raises(tmp_path):
    with pytest.raises(ValueError):
        fs.save_gains(str(tmp_path / "cal.json"), [1, 2, 3])
