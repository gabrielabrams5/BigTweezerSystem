"""Tests for the frame-calibration fit math (pure functions, no Qt).

The core regression: the fit must be DIRECTION-TRUE — commanding
``w = M_sw @ u`` must produce screen motion ``S @ w`` parallel to the stick
vector ``u`` for every direction, even on skewed / anisotropic / mirrored
rigs. The old per-column normalization broke exactly this.

Run with:  python3 -m pytest classes/gui/test_frame_calibration_fit.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from classes.gui.frame_calibration import (
    build_screen_columns,
    fit_quality,
    fit_screen_to_world,
)


# A realistic nasty rig: skewed (~57° between columns), mirrored (det < 0),
# anisotropic (column norms 1.0 vs ~1.66).
S_RIG = np.array([[1.0, 0.9],
                  [0.0, -1.4]])


def simulate(S, k=1.0, drift=(0.0, 0.0)):
    """Displacements a rig with screen-up response S would produce, in RAW
    image coords (y grows down), for one hold of common gain k, plus a
    constant tracker drift per hold."""
    out = {}
    for key, v in (("+x", (1, 0)), ("-x", (-1, 0)),
                   ("+y", (0, 1)), ("-y", (0, -1))):
        s = k * (S @ np.asarray(v, dtype=float))
        out[key] = (s[0] + drift[0], -s[1] + drift[1])
    return out


def unit(deg):
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


STICK_ANGLES = (0, 30, 45, 90, 137, 200, 300)


def test_fit_is_direction_true():
    M_sw, q = fit_screen_to_world(simulate(S_RIG))
    assert q["ok"]
    for deg in STICK_ANGLES:
        u = unit(deg)
        s = S_RIG @ (M_sw @ u)
        cross = s[0] * u[1] - s[1] * u[0]
        assert cross == pytest.approx(0.0, abs=1e-9), f"angle {deg}"
        assert float(np.dot(s, u)) > 0.0, f"angle {deg}"


def test_old_normalized_column_fit_is_not_direction_true():
    """Documents the bug the raw fit replaces: per-column normalization
    before inversion distorts directions on anisotropic rigs."""
    norms = np.linalg.norm(S_RIG, axis=0)
    M_old = np.linalg.inv(S_RIG / norms)
    worst = max(
        abs((S_RIG @ (M_old @ unit(d)))[0] * unit(d)[1] -
            (S_RIG @ (M_old @ unit(d)))[1] * unit(d)[0])
        for d in STICK_ANGLES)
    assert worst > 1e-3


def test_fit_is_invariant_to_pulse_gain():
    M1, _ = fit_screen_to_world(simulate(S_RIG, k=1.0))
    M2, _ = fit_screen_to_world(simulate(S_RIG, k=57.3))
    assert np.allclose(M1, M2, atol=1e-12)


def test_scalar_normalization_max_col_norm_is_one():
    M_sw, _ = fit_screen_to_world(simulate(S_RIG))
    n0 = float(np.linalg.norm(M_sw[:, 0]))
    n1 = float(np.linalg.norm(M_sw[:, 1]))
    assert max(n0, n1) == pytest.approx(1.0)
    assert n0 <= 1.0 + 1e-12 and n1 <= 1.0 + 1e-12


def test_two_sided_averaging_cancels_drift():
    M_clean, _ = fit_screen_to_world(simulate(S_RIG))
    M_drift, _ = fit_screen_to_world(simulate(S_RIG, drift=(4.0, -3.0)))
    assert np.allclose(M_clean, M_drift, atol=1e-12)


def test_one_sided_axis_still_fits():
    d = simulate(S_RIG)
    del d["-x"], d["-y"]
    M_sw, q = fit_screen_to_world(d)
    assert q["ok"]
    for deg in STICK_ANGLES:
        u = unit(deg)
        s = S_RIG @ (M_sw @ u)
        assert s[0] * u[1] - s[1] * u[0] == pytest.approx(0.0, abs=1e-9)
        assert float(np.dot(s, u)) > 0.0


def test_dict_measurements_normalized_by_freq_dur():
    """Dict-form measurements divide by freq × dur, so pulses of different
    hold lengths still fit the same matrix."""
    base = simulate(S_RIG)
    mixed = {}
    for i, (key, d) in enumerate(base.items()):
        gain = [1.0, 2.0, 0.5, 3.0][i]
        mixed[key] = {"d": (d[0] * gain, d[1] * gain),
                      "mag": 1.0, "freq": gain, "dur": 1.0}
    M_plain, _ = fit_screen_to_world(base)
    M_mixed, _ = fit_screen_to_world(mixed)
    assert np.allclose(M_plain, M_mixed, atol=1e-12)


def test_static_measurements_normalize_by_duration_only():
    """A Mode B pulse has no frequency — the bead drifts at terminal
    velocity, so displacement scales with time alone and the fitted matrix
    is px/s. Dividing by a stale `freq` here would corrupt the scale."""
    base = simulate(S_RIG)
    mixed = {}
    for i, (key, d) in enumerate(base.items()):
        dur = [1.0, 2.0, 0.5, 3.0][i]
        mixed[key] = {"d": (d[0] * dur, d[1] * dur),
                      "mag": 1.0,
                      # A leftover freq value must be IGNORED for static.
                      "freq": 7.0, "dur": dur, "recipe": "static"}
    M_plain, _ = fit_screen_to_world(base)
    M_mixed, q = fit_screen_to_world(mixed)
    assert np.allclose(M_plain, M_mixed, atol=1e-12)
    # And the response matrix keeps the per-second scale.
    assert np.allclose(q["S"], build_screen_columns(base), atol=1e-12)


def test_rolling_and_static_recipes_normalize_differently():
    """Same raw displacements, same dur, differing freq: rolling divides it
    out, static does not. If these ever agree the recipe tag stopped
    reaching _measurement_delta."""
    raw = simulate(S_RIG)
    def tag(recipe):
        return {k: {"d": v, "mag": 1.0, "freq": 4.0, "dur": 1.0,
                    "recipe": recipe} for k, v in raw.items()}
    S_roll = build_screen_columns(tag("rolling"))
    S_static = build_screen_columns(tag("static"))
    assert np.allclose(S_static, S_roll * 4.0, atol=1e-12)


def test_untagged_measurements_still_treated_as_rolling():
    """Back-compat: measurements saved before the recipe tag existed."""
    raw = simulate(S_RIG)
    untagged = {k: {"d": v, "mag": 1.0, "freq": 4.0, "dur": 1.0}
                for k, v in raw.items()}
    tagged = {k: dict(v, recipe="rolling") for k, v in untagged.items()}
    assert np.allclose(build_screen_columns(untagged),
                       build_screen_columns(tagged), atol=1e-12)


def test_quality_reports_angle_and_mirror():
    S = build_screen_columns(simulate(S_RIG))
    q = fit_quality(S)
    assert q["ok"]
    assert q["mirrored"] is True
    assert q["angle_deg"] == pytest.approx(57.26, abs=0.5)
    assert q["anisotropy"] == pytest.approx(1.664, abs=0.01)


def test_collinear_columns_rejected():
    # Nearly-collinear columns at huge pixel scale: |det| is far above the
    # old absolute 1e-6 threshold, but the scale-free check must reject.
    S = 1000.0 * np.array([[1.0, 2.0], [1.0, 2.0001]])
    d = simulate(S)
    M_sw, q = fit_screen_to_world(d)
    assert M_sw is None
    assert not q["ok"]
    assert "collinear" in q["reason"]


def test_zero_displacement_axis_rejected():
    d = simulate(S_RIG)
    d["+x"] = (0.0, 0.0)
    d["-x"] = (0.0, 0.0)
    M_sw, q = fit_screen_to_world(d)
    assert M_sw is None
    assert not q["ok"]
    assert "±X" in q["reason"]


def test_identity_rig_round_trips():
    M_sw, q = fit_screen_to_world(simulate(np.eye(2)))
    assert q["ok"]
    assert np.allclose(M_sw, np.eye(2), atol=1e-12)
