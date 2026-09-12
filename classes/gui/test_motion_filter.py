"""Tests for classes.gui.motion_filter — the constant-velocity prior.

Run with:  python3 -m pytest classes/gui/test_motion_filter.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from classes.gui.motion_filter import ConstantVelocityKF, _lag_alpha


DT = 0.05


def test_lag_alpha_bounds():
    assert _lag_alpha(0.0, 0.15) == 0.0
    assert _lag_alpha(1e9, 0.15) == pytest.approx(1.0)
    # tau=0 means "velocity equals command immediately".
    assert _lag_alpha(DT, 0.0) == 1.0
    # Monotone in dt.
    a = [_lag_alpha(d, 0.15) for d in (0.01, 0.05, 0.2, 1.0)]
    assert a == sorted(a)


def test_project_does_not_mutate():
    kf = ConstantVelocityKF((100.0, 50.0))
    before_x = kf.x.copy()
    before_P = kf.P.copy()
    kf.project(DT)
    # A dropped frame must leave the filter untouched — this is why the
    # prior is built with project() and the filter only moves in step().
    assert np.array_equal(kf.x, before_x)
    assert np.array_equal(kf.P, before_P)


def test_tracks_constant_velocity():
    """Feed exact constant-velocity measurements; velocity should converge."""
    vx, vy = 40.0, -15.0
    kf = ConstantVelocityKF((0.0, 0.0), meas_noise_px=0.5)
    for k in range(1, 60):
        kf.step(DT, None, (vx * k * DT, vy * k * DT))
    assert kf.vel[0] == pytest.approx(vx, rel=0.05)
    assert kf.vel[1] == pytest.approx(vy, rel=0.05)


def test_prediction_leads_the_measurement():
    """The whole point: the prior must sit ahead of the last measurement."""
    kf = ConstantVelocityKF((0.0, 0.0), meas_noise_px=0.5)
    for k in range(1, 40):
        kf.step(DT, None, (30.0 * k * DT, 0.0))
    last_x = kf.pos[0]
    pred, _sigma = kf.project(DT)
    assert pred[0] > last_x + 1.0


def test_coasting_widens_sigma():
    kf = ConstantVelocityKF((0.0, 0.0), meas_noise_px=0.5)
    for k in range(1, 30):
        kf.step(DT, None, (30.0 * k * DT, 0.0))
    tight = kf.pos_sigma
    for _ in range(5):
        kf.step(DT, None, None, inflate=4.0)
    # Coasting must express growing doubt, otherwise the next prior would
    # stay narrow and could lock onto the wrong peak with false confidence.
    assert kf.pos_sigma > tight


def test_measurement_pulls_toward_truth():
    kf = ConstantVelocityKF((0.0, 0.0), meas_noise_px=1.0)
    kf.step(DT, None, (10.0, 0.0))
    assert 0.0 < kf.pos[0] <= 10.0


def test_control_input_moves_prediction_without_measurements():
    """With a commanded velocity the filter predicts motion from a
    standing start — it does not need to observe the move first."""
    kf = ConstantVelocityKF((0.0, 0.0), velocity_tau_s=0.1)
    plain, _ = kf.project(DT, None)
    driven, _ = kf.project(DT, (200.0, 0.0))
    assert plain[0] == pytest.approx(0.0)
    assert driven[0] > 0.0


def test_control_input_converges_to_commanded_speed():
    kf = ConstantVelocityKF((0.0, 0.0), velocity_tau_s=0.1)
    for _ in range(80):
        kf.step(DT, (120.0, 0.0), None)
    assert kf.vel[0] == pytest.approx(120.0, rel=0.05)


def test_covariance_stays_symmetric_under_coast_fuse_cycling():
    """Joseph-form update: the contested state machine alternates coast
    and fuse indefinitely, which is where naive (I-KH)P loses symmetry."""
    kf = ConstantVelocityKF((0.0, 0.0))
    for k in range(200):
        if k % 3 == 0:
            kf.step(DT, None, None, inflate=4.0)
        else:
            kf.step(DT, None, (float(k), 0.0))
    assert np.allclose(kf.P, kf.P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(kf.P) > -1e-9)


def test_reset_position():
    kf = ConstantVelocityKF((0.0, 0.0))
    kf.step(DT, None, (5.0, 5.0))
    kf.reset_position((900.0, 20.0), sigma_px=7.0)
    assert kf.pos == pytest.approx((900.0, 20.0))
    assert kf.pos_sigma == pytest.approx(7.0, rel=1e-6)
