"""motion_filter — constant-velocity Kalman filter for tracked objects.

Pure numpy. No Qt, no OpenCV, no hardware. The tracker's correlation peak
is a *maximum-likelihood* pick: it takes whichever peak is marginally
higher, so two near-identical beads inside the search window are decided
by sensor noise. Weighting the correlation surface by a prior centred on
this filter's prediction turns that into a *MAP* estimate — the peak
consistent with where the bead was heading wins.

State is ``[x, y, vx, vy]`` in raw-image pixels and px/s.

The process model is first-order-lag rather than plain constant velocity:

    v ← v + α·(u − v),      α = 1 − exp(−dt/τ)
    x ← x + v·dt

with ``u`` the commanded velocity in px/s. Microrobots are overdamped —
they have no meaningful momentum, so velocity relaxes toward whatever the
field is commanding with time constant τ rather than persisting. Passing
``u=None`` sets α=0, which degrades exactly to constant velocity (correct
when nothing is driving, or when the drive isn't characterized).

Two entry points, deliberately split so dropped frames can't corrupt the
filter:

* ``project(dt, u)`` — pure, no mutation. Used to build the search prior
  for a frame *before* it's handed to the tracker worker. If that frame
  is later dropped, nothing has been advanced.
* ``step(dt, u, meas)`` — mutates. Called when a detection lands.
  ``meas=None`` coasts (predict only, inflated process noise), which is
  what the contested-state machine uses instead of trusting an ambiguous
  peak.

Run with:  python3 -m pytest classes/gui/test_motion_filter.py -v
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def _lag_alpha(dt: float, tau: float) -> float:
    """Fraction of the way velocity relaxes toward the command over dt."""
    if tau <= 1e-9:
        return 1.0
    if dt <= 0.0:
        return 0.0
    return float(1.0 - math.exp(-dt / tau))


class ConstantVelocityKF:
    """4-state [x, y, vx, vy] filter with an optional velocity command.

    ``accel_noise`` is the white-noise-acceleration spectral density in
    px/s² — the knob that says how hard the bead can deviate from the
    model. ``meas_noise`` is the per-axis detection standard deviation in
    px. Both are per-axis and isotropic.
    """

    def __init__(self,
                 pos: Tuple[float, float],
                 accel_noise_px_s2: float = 400.0,
                 meas_noise_px: float = 2.0,
                 velocity_tau_s: float = 0.15,
                 initial_pos_sigma_px: float = 5.0,
                 initial_vel_sigma_px_s: float = 50.0):
        self.x = np.array([float(pos[0]), float(pos[1]), 0.0, 0.0],
                          dtype=float)
        self.P = np.diag([
            float(initial_pos_sigma_px) ** 2,
            float(initial_pos_sigma_px) ** 2,
            float(initial_vel_sigma_px_s) ** 2,
            float(initial_vel_sigma_px_s) ** 2,
        ])
        self.accel_noise = float(accel_noise_px_s2)
        self.meas_noise = float(meas_noise_px)
        self.tau = float(velocity_tau_s)

    # ---- model pieces ------------------------------------------------

    def _F_B(self, dt: float, u_active: bool):
        """Transition matrix and control gain for a step of ``dt``."""
        a = _lag_alpha(dt, self.tau) if u_active else 0.0
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 2] = 1.0 - a
        F[3, 3] = 1.0 - a
        return F, a

    def _Q(self, dt: float, inflate: float = 1.0) -> np.ndarray:
        """Discrete white-noise-acceleration process covariance."""
        q = (self.accel_noise ** 2) * float(inflate)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q = np.zeros((4, 4))
        Q[0, 0] = Q[1, 1] = dt4 / 4.0
        Q[0, 2] = Q[2, 0] = dt3 / 2.0
        Q[1, 3] = Q[3, 1] = dt3 / 2.0
        Q[2, 2] = Q[3, 3] = dt2
        return Q * q

    def _predict_raw(self, dt: float, u, inflate: float):
        """Return (x_pred, P_pred) without touching self."""
        dt = max(0.0, float(dt))
        F, a = self._F_B(dt, u is not None)
        x = F @ self.x
        if u is not None and a > 0.0:
            x[2] += a * float(u[0])
            x[3] += a * float(u[1])
            # The position row already advanced on the *old* velocity;
            # add the half-step the command contributes over this interval
            # so a step change in command doesn't lag a full frame.
            x[0] += 0.5 * a * float(u[0]) * dt
            x[1] += 0.5 * a * float(u[1]) * dt
        P = F @ self.P @ F.T + self._Q(dt, inflate)
        return x, P

    # ---- public API --------------------------------------------------

    def project(self, dt: float, u=None,
                inflate: float = 1.0) -> Tuple[Tuple[float, float], float]:
        """Predict forward ``dt`` seconds **without mutating** the filter.

        Returns ``((x, y), sigma_px)`` where sigma is the 1-D positional
        standard deviation implied by the predicted covariance — the width
        of the Gaussian to weight the correlation surface with.
        """
        x, P = self._predict_raw(dt, u, inflate)
        var = 0.5 * (float(P[0, 0]) + float(P[1, 1]))
        return (float(x[0]), float(x[1])), math.sqrt(max(var, 0.0))

    def step(self, dt: float, u=None,
             meas: Optional[Tuple[float, float]] = None,
             meas_noise_px: Optional[float] = None,
             inflate: float = 1.0) -> Tuple[float, float]:
        """Advance the filter and optionally fuse a measurement.

        ``meas=None`` coasts: the prediction becomes the new state and the
        covariance grows, so the next prior is correspondingly wider.
        Returns the resulting ``(x, y)``.
        """
        x, P = self._predict_raw(dt, u, inflate)
        self.x, self.P = x, P
        if meas is not None:
            self._fuse(meas, meas_noise_px)
        return (float(self.x[0]), float(self.x[1]))

    def _fuse(self, meas, meas_noise_px: Optional[float]) -> None:
        r = float(meas_noise_px if meas_noise_px is not None
                  else self.meas_noise)
        R = np.eye(2) * (r * r)
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        z = np.array([float(meas[0]), float(meas[1])])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ y
        # Joseph form — stays symmetric positive-definite under the
        # repeated coast/fuse cycling the contested state machine does.
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # ---- accessors ---------------------------------------------------

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))

    @property
    def pos_sigma(self) -> float:
        var = 0.5 * (float(self.P[0, 0]) + float(self.P[1, 1]))
        return math.sqrt(max(var, 0.0))

    def reset_position(self, pos: Tuple[float, float],
                       sigma_px: float = 5.0) -> None:
        """Hard-reseat the position (operator reselect / mask relocalize)."""
        self.x[0] = float(pos[0])
        self.x[1] = float(pos[1])
        self.P[0, 0] = float(sigma_px) ** 2
        self.P[1, 1] = float(sigma_px) ** 2
