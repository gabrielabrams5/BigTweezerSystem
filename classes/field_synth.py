"""
Field synthesis for the 6-coil aligned-ring 3D tweezer.

Rig geometry (single source of truth for the whole app):

  Three top coils and three bottom coils. Top and bottom rings share the same
  three azimuths (0, 120, 240 degrees) -- NOT staggered. Each coil axis makes
  a 45 degree angle with vertical, tips pointing inward toward the workspace
  center. Positive current on coil i produces field at the origin in the
  direction of its unit axis n_i (as defined below).

  Coil layout (looking down the +Z axis):

                  C2 (top, 120)
                     .
                   .   .
       C3 (top,   .     .    C1 (top, 0)
       240) ----- .  o  . ------------ +X axis
                   .   .
                     .
                  C2..C1: 45 deg from vertical
                  C4..C6: mirrored below,
                    same azimuths, 45 deg
                    below horizontal.

The columns of `COIL_AXES` are the six unit axis vectors (in coil order
C1..C6). Multiplying by a per-coil current vector gives the resulting
uniform field at the origin, up to a global scale factor absorbed into
the calibration gains.

All quantities are dimensionless in this module: field targets are on the
scale [-1, 1] and per-coil currents are on the scale [-1, 1] (the H-bridge
PWM duty). Real-world scaling is handled by the calibration gains.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np


# ------------------------------------------------------------------ geometry

# sin(45 deg) = cos(45 deg) = sqrt(2)/2
_S = np.sqrt(2) / 2

# Azimuths for top and bottom rings. Aligned (same three azimuths, not staggered).
_AZIMUTHS = np.deg2rad([0.0, 120.0, 240.0])

# Coil axis unit vectors (columns of a 3x6 matrix). Order C1, C2, C3, C4, C5, C6.
# Top ring: axes tilt 45 deg above horizontal (positive Z component).
# Bottom ring: mirror below (negative Z component).
COIL_AXES: np.ndarray = np.column_stack([
    # Top ring C1..C3
    *(np.array([_S * np.cos(a), _S * np.sin(a),  _S]) for a in _AZIMUTHS),
    # Bottom ring C4..C6
    *(np.array([_S * np.cos(a), _S * np.sin(a), -_S]) for a in _AZIMUTHS),
])
assert COIL_AXES.shape == (3, 6)

# A @ A.T works out to diag(3/2, 3/2, 3) exactly for this geometry. Its
# inverse is diag(2/3, 2/3, 1/3). We precompute it so uniform_currents is
# just two matmuls.
_A_ATRANS_INV = np.diag([2 / 3, 2 / 3, 1 / 3])


# ------------------------------------------------------------------ synthesis

def uniform_currents(B_target: Iterable[float]) -> np.ndarray:
    """Minimum-norm currents for a target uniform field at origin.

    Solves ``A @ I = B_target`` where A = COIL_AXES. Since A is 3x6 the
    system is underdetermined; we return the minimum-norm solution
    ``I = A^T (A A^T)^-1 B_target`` which spreads current across all six
    coils rather than concentrating in a few.

    B_target scale is arbitrary but usually in [-1, 1]; the resulting
    currents scale linearly. Values may exceed [-1, 1] for large targets --
    callers should clamp at the final synthesize() stage.
    """
    B = np.asarray(B_target, dtype=float).reshape(3)
    return COIL_AXES.T @ _A_ATRANS_INV @ B


def gradient_currents(direction: Iterable[float], magnitude: float) -> np.ndarray:
    """Currents that push the field magnitude peak toward the given direction.

    Model: for each coil i, drive current ``g * (n_i . d_hat)`` where d_hat is
    the unit direction and g is the magnitude. This is a heuristic (the
    aligned geometry does not give clean anti-Helmholtz pairs), but it maps
    cleanly:

      * direction = +Z: top ring fires positive, bottom ring negative -> field
        peaks on the +Z side, pulling paramagnetic beads upward.
      * direction = +X: coils C1 and C4 (both axes have +X component) fire
        strongly; C2, C3, C5, C6 fire negative -> peak on +X side.
    """
    d = np.asarray(direction, dtype=float).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.zeros(6)
    d_hat = d / n
    return float(magnitude) * (COIL_AXES.T @ d_hat)


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation. `axis` is auto-normalized; a zero axis returns identity."""
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    u = axis / n
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    ux, uy, uz = u
    K = np.array([
        [0.0, -uz,  uy],
        [ uz, 0.0, -ux],
        [-uy,  ux, 0.0],
    ])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(u, u)


def roll_currents(base_B: Iterable[float], axis: Iterable[float],
                  freq_hz: float, t: float) -> np.ndarray:
    """Zero-mean current contribution from rolling `base_B` around `axis` at freq.

    Rolling rotates the base uniform field vector around a chosen axis at
    ``freq_hz`` Hz. The rolled contribution is *added on top of* the static
    uniform currents -- to avoid double-counting we return the rolled-minus-
    base currents, so the total is (roll_freq -> 0) -> exact static field.
    """
    base = np.asarray(base_B, dtype=float).reshape(3)
    ax = np.asarray(axis, dtype=float).reshape(3)
    if abs(float(freq_hz)) < 1e-9:
        return np.zeros(6)
    R = _rotation_matrix(ax, 2 * np.pi * float(freq_hz) * float(t))
    return uniform_currents(R @ base) - uniform_currents(base)


#: Set True when the physical rig can only attract (electromagnet + paramagnetic
#: bead). Under pull-only physics, current sign doesn't matter -- reversing a
#: coil's current still pulls the bead toward that coil, just wastes power.
#: The correct synthesis is I_i = max(0, n_i . B_commanded): fire each coil
#: proportional to how well its axis projects onto the commanded direction,
#: and never in reverse.
PULL_ONLY = True


def pull_only_currents(B_target: Iterable[float]) -> np.ndarray:
    """Fire each coil proportional to max(0, n_i . B_target). This is the
    direct formula for attracting a paramagnetic bead toward B_target's
    direction: only coils whose axes point *toward* that direction fire,
    and their strength scales with how well they're aligned. No opposing
    coil "cancels" another's pull -- the total force always adds toward
    B_target."""
    B = np.asarray(B_target, dtype=float).reshape(3)
    return np.maximum(COIL_AXES.T @ B, 0.0)


def synthesize(uniform_B: Iterable[float],
               gradient_dir: Iterable[float],
               gradient_mag: float,
               roll_axis: Iterable[float],
               roll_freq_hz: float,
               t: float,
               gains: Iterable[float] | None = None,
               pull_only: bool | None = None) -> np.ndarray:
    """Compose the commanded field and map to per-coil currents.

    Composition:
      1) Start from uniform_B (operator's base direction).
      2) Add gradient_mag * gradient_dir_hat (a directional bias).
      3) If roll_freq_hz != 0, rotate the composed vector around roll_axis
         at roll_freq_hz Hz -- this is how rolling works under pull-only
         physics: the commanded direction precesses, dragging the bead's
         magnetic dipole with it.
      4) Fire coils.

    Under pull_only (default = PULL_ONLY = True), step 4 uses
    pull_only_currents(). Otherwise the pseudoinverse (uniform_currents)
    gives a proper signed B field for rigs where reversing polarity
    actually pushes the bead."""
    B_u = np.asarray(uniform_B, dtype=float).reshape(3)
    grad_d = np.asarray(gradient_dir, dtype=float).reshape(3)
    grad_n = np.linalg.norm(grad_d)
    B_g = float(gradient_mag) * grad_d / grad_n if grad_n > 1e-12 else np.zeros(3)
    B_base = B_u + B_g

    if abs(float(roll_freq_hz)) > 1e-9:
        R = _rotation_matrix(
            np.asarray(roll_axis, dtype=float).reshape(3),
            2 * np.pi * float(roll_freq_hz) * float(t),
        )
        B_cmd = R @ B_base
    else:
        B_cmd = B_base

    if pull_only is None:
        pull_only = PULL_ONLY
    if pull_only:
        I = pull_only_currents(B_cmd)
    else:
        I = uniform_currents(B_cmd)  # signed pseudoinverse

    if gains is not None:
        g = np.asarray(list(gains), dtype=float)
        if g.shape != (6,):
            raise ValueError(f"gains must have shape (6,), got {g.shape}")
        I = I * g

    lo = 0.0 if pull_only else -1.0
    return np.clip(I, lo, 1.0)


# ------------------------------------------------------------------ calibration

def default_gains() -> list:
    return [1.0] * 6


def default_channel_map() -> list:
    """Identity map: logical coil i -> physical driver i. Override in
    calibration.json if the rig's wiring routes a driver output to a
    different physical coil position."""
    return [0, 1, 2, 3, 4, 5]


def _read_calibration(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_gains(path: str):
    """Return the six calibration gains, or None if missing/invalid.
    File format (both fields optional; missing fields fall back to defaults)::

        { "coil_gains": [g1..g6], "channel_map": [c1..c6] }

    Negative gains flip H-bridge polarity for a coil wound backward. Zero
    silences that coil.
    """
    data = _read_calibration(path)
    if data is None:
        return None
    g = data.get("coil_gains")
    if not isinstance(g, list) or len(g) != 6:
        return None
    try:
        return [float(x) for x in g]
    except (TypeError, ValueError):
        return None


def load_channel_map(path: str):
    """Return the six-entry driver permutation. channel_map[i] = which
    physical driver logical coil i is wired to. Default identity if the
    file has no channel_map field.
    """
    data = _read_calibration(path)
    if data is None:
        return None
    m = data.get("channel_map")
    if not isinstance(m, list) or len(m) != 6:
        return None
    try:
        m = [int(x) for x in m]
    except (TypeError, ValueError):
        return None
    if sorted(m) != [0, 1, 2, 3, 4, 5]:
        # Not a valid permutation of 0..5; refuse silently.
        return None
    return m


def save_calibration(path: str, gains, channel_map=None) -> None:
    """Save both gains and channel_map (optional) to calibration.json.
    channel_map is accepted as any six-tuple of ints in 0..5; permutation
    validation happens at load time only, so mid-edit state isn't lost."""
    g = list(gains)
    if len(g) != 6:
        raise ValueError(f"expected 6 gains, got {len(g)}")
    payload = {"coil_gains": [float(x) for x in g]}
    if channel_map is not None:
        m = [int(x) for x in channel_map]
        if len(m) != 6:
            raise ValueError(f"channel_map must have 6 entries, got {len(m)}")
        payload["channel_map"] = m
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# Backwards-compat alias used by earlier code paths.
def save_gains(path: str, gains: Iterable[float]) -> None:
    save_calibration(path, gains)
