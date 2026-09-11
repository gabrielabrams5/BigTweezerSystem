"""robot_model — physics models for the three robot types.

Selected at startup from ``config.robot.type``. Each model exposes:

  * ``moment(B)`` — the robot's magnetic moment given the current field.
    Used by the DLS solver to build ``Fmap(m)``. For paramagnetic beads the
    force is not linear in current so the model returns None and the solver
    falls back to the closed-form pull rule instead.

  * ``label`` — a short display name for the GUI.

The paramagnetic model is what the Spherotech CFM rig uses. The hard/soft
models are stubs kept for the future when we add magnetic swimmers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


PARAMAGNETIC = "paramagnetic"
SOFT = "soft"
HARD = "hard"


@dataclass
class RobotModel:
    kind: str
    label: str
    m_mag: float = 0.0        # hard-magnetic |m| (A·m²)
    k_susc: float = 0.0       # soft-magnetic susceptibility (A·m²/T)

    def moment(self, B: np.ndarray) -> Optional[np.ndarray]:
        """Return the robot moment vector m for a given field B, or None if
        the linear F = G·m model doesn't apply (paramagnetic case).
        """
        B = np.asarray(B, dtype=float).reshape(3)
        if self.kind == HARD:
            n = float(np.linalg.norm(B))
            if n < 1e-12:
                # No field → no defined moment direction. Return the +z
                # default so callers don't crash; the resulting Fmap will
                # be a zero-force starting point.
                return np.array([0.0, 0.0, self.m_mag])
            return self.m_mag * B / n
        if self.kind == SOFT:
            return self.k_susc * B
        return None  # paramagnetic — force isn't linear, no moment needed


def build_from_config(config) -> RobotModel:
    """Read ``robot.*`` from Config and return a RobotModel instance."""
    kind = config.get("robot.type", PARAMAGNETIC).lower()
    if kind not in (PARAMAGNETIC, SOFT, HARD):
        raise ValueError(f"unknown robot.type {kind!r}")
    return RobotModel(
        kind=kind,
        label={
            PARAMAGNETIC: "Paramagnetic bead (pull-only)",
            SOFT: "Soft-magnetic (induced m = k·B)",
            HARD: "Hard-magnetic (permanent |m|)",
        }[kind],
        m_mag=float(config.get("robot.m_mag", 0.0)),
        k_susc=float(config.get("robot.k_susc", 0.0)),
    )
