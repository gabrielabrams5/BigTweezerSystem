"""field_solver — pure-math coil-current solver.

Given a target field (and optionally a target force for signed-drive rigs),
return the six per-coil duties. Zero hardware imports so this module is
fully unit-testable against synthetic matrices.

Three physics reductions live here:

  * **paramagnetic** (default for Spherotech CFM beads and similar): force
    is always attractive (``F ∝ ∇|B|²``); the correct coil pattern is
    ``max(0, Bmap.T @ B_des)`` peak-normalized so the strongest coil
    saturates at ``|B_des|``. Closed form, no DLS.

  * **soft / hard** (signed-drive rigs, planned but not the current
    hardware): damped-least-squares (Tikhonov) solve of
    ``[Bmap; Fmap(m)] · i = [B_des; F_des]``, with null-space projection
    for underdetermined Mode A (only field constrained).

The paramagnetic path is what the operator actually uses today. The
DLS path is written but exercised only by tests until we get hard-magnetic
swimmers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Robot types (mirrors classes/robot_model.py).
PARAMAGNETIC = "paramagnetic"
SOFT = "soft"
HARD = "hard"

# Saturation policies.
RESCALE = "rescale_preserve_direction"
FLAG = "flag"

# Static (Mode B) synthesis rules for paramagnetic rigs.
#   NNLS       — non-negative least squares; realizes the commanded
#                direction (default).
#   PROJECTION — historical max(0, n̂·B) heuristic; ±30° direction error.
#                Kept as an escape hatch.
NNLS = "nnls"
PROJECTION = "projection"


@dataclass
class SolveResult:
    """Solver output. ``i`` is always the 6-vector of duties; the flags let
    the caller notice saturation or ill-conditioned solves without having to
    re-run the math."""
    i: np.ndarray
    saturated: bool
    residual: float          # ‖A·i − y‖ post-saturation


class FieldSolver:
    """Pure-math solver. Owns the calibration matrices and solver knobs.

    Callers rebuild the solver when calibration changes (rare) but reuse it
    across every inner-loop tick. Nothing here allocates in the hot path.
    """

    def __init__(self,
                 Bmap: np.ndarray,
                 Gmap: np.ndarray | None = None,
                 robot_type: str = PARAMAGNETIC,
                 lam: float = 0.01,
                 W: np.ndarray | None = None,
                 null_objective: str = "minimize_gradient",
                 saturation_policy: str = RESCALE,
                 i_max_per_coil: np.ndarray | None = None,
                 per_coil_gains: np.ndarray | None = None,
                 static_synthesis: str = NNLS,
                 static_ridge: float = 1e-4):
        self.Bmap = np.asarray(Bmap, dtype=float)
        if self.Bmap.shape != (3, 6):
            raise ValueError(f"Bmap must be (3,6), got {self.Bmap.shape}")
        self.Gmap = np.asarray(Gmap, dtype=float) if Gmap is not None else np.zeros((5, 6))
        if self.Gmap.shape != (5, 6):
            raise ValueError(f"Gmap must be (5,6), got {self.Gmap.shape}")
        self.robot_type = robot_type
        self.lam = float(lam)
        self.W = np.asarray(W, dtype=float) if W is not None else np.ones(6)
        if self.W.shape != (6,):
            raise ValueError(f"W must be (6,), got {self.W.shape}")
        self.null_objective = null_objective
        self.saturation_policy = saturation_policy
        self.i_max_per_coil = (np.asarray(i_max_per_coil, dtype=float)
                               if i_max_per_coil is not None else np.ones(6))
        self.per_coil_gains = (np.asarray(per_coil_gains, dtype=float)
                               if per_coil_gains is not None else np.ones(6))
        self.static_synthesis = str(static_synthesis)
        self.static_ridge = float(static_ridge)

    # ---- classmethods ------------------------------------------------

    @classmethod
    def from_config(cls, config) -> "FieldSolver":
        """Build a solver from a Config instance (classes.config_loader.Config)."""
        return cls(
            Bmap=config.get_matrix("calibration.Bmap"),
            Gmap=config.get_matrix("calibration.Gmap"),
            robot_type=config.get("robot.type", PARAMAGNETIC),
            lam=float(config.get("solver.lambda", 0.01)),
            W=np.asarray(config.get("solver.W", [1.0] * 6), dtype=float),
            null_objective=config.get("solver.null_objective", "minimize_gradient"),
            saturation_policy=config.get("solver.saturation_policy", RESCALE),
            i_max_per_coil=np.asarray(
                config.get("calibration.i_max_per_coil", [1.0] * 6), dtype=float),
            per_coil_gains=np.asarray(
                config.get("calibration.per_coil_gains", [1.0] * 6), dtype=float),
            static_synthesis=config.get("solver.static_synthesis", NNLS),
            static_ridge=float(config.get("solver.static_ridge", 1e-4)),
        )

    # ---- solve -------------------------------------------------------

    def solve(self,
              B_des: Any,
              F_des: Any = None,
              m: Any = None,
              static: bool = False) -> SolveResult:
        """Return per-coil duties for the commanded field (and force, for
        signed-drive rigs).

        For ``robot_type == PARAMAGNETIC`` the closed-form
        ``max(0, Bmap.T @ B_des)`` with peak normalization is used and
        ``F_des`` / ``m`` are ignored (force follows from |B| gradient).

        ``static=True`` marks a Mode-B (static pull) command, where the
        realized field direction is what the operator steers by and the
        projection heuristic's direction error is therefore intolerable —
        see :meth:`_solve_paramagnetic_exact`. Mode A leaves this False and
        keeps the historical rule bit-for-bit.
        """
        B_des = np.asarray(B_des, dtype=float).reshape(3)
        if self.robot_type == PARAMAGNETIC:
            if static and self.static_synthesis == NNLS:
                i = self._solve_paramagnetic_exact(B_des)
            else:
                i = self._solve_paramagnetic(B_des)
        else:
            i = self._solve_dls(B_des, F_des, m)
        i = self.per_coil_gains * i
        i, saturated = self._apply_saturation(i)
        # Residual computed after saturation, in the same frame the solver
        # actually optimized.
        residual = float(np.linalg.norm(self.Bmap @ i - B_des))
        return SolveResult(i=i, saturated=saturated, residual=residual)

    # ---- paramagnetic closed form -----------------------------------

    def _solve_paramagnetic(self, B_des: np.ndarray) -> np.ndarray:
        """max(0, n̂_j · B_des) per coil, peak-normalized so the strongest
        coil hits |B_des|. Equivalent to the pull-only rule from
        field_synth.pull_only_currents but generalized to arbitrary
        (measured) Bmap matrices — the column direction is what matters,
        not just the geometric axis."""
        raw = np.maximum(self.Bmap.T @ B_des, 0.0)
        return self._peak_normalize(raw, B_des)

    def _peak_normalize(self, raw: np.ndarray,
                        B_des: np.ndarray) -> np.ndarray:
        """Scale so the strongest coil sits at ``|B_des|``.

        Operator ergonomics, not physics: without it a full-deflection
        command would never show 100% duty on any coil.
        """
        peak = float(raw.max())
        if peak < 1e-12:
            return raw
        return raw * (float(np.linalg.norm(B_des)) / peak)

    def _solve_paramagnetic_exact(self, B_des: np.ndarray) -> np.ndarray:
        """Non-negative least squares for the commanded field direction.

        The projection rule above is a *heuristic*: it fires each coil by
        how well its axis projects onto the command, which does not
        reproduce the commanded direction. With three coil azimuths
        (0/120/240°) the realized horizontal field is a ±30° sawtooth about
        the command — exact only at multiples of 60°, and with flat dead
        zones where a whole 30° range of commands maps to one realized
        direction. Rolling averages that away over each precession cycle
        and is calibrated end-to-end anyway; static pull steers directly by
        the realized direction, so there it is the dominant error.

        The commanded direction *is* reachable with non-negative currents —
        the heuristic simply doesn't find it. Solving

            min ‖[Bmap; √λ·I]·i − [B_des; 0]‖   subject to   i ≥ 0

        does, to within 0.004° across the full azimuth sweep, while keeping
        every current non-negative (pull-only physics is preserved — a
        reversed coil still attracts, so signed drive would be meaningless).

        The ridge term λ is not for conditioning. Plain NNLS returns a basic
        solution with as few active coils as possible, which breaks the
        top/bottom ring symmetry and tilts the |B| landscape out of plane.
        A small λ selects the minimum-norm solution among the exact ones,
        which restores symmetric firing (measured: asymmetry 1.0 → 0.0 at
        λ=1e-4) and costs 0.004° of direction accuracy.
        """
        from scipy.optimize import nnls

        lam = max(0.0, float(self.static_ridge))
        A = np.vstack([self.Bmap, np.sqrt(lam) * np.eye(6)])
        b = np.concatenate([B_des, np.zeros(6)])
        try:
            raw, _residual = nnls(A, b)
        except Exception:
            # Degenerate input — fall back rather than dropping a tick.
            return self._solve_paramagnetic(B_des)
        return self._peak_normalize(raw, B_des)

    # ---- DLS for signed-drive rigs ----------------------------------

    def _solve_dls(self,
                   B_des: np.ndarray,
                   F_des: Any,
                   m: Any) -> np.ndarray:
        """Tikhonov damped least squares.

        Underdetermined (Mode A, only 3 rows from Bmap): solves for
        minimum-‖i‖ (or minimum-‖Gmap·i‖, per null_objective) satisfying
        the field constraint approximately.

        Exactly determined (Mode B, 6 rows: field + force): standard 6×6
        DLS solve.
        """
        A_rows: list[np.ndarray] = [self.Bmap]
        y_rows: list[np.ndarray] = [B_des]

        if F_des is not None and m is not None:
            m_vec = np.asarray(m, dtype=float).reshape(3)
            Fmap = self._build_fmap(m_vec)
            A_rows.append(Fmap)
            y_rows.append(np.asarray(F_des, dtype=float).reshape(3))

        A = np.vstack(A_rows)
        y = np.concatenate(y_rows)

        Winv = np.diag(1.0 / self.W)
        M = A @ Winv @ A.T + self.lam * np.eye(A.shape[0])
        # np.linalg.solve is stable enough for 3×3 or 6×6.
        i = Winv @ A.T @ np.linalg.solve(M, y)

        if A.shape[0] < 6 and self.null_objective == "minimize_gradient":
            # Project a small "shrink ‖Gmap·i‖" step onto the null space of A.
            # Cheap heuristic: one Newton step toward min ‖Gmap·(i+Δ)‖² under
            # A·Δ = 0. Kept intentionally soft (μ=0.5) so this never fights
            # the primary constraint.
            i = self._nullspace_shrink_gradient(A, i, mu=0.5)
        return i

    def _build_fmap(self, m: np.ndarray) -> np.ndarray:
        """Contract Gmap (5 independent components) with m to get 3×6 Fmap.

        Independent components ordering (matches config layout):
          g0 = ∂Bx/∂x, g1 = ∂Bx/∂y, g2 = ∂Bx/∂z, g3 = ∂By/∂y, g4 = ∂By/∂z.

        Recover the rest via Maxwell:
          ∂By/∂x = g1, ∂Bz/∂x = g2, ∂Bz/∂y = g4, ∂Bz/∂z = -(g0 + g3).

        F = G · m (in the linearized model). Row a of Fmap is
          Σ_j Gmap[k][j] * (∂B_a/∂x_b matrix entry) * m_b — computed as
          a linear combination of Gmap columns per coil.
        """
        mx, my, mz = m
        # Build a 3×5 matrix S such that F_a = S[a, k] * g_k applied column-wise.
        S = np.array([
            [mx,  my, mz, 0.0, 0.0],           # F_x = mx*g0 + my*g1 + mz*g2
            [0.0, mx, 0.0, my, mz],            # F_y = mx*g1 + my*g3 + mz*g4
            [-mz, 0.0, mx, -mz, my],           # F_z = mx*g2 + my*g4 - mz*(g0+g3)
        ])
        # For F_z the algebra: ∂Bz/∂x = g2 → mx*g2. ∂Bz/∂y = g4 → my*g4.
        # ∂Bz/∂z = -(g0+g3) → mz*(-(g0+g3)) = -mz*g0 - mz*g3.
        # So F_z row is [-mz, 0, mx, -mz, my] — matches above.
        return S @ self.Gmap  # (3,5) @ (5,6) = (3,6)

    def _nullspace_shrink_gradient(self,
                                   A: np.ndarray,
                                   i: np.ndarray,
                                   mu: float) -> np.ndarray:
        """Project one step of ‖Gmap·i‖² gradient onto the null space of A."""
        try:
            # Basis for null space of A (columns).
            s, vt = np.linalg.svd(A, full_matrices=True)[1:]
            rank = int(np.sum(s > 1e-10))
            if rank >= A.shape[1]:
                return i
            N = vt[rank:].T  # (6, dim_null)
            grad = 2.0 * self.Gmap.T @ (self.Gmap @ i)
            delta = -mu * N @ (N.T @ grad)
            # Cap Δ so the shrink can't blow up in near-singular directions.
            n_delta = float(np.linalg.norm(delta))
            if n_delta > 1.0:
                delta = delta / n_delta
            return i + delta
        except np.linalg.LinAlgError:
            return i

    # ---- saturation --------------------------------------------------

    def _apply_saturation(self, i: np.ndarray) -> tuple[np.ndarray, bool]:
        """Enforce per-coil ranges. For paramagnetic the lower bound is 0.

        rescale_preserve_direction:
            If any coil exceeds its i_max, divide the whole vector by the
            worst overshoot so the direction is preserved but magnitude
            drops. This is what the user asked for by default — "keep the
            robot going the right way, just slower."

        flag:
            Just clip and mark saturated.
        """
        lo = 0.0 if self.robot_type == PARAMAGNETIC else -1.0
        i_max = np.maximum(self.i_max_per_coil, 1e-9)

        if self.saturation_policy == RESCALE:
            # Compute overshoot ratio; scale down if any coil is over.
            ratio_high = np.max(i / i_max) if np.max(i) > 0 else 0.0
            ratio_low = 0.0
            if lo < 0:
                ratio_low = np.max(-i / i_max) if np.min(i) < 0 else 0.0
            worst = max(ratio_high, ratio_low, 1.0)
            saturated = worst > 1.0 + 1e-9
            if saturated:
                i = i / worst
            # Belt-and-braces clip for any negatives on paramagnetic rigs.
            i = np.clip(i, lo, i_max)
            return i, saturated
        else:  # FLAG
            clipped = np.clip(i, lo, i_max)
            saturated = bool(np.any(np.abs(clipped - i) > 1e-9))
            return clipped, saturated
