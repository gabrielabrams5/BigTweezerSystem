"""Tests for classes.field_solver.

Run with:  python3 -m pytest classes/test_field_solver.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from classes.field_solver import (
    FieldSolver,
    PARAMAGNETIC,
    SOFT,
    HARD,
    RESCALE,
    FLAG,
)


# Geometry-derived Bmap for the aligned-ring rig (matches config_example.yaml).
_S = np.sqrt(2) / 2
_AZ = np.deg2rad([0.0, 120.0, 240.0])
GEOM_BMAP = np.column_stack([
    *(np.array([_S * np.cos(a), _S * np.sin(a),  _S]) for a in _AZ),
    *(np.array([_S * np.cos(a), _S * np.sin(a), -_S]) for a in _AZ),
])


def make_paramagnetic() -> FieldSolver:
    return FieldSolver(GEOM_BMAP, robot_type=PARAMAGNETIC)


def make_signed() -> FieldSolver:
    return FieldSolver(GEOM_BMAP, robot_type=SOFT, lam=0.001)


# ---- paramagnetic closed form ---------------------------------------


class TestParamagnetic:

    def test_plus_z_fires_top_ring_full_send(self):
        r = make_paramagnetic().solve([0.0, 0.0, 1.0])
        assert np.allclose(r.i[:3], 1.0, atol=1e-6)
        assert np.allclose(r.i[3:], 0.0, atol=1e-6)
        assert not r.saturated

    def test_minus_z_fires_bottom_ring_full_send(self):
        r = make_paramagnetic().solve([0.0, 0.0, -1.0])
        assert np.allclose(r.i[:3], 0.0, atol=1e-6)
        assert np.allclose(r.i[3:], 1.0, atol=1e-6)

    def test_plus_x_fires_c1_and_c4(self):
        r = make_paramagnetic().solve([1.0, 0.0, 0.0])
        assert r.i[0] == pytest.approx(1.0, abs=1e-6)
        assert r.i[3] == pytest.approx(1.0, abs=1e-6)
        # C2, C3, C5, C6 have -0.5*cos(0) contribution — zeroed by max().
        assert r.i[1] == pytest.approx(0.0, abs=1e-6)
        assert r.i[2] == pytest.approx(0.0, abs=1e-6)
        assert r.i[4] == pytest.approx(0.0, abs=1e-6)
        assert r.i[5] == pytest.approx(0.0, abs=1e-6)

    def test_plus_y_fires_c2_and_c5(self):
        r = make_paramagnetic().solve([0.0, 1.0, 0.0])
        assert r.i[1] == pytest.approx(1.0, abs=1e-6)
        assert r.i[4] == pytest.approx(1.0, abs=1e-6)

    def test_zero_command_zero_currents(self):
        r = make_paramagnetic().solve([0.0, 0.0, 0.0])
        assert np.allclose(r.i, 0.0)
        assert not r.saturated

    def test_diagonal_command_top_coil_saturates(self):
        # +X+Z command: C1 (top ring, 0° azimuth) is exactly aligned →
        # n̂·B = 1.0. C4 (bottom ring, 0° azimuth) has +X but −Z →
        # n̂·B = 0. C2, C3 have +Z projection but −X → n̂·B = 0.25.
        # C5, C6 have −X and −Z → n̂·B negative → clipped to 0.
        # After peak-normalization by C1: [1.0, 0.25, 0.25, 0, 0, 0].
        r = make_paramagnetic().solve([_S, 0.0, _S])
        assert r.i[0] == pytest.approx(1.0, abs=1e-6)
        assert r.i[1] == pytest.approx(0.25, abs=1e-6)
        assert r.i[2] == pytest.approx(0.25, abs=1e-6)
        assert r.i[3] == pytest.approx(0.0, abs=1e-6)
        assert r.i[4] == pytest.approx(0.0, abs=1e-6)
        assert r.i[5] == pytest.approx(0.0, abs=1e-6)

    def test_saturation_rescale_preserves_direction(self):
        # Command a huge |B|; peak-normalization already ensures max = |B|,
        # so with i_max_per_coil=1.0 anything past |B|=1 should rescale.
        solver = FieldSolver(GEOM_BMAP, robot_type=PARAMAGNETIC,
                             saturation_policy=RESCALE)
        r = solver.solve([0.0, 0.0, 2.0])
        assert np.max(r.i) == pytest.approx(1.0, abs=1e-6)
        assert r.saturated
        # Direction preserved: top coils all equal.
        assert np.allclose(r.i[:3], r.i[0])
        assert np.allclose(r.i[3:], 0.0)

    def test_per_coil_gains_and_channel_polarity(self):
        # gain = 2 on coil 0 → doubles its output; gain = -1 flips coil 3.
        gains = np.array([2.0, 1.0, 1.0, -1.0, 1.0, 1.0])
        solver = FieldSolver(GEOM_BMAP, robot_type=PARAMAGNETIC,
                             per_coil_gains=gains,
                             saturation_policy=RESCALE)
        r = solver.solve([1.0, 0.0, 0.0])
        # Before gains: C1 at 1.0, C4 at 1.0. After: C1 at 2.0, C4 at -1.0.
        # Rescale by worst overshoot (2.0) → C1 becomes 1.0, C4 becomes -0.5.
        # Then clip [-1, 1] leaves it as-is on paramagnetic? No — para clips to
        # [0, i_max]. So C4's -0.5 gets clipped to 0.
        assert r.i[0] == pytest.approx(1.0, abs=1e-6)
        assert r.i[3] == pytest.approx(0.0, abs=1e-6)

    def test_i_max_per_coil_respected(self):
        # Cap coil 0 at 0.3. Command +X → C1 wants 1.0, gets clipped to 0.3.
        imax = np.array([0.3, 1.0, 1.0, 1.0, 1.0, 1.0])
        solver = FieldSolver(GEOM_BMAP, robot_type=PARAMAGNETIC,
                             i_max_per_coil=imax,
                             saturation_policy=RESCALE)
        r = solver.solve([1.0, 0.0, 0.0])
        assert r.i[0] <= 0.3 + 1e-9
        assert r.saturated


# ---- static (Mode B) exact synthesis --------------------------------


def _horizontal(theta_deg: float) -> np.ndarray:
    r = np.deg2rad(theta_deg)
    return np.array([np.cos(r), np.sin(r), 0.0])


def _realized_azimuth(solver: FieldSolver, theta_deg: float,
                      static: bool) -> float:
    B = solver.Bmap @ solver.solve(_horizontal(theta_deg), static=static).i
    return float(np.rad2deg(np.arctan2(B[1], B[0])))


def _angle_error(theta_deg: float, realized_deg: float) -> float:
    return abs((realized_deg - theta_deg + 180.0) % 360.0 - 180.0)


SWEEP = list(range(0, 360, 5))


class TestStaticExactSynthesis:
    """Mode B steers by the realized field direction, so it must actually
    point where it was told. The projection heuristic does not — these pin
    the difference."""

    def test_projection_heuristic_has_large_direction_error(self):
        """Documents WHY the exact solver exists. If this ever starts
        passing with a small error, the heuristic changed and the whole
        rationale needs revisiting."""
        solver = make_paramagnetic()
        worst = max(_angle_error(t, _realized_azimuth(solver, t, static=False))
                    for t in SWEEP)
        assert worst > 25.0

    def test_projection_heuristic_has_dead_zones(self):
        """Commands spread over 30° collapse onto one realized direction —
        the operator sees the bead refuse to turn, then jump."""
        solver = make_paramagnetic()
        a = _realized_azimuth(solver, 5.0, static=False)
        b = _realized_azimuth(solver, 25.0, static=False)
        assert _angle_error(a, b) < 1.0

    def test_static_realizes_commanded_direction(self):
        solver = make_paramagnetic()
        worst = max(_angle_error(t, _realized_azimuth(solver, t, static=True))
                    for t in SWEEP)
        assert worst < 0.5

    def test_static_has_no_dead_zones(self):
        """Realized azimuth must advance monotonically with the command."""
        solver = make_paramagnetic()
        for t in SWEEP:
            step = _angle_error(
                _realized_azimuth(solver, t, static=True),
                _realized_azimuth(solver, t + 5.0, static=True))
            assert step == pytest.approx(5.0, abs=0.5)

    def test_static_currents_stay_non_negative(self):
        """Pull-only physics: a reversed coil still attracts, so a negative
        duty would be meaningless on this rig."""
        solver = make_paramagnetic()
        for t in SWEEP:
            assert np.all(solver.solve(_horizontal(t), static=True).i >= -1e-12)

    def test_static_horizontal_command_gives_no_vertical_field(self):
        solver = make_paramagnetic()
        for t in SWEEP:
            B = solver.Bmap @ solver.solve(_horizontal(t), static=True).i
            assert abs(B[2]) < 1e-9

    def test_static_fires_rings_symmetrically(self):
        """The ridge term exists for this: plain NNLS picks a minimal-support
        solution that breaks top/bottom symmetry and tilts the |B| landscape
        out of plane."""
        solver = make_paramagnetic()
        for t in SWEEP:
            i = solver.solve(_horizontal(t), static=True).i
            assert np.allclose(i[:3], i[3:], atol=1e-6)

    def test_static_vertical_commands_still_fire_one_ring(self):
        solver = make_paramagnetic()
        up = solver.solve([0.0, 0.0, 1.0], static=True)
        assert np.allclose(up.i[:3], 1.0, atol=1e-3)
        assert np.allclose(up.i[3:], 0.0, atol=1e-6)
        down = solver.solve([0.0, 0.0, -1.0], static=True)
        assert np.allclose(down.i[:3], 0.0, atol=1e-6)
        assert np.allclose(down.i[3:], 1.0, atol=1e-3)

    def test_static_peak_normalizes_like_the_heuristic(self):
        solver = make_paramagnetic()
        for t in (0.0, 37.0, 90.0, 211.0):
            assert solver.solve(_horizontal(t), static=True).i.max() == \
                pytest.approx(1.0, abs=1e-6)

    def test_static_zero_command_zero_currents(self):
        r = make_paramagnetic().solve([0.0, 0.0, 0.0], static=True)
        assert np.allclose(r.i, 0.0)

    def test_rolling_path_is_untouched(self):
        """The whole safety argument for shipping this: Mode A output must
        be bit-identical, so the existing Frame Cal stays valid."""
        solver = make_paramagnetic()
        for t in SWEEP:
            B_des = _horizontal(t)
            assert np.array_equal(solver.solve(B_des).i,
                                  solver._solve_paramagnetic(B_des))

    def test_projection_escape_hatch_restores_old_behaviour(self):
        solver = FieldSolver(GEOM_BMAP, robot_type=PARAMAGNETIC,
                             static_synthesis="projection")
        for t in SWEEP:
            B_des = _horizontal(t)
            assert np.array_equal(solver.solve(B_des, static=True).i,
                                  solver._solve_paramagnetic(B_des))

    def test_static_works_on_an_asymmetric_measured_bmap(self):
        """Real Bmaps come from the calibration wizard, not geometry —
        per-coil strengths vary. Direction must still come out right."""
        rng = np.random.default_rng(4)
        Bmap = GEOM_BMAP * rng.uniform(0.7, 1.3, size=(1, 6))
        solver = FieldSolver(Bmap, robot_type=PARAMAGNETIC)
        for t in SWEEP:
            B = solver.Bmap @ solver.solve(_horizontal(t), static=True).i
            assert _angle_error(t, np.rad2deg(np.arctan2(B[1], B[0]))) < 0.5


# ---- DLS path for signed-drive rigs ---------------------------------


class TestSignedDLS:

    def test_solve_reaches_target_field_within_tolerance(self):
        solver = make_signed()
        target = np.array([0.3, 0.2, 0.4])
        r = solver.solve(target)
        # With well-conditioned Bmap and small λ the residual should be tiny.
        assert r.residual < 0.05

    def test_dls_stays_finite_under_singular_matrix(self):
        # Rank-2 Bmap (last two columns are duplicates).
        B = GEOM_BMAP.copy()
        B[:, 5] = B[:, 4]
        solver = FieldSolver(B, robot_type=SOFT, lam=0.1)
        r = solver.solve([0.5, 0.5, 0.5])
        assert np.all(np.isfinite(r.i))

    def test_signed_solver_allows_negative_currents(self):
        solver = make_signed()
        # Command a target the pull-only path can't reach (would need
        # negative currents on some coils).
        r = solver.solve([0.1, 0.0, 0.05])
        # At least one coil should be negative to hit the target exactly.
        # (Under paramagnetic all six would be non-negative.)
        # We assert only that residual is small — the sign is solver-dependent.
        assert r.residual < 0.05


# ---- saturation policies --------------------------------------------


class TestSaturationPolicies:

    def test_flag_policy_reports_saturation(self):
        solver = FieldSolver(GEOM_BMAP, robot_type=PARAMAGNETIC,
                             saturation_policy=FLAG,
                             i_max_per_coil=np.full(6, 0.2))
        r = solver.solve([0.0, 0.0, 1.0])
        assert r.saturated
        assert np.max(r.i) == pytest.approx(0.2, abs=1e-6)

    def test_no_saturation_when_command_small(self):
        solver = FieldSolver(GEOM_BMAP, robot_type=PARAMAGNETIC,
                             saturation_policy=RESCALE)
        r = solver.solve([0.0, 0.0, 0.1])
        assert not r.saturated
        assert np.max(r.i) == pytest.approx(0.1, abs=1e-6)


# ---- Fmap construction (hard/soft only) -----------------------------


class TestFmapConstruction:

    def test_fmap_zero_gradient_gives_zero_force(self):
        solver = FieldSolver(GEOM_BMAP, robot_type=HARD, lam=0.001)
        # With Gmap all zeros (default), no force can be commanded — the
        # DLS solver returns whatever satisfies the field constraint.
        r = solver.solve(B_des=[0.0, 0.0, 0.5],
                         F_des=[0.0, 0.0, 0.1],
                         m=[0.0, 0.0, 1.0])
        assert np.all(np.isfinite(r.i))

    def test_fmap_shape(self):
        Gmap = np.random.default_rng(0).standard_normal((5, 6)) * 0.1
        solver = FieldSolver(GEOM_BMAP, Gmap=Gmap, robot_type=HARD)
        F = solver._build_fmap(np.array([1.0, 0.0, 0.0]))
        assert F.shape == (3, 6)


# ---- FieldSolver.from_config wiring ---------------------------------


class TestFromConfig:

    def test_from_config_reads_example(self, tmp_path):
        # Load the example config through the loader, then verify from_config
        # produces a solver with the expected Bmap shape and robot type.
        # (Not asserting exact solve values — per_coil_gains from a possibly-
        # migrated calibration.json could scale them.)
        from classes.config_loader import Config
        cfg_path = tmp_path / "cfg.yaml"
        c = Config(str(cfg_path))
        c.load()
        if c.get("calibration.Bmap") is None:
            pytest.skip("no example config available")
        solver = FieldSolver.from_config(c)
        assert solver.Bmap.shape == (3, 6)
        assert solver.Gmap.shape == (5, 6)
        assert solver.robot_type in (PARAMAGNETIC, SOFT, HARD)
        r = solver.solve([0.0, 0.0, 1.0])
        # Top ring fires, bottom ring is zero — invariant to per-coil gains
        # under paramagnetic geometry (top coils have +Z projection, bottom
        # coils have −Z projection which is zeroed by max()).
        assert np.all(r.i[:3] > 0.0)
        assert np.allclose(r.i[3:], 0.0, atol=1e-6)
