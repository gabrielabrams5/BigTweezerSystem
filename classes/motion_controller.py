"""motion_controller — mode state machine + inner-loop tick.

Two active modes (paramagnetic reduces the spec's three to two):

  * ``Mode A — Rolling pull``: the base direction (shared with Mode B)
    precesses around ``roll_axis`` at ``freq`` Hz. The commanded B vector
    is ``B_des(t) = |B| · R(roll_axis, 2πft) @ direction``. Under
    paramagnetic physics the strongest coil identity rotates around the
    axis each tick, dragging the bead. Matches legacy
    ``classes/field_synth.py::synthesize()``.

  * ``Mode B — Static pull``: the commanded direction is fixed.
    ``B_des = |B| · direction``.

Command sources (arbitrated by priority, last-writer-wins within a class):

  1. Supervisor override — sets B_des = 0 during watchdog trip or e-stop.
  2. Algorithm / tracker — closed-loop from position PID.
  3. Joystick — operator manual.
  4. GUI panels — direct spinbox entry.

The controller's ``tick(dt)`` runs the phase accumulator, invokes the
solver, and emits a solved 6-vector to whoever's listening (usually the
Supervisor followed by the HAL).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

import numpy as np

from classes.field_solver import FieldSolver, SolveResult


class Mode(Enum):
    A_ROTATING = "A"
    B_STATIC = "B"
    OFF = "OFF"


def roll_axis_for(tx: float, ty: float, min_norm: float = 1e-9) -> list:
    """Horizontal roll axis a = ẑ × t̂ for rolling travel along (tx, ty).

    A field rotating about this axis sweeps the vertical plane containing
    the travel vector, so the bead rolls along the substrate toward
    (tx, ty). Returns [0, 0, 0] when the horizontal component is below
    ``min_norm`` — a zero axis makes the Rodrigues rotation collapse to
    identity, i.e. a plain static pull along ``direction``.
    """
    h = float(np.hypot(tx, ty))
    if h < min_norm:
        return [0.0, 0.0, 0.0]
    return [-ty / h, tx / h, 0.0]


def _rotation_matrix(axis, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation. Zero-length axis returns identity."""
    ax = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(ax))
    if n < 1e-12:
        return np.eye(3)
    u = ax / n
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    ux, uy, uz = u
    K = np.array([
        [0.0, -uz,  uy],
        [ uz, 0.0, -ux],
        [-uy,  ux, 0.0],
    ])
    return c * np.eye(3) + s * K + (1.0 - c) * np.outer(u, u)


class Source(Enum):
    SUPERVISOR = 0
    ALGORITHM = 1
    JOYSTICK = 2
    GUI = 3


class MotionController:
    """Owns mode state + phase accumulator. Not a QObject — Qt signal wiring
    happens at the GUI boundary. Keeping this framework-free makes it
    unit-testable and lets a headless main loop drive it.
    """

    def __init__(self,
                 solver: FieldSolver,
                 acoustic_freq: float = 0.0):
        self.solver = solver
        self.mode = Mode.OFF
        self.source = Source.GUI  # who set the current command

        # Shared state (both modes use `direction` as the base B direction).
        self.magnitude = 0.5
        self.direction = np.array([0.0, 0.0, 1.0])

        # Mode A state (rolling)
        self.roll_axis = np.array([0.0, 0.0, 1.0])
        self.freq_hz = 1.0
        self.freq_target_hz = 1.0
        self.freq_ramp_rate = 0.5  # Hz/s
        self.phase = 0.0

        # Acoustic pass-through (unchanged by field synthesis)
        self.acoustic_freq = float(acoustic_freq)

        # Listeners for solved currents
        self._listeners: list[Callable[[SolveResult, float], None]] = []

        # Watchdog: latest tick time (monotonic seconds).
        self._last_tick = 0.0

    @classmethod
    def from_config(cls, config, solver: FieldSolver) -> "MotionController":
        mc = cls(solver, acoustic_freq=float(config.get("acoustic.freq_hz", 0.0)))
        mc.magnitude = float(config.get("modes.mode_a.magnitude_default", 0.5))
        mc.freq_hz = float(config.get("modes.mode_a.freq_default", 1.0))
        mc.freq_target_hz = mc.freq_hz
        mc.freq_ramp_rate = float(config.get("modes.mode_a.freq_ramp_rate", 0.5))
        mc.roll_axis = np.asarray(
            config.get("modes.mode_a.roll_axis_default", [0.0, 0.0, 1.0]),
            dtype=float,
        )
        mc.direction = np.asarray(
            config.get("modes.mode_b.direction_default", [0.0, 0.0, 1.0]),
            dtype=float,
        )
        return mc

    # ---- command inputs -------------------------------------------

    def set_mode(self, mode: Mode) -> None:
        if mode != self.mode:
            self.phase = 0.0
        self.mode = mode

    def set_static_direction(self, direction, source: Source = Source.GUI) -> None:
        d = np.asarray(direction, dtype=float).reshape(3)
        n = float(np.linalg.norm(d))
        self.direction = d / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
        self.source = source

    def set_magnitude(self, mag: float, source: Source = Source.GUI) -> None:
        self.magnitude = float(np.clip(mag, 0.0, 1.0))
        self.source = source

    def set_roll_axis(self, axis, source: Source = Source.GUI) -> None:
        a = np.asarray(axis, dtype=float).reshape(3)
        n = float(np.linalg.norm(a))
        # Zero-length axis means "no rotation" — keep as-is; the Rodrigues
        # helper returns identity so B_des collapses to |B|·direction.
        self.roll_axis = a / n if n > 1e-12 else a
        self.source = source

    def set_frequency(self, freq_hz: float, source: Source = Source.GUI) -> None:
        # Ramp toward the target rather than jumping; jumps drop the bead
        # out of sync when it's rolling.
        self.freq_target_hz = float(freq_hz)
        self.source = source

    def set_acoustic(self, acoustic_freq: float) -> None:
        self.acoustic_freq = float(acoustic_freq)

    # ---- inner-loop tick ------------------------------------------

    def tick(self, dt: float) -> Optional[SolveResult]:
        """Advance state by ``dt`` seconds; solve; emit to listeners.

        Returns the SolveResult so the outer scheduler can log or forward
        it. Under Mode.OFF, returns a zero-currents result immediately.
        """
        self._last_tick += dt

        if self.mode == Mode.OFF:
            zero = SolveResult(i=np.zeros(6), saturated=False, residual=0.0)
            self._emit(zero)
            return zero

        # Ramp frequency toward target — no abrupt jumps.
        if abs(self.freq_target_hz - self.freq_hz) > 1e-9:
            step = self.freq_ramp_rate * dt
            delta = self.freq_target_hz - self.freq_hz
            if abs(delta) <= step:
                self.freq_hz = self.freq_target_hz
            else:
                self.freq_hz += step if delta > 0 else -step

        if self.mode == Mode.A_ROTATING:
            self.phase += 2.0 * np.pi * self.freq_hz * dt
            # Keep phase bounded (either sign — frequency may be negative)
            # so accumulated float error doesn't eventually eat the
            # precision of cos()/sin().
            if abs(self.phase) > 2.0 * np.pi * 1e6:
                self.phase -= 2.0 * np.pi * round(self.phase / (2.0 * np.pi))
            R = _rotation_matrix(self.roll_axis, self.phase)
            B_des = self.magnitude * (R @ self.direction)
        elif self.mode == Mode.B_STATIC:
            B_des = self.magnitude * self.direction
        else:
            B_des = np.zeros(3)

        # Mode B steers by the *realized* field direction, so it needs the
        # solver's exact non-negative synthesis rather than the projection
        # heuristic (which is off by up to 30°, with dead zones). Mode A
        # precesses through a full cycle and is calibrated end-to-end, so it
        # keeps the historical rule unchanged.
        result = self.solver.solve(B_des, static=(self.mode == Mode.B_STATIC))
        self._emit(result)
        return result

    # ---- listeners -------------------------------------------------

    def on_solved(self, callback: Callable[[SolveResult, float], None]) -> None:
        """Register a callback ``cb(result, acoustic_freq)`` fired every tick."""
        self._listeners.append(callback)

    def _emit(self, result: SolveResult) -> None:
        for cb in self._listeners:
            try:
                cb(result, self.acoustic_freq)
            except Exception:
                # Never let a listener crash break the inner loop.
                pass
