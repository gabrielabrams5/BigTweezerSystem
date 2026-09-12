"""supervisor — safety authority. Has veto over every send.

Watches three things:

  * **Watchdog** — if the MotionController hasn't produced a fresh
    ``SolveResult`` within ``limits.watchdog_ms``, zero the coils.

  * **Power budget** — accumulate ``I²R`` per coil in a rolling window
    (stand-in for real thermistors). If the estimate crosses
    ``limits.power_budget_w * limits.duty_cycle_max`` for any coil, cut
    that coil.

  * **E-stop** — a latch triggered by the GUI button or a keyboard
    shortcut. Once tripped, all commands are ignored until the operator
    presses "Reset E-stop."

The Supervisor sits *between* the MotionController and the HAL: solved
currents pass through ``check_and_forward()``, which either sends them or
substitutes zeros. That way there is no path from the outer loop to the
coils that bypasses supervision.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from classes.field_solver import SolveResult
from classes.hal import BaseHAL


class Supervisor:

    def __init__(self,
                 hal: BaseHAL,
                 watchdog_ms: float = 500.0,
                 i_total_max: float = 6.0,
                 power_budget_w: float = 60.0,
                 duty_cycle_max: float = 0.9,
                 coil_resistance_ohm: float = 4.0,
                 printer=print):
        self.hal = hal
        self.watchdog_ms = float(watchdog_ms)
        self.i_total_max = float(i_total_max)
        self.power_budget_w = float(power_budget_w)
        self.duty_cycle_max = float(duty_cycle_max)
        self.R = float(coil_resistance_ohm)
        self.printer = printer

        self.estopped = False
        # Rolling I²R integrator per coil; decays with a fixed time
        # constant so brief spikes are tolerated.
        self.power_estimate = np.zeros(6)
        self.power_tau_s = 5.0  # seconds
        self._last_check_t: Optional[float] = None
        self._last_solve_t: Optional[float] = None

        # Latched cut-offs per coil (set when a coil's estimate crosses the
        # budget; cleared when it drops back under).
        self.coil_cut = np.zeros(6, dtype=bool)

        # When set, the Calibration wizard owns the HAL. Supervisor stops
        # writing (its Mode.OFF zeros would clobber wizard packets at 200 Hz)
        # and stops watchdog-tripping. E-stop remains authoritative.
        self.calibration_active = False

    @classmethod
    def from_config(cls, config, hal: BaseHAL, printer=print) -> "Supervisor":
        return cls(
            hal=hal,
            watchdog_ms=float(config.get("limits.watchdog_ms", 500)),
            i_total_max=float(config.get("limits.i_total_max", 6.0)),
            power_budget_w=float(config.get("limits.power_budget_w", 60.0)),
            duty_cycle_max=float(config.get("limits.duty_cycle_max", 0.9)),
            coil_resistance_ohm=float(config.get("limits.coil_resistance_ohm", 4.0)),
            printer=printer,
        )

    # ---- e-stop -----------------------------------------------------

    def estop(self) -> None:
        if not self.estopped:
            self.printer("Supervisor: E-STOP tripped — coils forced to zero")
        self.estopped = True
        self.hal.estop()

    def reset_estop(self) -> None:
        self.estopped = False
        if hasattr(self.hal, "reset_estop"):
            self.hal.reset_estop()
        self.power_estimate[:] = 0.0
        self.coil_cut[:] = False
        self.printer("Supervisor: E-STOP cleared")

    # ---- main gate --------------------------------------------------

    def check_and_forward(self, result: SolveResult, acoustic_freq: float = 0.0) -> np.ndarray:
        """Vet solved currents, apply cut-offs, forward to HAL. Returns the
        actually-sent 6-vector so listeners (GUI live bars) can display it.
        """
        now = time.monotonic()
        self._last_solve_t = now

        # 1. E-stop is absolute.
        if self.estopped:
            zero = np.zeros(6)
            self.hal.set_currents(zero, acoustic_freq)
            return zero

        # 2. Calibration wizard owns the HAL — do not write, and do not
        #    fall through to power accounting on stale zeros.
        if self.calibration_active:
            return np.zeros(6)

        i = np.asarray(result.i, dtype=float).reshape(6).copy()

        # 2. Total-current budget: rescale to preserve direction if over.
        total = float(np.sum(np.abs(i)))
        if total > self.i_total_max and total > 1e-9:
            i = i * (self.i_total_max / total)

        # 3. Per-coil power integrator update.
        if self._last_check_t is not None:
            dt = max(0.0, now - self._last_check_t)
            decay = np.exp(-dt / self.power_tau_s)
            # Instantaneous per-coil power = I² · R (duty is [0,1] treated
            # as fraction of the driver's max current; R absorbs the rest
            # of the scaling).
            inst = (i ** 2) * self.R
            self.power_estimate = self.power_estimate * decay + inst * (1.0 - decay)
        self._last_check_t = now

        # 4. Coil cut-off latches.
        budget = self.power_budget_w * self.duty_cycle_max
        for c in range(6):
            if self.power_estimate[c] > budget and not self.coil_cut[c]:
                self.coil_cut[c] = True
                self.printer(
                    f"Supervisor: coil C{c + 1} cut — power estimate "
                    f"{self.power_estimate[c]:.2f} W > budget {budget:.2f} W"
                )
            elif self.coil_cut[c] and self.power_estimate[c] < 0.7 * budget:
                self.coil_cut[c] = False
                self.printer(f"Supervisor: coil C{c + 1} re-armed")

        i[self.coil_cut] = 0.0

        self.hal.set_currents(i, acoustic_freq)
        return i

    # ---- watchdog ---------------------------------------------------

    def watchdog_tick(self) -> None:
        """Called by an independent QTimer (or a thread) at ~100 Hz. If
        the MotionController has gone silent, zero the coils. Independent
        cadence keeps this alive even if the inner loop hangs.
        """
        if self.estopped:
            self.hal.set_currents(np.zeros(6), 0.0)
            return
        if self.calibration_active:
            return
        if self._last_solve_t is None:
            return
        elapsed_ms = (time.monotonic() - self._last_solve_t) * 1000.0
        if elapsed_ms > self.watchdog_ms:
            self.printer(
                f"Supervisor: watchdog trip — {elapsed_ms:.0f} ms since last "
                f"solve; zeroing coils"
            )
            self.hal.set_currents(np.zeros(6), 0.0)

    # ---- readout ----------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "estopped": self.estopped,
            "power_estimate_w": self.power_estimate.tolist(),
            "coil_cut": self.coil_cut.tolist(),
        }
