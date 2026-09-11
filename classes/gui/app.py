"""build_and_run — wires the new control stack and shows the window.

Used by both ``main_new.py`` (parallel entry point for parallel testing)
and, eventually, the shipped ``main.py``.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from classes.config_loader import get_config
from classes.field_solver import FieldSolver
from classes.gui.main_window import MainWindow
from classes.hal import BaseHAL, PrintHAL, build_from_config as build_hal_from_config
from classes.motion_controller import MotionController
from classes.supervisor import Supervisor


class InnerLoopRunner(QtCore.QObject):
    """Drives ``MotionController.tick`` at ``inner_hz_target`` Hz.

    Runs on the Qt event loop's own thread — the operator's expectation is
    that GUI edits take effect immediately, and separate threads for a
    ~200 Hz control loop are pure overhead for what is a light-touch
    problem. If Qt starves the tick, the Supervisor's watchdog catches it.
    """

    def __init__(self, motion: MotionController, hz: float = 200.0, parent=None):
        super().__init__(parent)
        self.motion = motion
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.hz = hz
        self.timer.setInterval(int(1000 / hz))
        self._last = None
        self.timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._last = time.monotonic()
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - (self._last if self._last is not None else now)
        self._last = now
        try:
            self.motion.tick(dt)
        except Exception as e:
            print(f"InnerLoopRunner: tick error: {e}")


class WatchdogRunner(QtCore.QObject):
    """Ticks the Supervisor watchdog on its own timer, independent of the
    inner loop, so a stalled MotionController still gets caught."""

    def __init__(self, supervisor: Supervisor, hz: float = 20.0, parent=None):
        super().__init__(parent)
        self.supervisor = supervisor
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / hz))
        self.timer.timeout.connect(supervisor.watchdog_tick)

    def start(self) -> None:
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()


def build_and_run(argv=None, force_print_hal: bool = False) -> int:
    """Build the whole stack, show the window, run the event loop.

    Return code is the QApplication exit code.
    """
    if argv is None:
        argv = sys.argv
    app = QtWidgets.QApplication(argv)

    config = get_config()
    solver = FieldSolver.from_config(config)
    hal: BaseHAL = (PrintHAL()
                    if force_print_hal
                    else build_hal_from_config(config))
    supervisor = Supervisor.from_config(config, hal)
    motion = MotionController.from_config(config, solver)

    # Rebuilding the solver when calibration or solver knobs change is a
    # cheap allocation; we just re-run from_config. MotionController holds
    # a reference to solver so we swap it in place.
    def rebuild_solver() -> None:
        new_solver = FieldSolver.from_config(config)
        motion.solver = new_solver

    window = MainWindow(config, motion, supervisor, hal, rebuild_solver)
    window.show()

    inner = InnerLoopRunner(motion, hz=float(config.get("loop.inner_hz_target", 200)))
    watchdog = WatchdogRunner(supervisor, hz=20.0)
    inner.start()
    watchdog.start()

    return app.exec_()
