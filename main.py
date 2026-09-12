"""BigTweezerSystem entry point (post-overhaul).

Runs the new control stack:
  * ``classes.config_loader``    — YAML config as single source of truth
  * ``classes.field_solver``     — pure-math solver (paramagnetic closed
                                   form; DLS reserved for signed-drive rigs)
  * ``classes.motion_controller``— Mode A (rotating) / Mode B (static)
                                   with phase accumulator
  * ``classes.supervisor``       — watchdog, current budget, I²R estimate,
                                   e-stop with authority over the HAL
  * ``classes.hal``              — HAL over ArduinoHandler; auto-discovers
                                   the serial port from ``serial.port_glob``
  * ``classes.gui.*``            — pure-Python PyQt5, no Qt Designer .ui

If the new stack has a regression that blocks work, ``main_legacy.py``
still runs the old ``classes.gui_functions.MainWindow`` verbatim.

Usage:
    python3 main.py               # tries to open serial, falls back to log
    python3 main.py --print-only  # no serial; PrintHAL logs packets
"""

from __future__ import annotations

import sys

from classes.gui.app import build_and_run


def main() -> int:
    force_print = "--print-only" in sys.argv
    return build_and_run(argv=sys.argv, force_print_hal=force_print)


if __name__ == "__main__":
    sys.exit(main())
