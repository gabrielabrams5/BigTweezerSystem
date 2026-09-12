"""Pure-Python PyQt5 GUI for the new control stack. No Qt Designer .ui files.

Entry point: :func:`app.build_and_run`. Structure:

  * :mod:`classes.gui.widgets` — small reusable helpers (LabeledSpinBox etc).
  * :mod:`classes.gui.panels`  — Mode A, Mode B, Solver, Supervisor panels.
  * :mod:`classes.gui.calibration_wizard` — guided Bmap/Gmap sweep wizard.
  * :mod:`classes.gui.main_window` — the QMainWindow that hosts everything.
  * :mod:`classes.gui.app`         — build_and_run() that wires the stack.
"""
