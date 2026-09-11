"""Legacy entry point (pre-overhaul stack).

Kept as a rollback path. The new stack now lives in ``main.py`` and uses
``classes.gui.*`` with a ``config.yaml`` source of truth. This file is
the last surviving link to the original ``classes.gui_functions.MainWindow``
plus its per-OS ``uis/*.ui`` files.

Only use this if the new stack has a regression that blocks work on the rig.
"""

from PyQt5 import QtWidgets
import sys

from classes.gui_functions import MainWindow


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
