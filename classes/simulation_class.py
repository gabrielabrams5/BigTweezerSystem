"""3D field simulator widget.

Renders three things in real time inside the GUI's magnetic-field label:
  1. The six coil axis unit vectors in gray, tips labelled C1..C6.
  2. The commanded uniform B vector as a red arrow at the origin.
  3. The per-coil currents as a small bar chart below the 3D axes.

The class is still called `HelmholtzSimulator` for backward compatibility
with gui_functions.py's `self.simulator = HelmholtzSimulator(...)`
constructor -- Step 6 renames the reference. Legacy attributes like
`alpha`, `gamma`, `psi`, `freq`, `roll`, `omega` are accepted as writable
attributes so the old widget code paths don't crash on setattr; they're
just ignored by the draw logic.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QTimer

from classes import field_synth


_COIL_LABELS = ["C1", "C2", "C3", "C4", "C5", "C6"]


class HelmholtzSimulator(FigureCanvas):
    def __init__(self, parent=None, width=310, height=310, dpi=200):
        fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.subplots_adjust(top=1.0, bottom=0.28, left=0.0, right=1.0)
        self.ax = fig.add_subplot(2, 1, 1, projection="3d")
        self.bar_ax = fig.add_subplot(2, 1, 2)
        # Give the bar chart room and shrink the 3D plot
        fig.subplots_adjust(hspace=0.15)

        super().__init__(fig)
        self.setParent(parent)

        # Field-synth state -- the only inputs that actually matter.
        self.uniform_B = np.zeros(3)
        self.currents = np.zeros(6)

        # Legacy compat attributes. Old code writes to these; we ignore them
        # in the draw and just keep uniform_B/currents authoritative.
        self.Bx = 0.0
        self.By = 0.0
        self.Bz = 0.0
        self.alpha = 0.0
        self.gamma = 0.0
        self.psi = 0.0
        self.freq = 0.0
        self.omega = 0.0
        self.roll = False

        self._init_axes()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    # ------------------------------------------------------ public API

    def set_state(self, uniform_B, currents):
        """Called by apply_actions once per Arduino send."""
        self.uniform_B = np.asarray(uniform_B, dtype=float).reshape(3)
        self.currents = np.asarray(currents, dtype=float).reshape(6)

    def start(self):
        self.timer.start(75)  # ~13 Hz redraw

    def stop(self):
        self.timer.stop()
        self.zero()

    def zero(self):
        self.uniform_B = np.zeros(3)
        self.currents = np.zeros(6)
        self._draw()

    # ------------------------------------------------------ internals

    def _init_axes(self):
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_zlim(-1.1, 1.1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_xlabel("x", labelpad=-15)
        self.ax.set_ylabel("y", labelpad=-15)
        self.ax.set_zlabel("z", labelpad=-17)

    def _tick(self):
        # Fold legacy Bx/By/Bz writes into uniform_B on every tick. Once
        # Step 6 wires callers to set_state() directly this fallback is
        # dead code and gets removed.
        legacy = np.array([self.Bx, self.By, self.Bz])
        if np.any(legacy != 0.0):
            self.uniform_B = legacy
        self._draw()

    def _draw(self):
        # 3D field vector
        self.ax.clear()
        self._init_axes()

        # Draw the six coil axes as thin gray sticks
        for i in range(6):
            n = field_synth.COIL_AXES[:, i]
            self.ax.plot([0, n[0]], [0, n[1]], [0, n[2]],
                         color=(0.6, 0.6, 0.6), linewidth=0.7)
            self.ax.text(n[0] * 1.05, n[1] * 1.05, n[2] * 1.05,
                         _COIL_LABELS[i], fontsize=4, color=(0.4, 0.4, 0.4))

        # Draw commanded uniform B vector in red
        b = self.uniform_B
        if np.linalg.norm(b) > 1e-9:
            self.ax.quiver(0, 0, 0, b[0], b[1], b[2],
                           color="red", linewidth=1.2,
                           arrow_length_ratio=0.15)

        # Per-coil current bar chart
        self.bar_ax.clear()
        idx = np.arange(6)
        # Top ring in blue, bottom ring in orange
        colors = ["#3d7dd6", "#3d7dd6", "#3d7dd6",
                  "#e28c3d", "#e28c3d", "#e28c3d"]
        self.bar_ax.bar(idx, self.currents, color=colors, width=0.7)
        self.bar_ax.set_xticks(idx)
        self.bar_ax.set_xticklabels(_COIL_LABELS, fontsize=5)
        self.bar_ax.set_ylim(-1.1, 1.1)
        self.bar_ax.axhline(0, color="black", linewidth=0.4)
        self.bar_ax.tick_params(axis="y", labelsize=5)
        self.bar_ax.set_ylabel("PWM", fontsize=5, labelpad=1)

        self.draw()
