"""Tabbed field-control panel for the new field_synth model.

Adds a floating QDockWidget with three tabs:

  Field & Gradient
    - Uniform Bx / By / Bz spinboxes  (units: normalized [-1, 1] × 100 = %)
    - Gradient direction (dx / dy / dz) spinboxes + magnitude spinbox
    - Live readout: |B| and the six per-coil currents

  Rotation
    - Rotation axis (x, y, z) spinboxes
    - Roll frequency in Hz
    - Roll on/off checkbox

  Calibration
    - Six per-coil gain spinboxes (range 0..2, default 1.0)
    - Test button per coil that fires only that coil at a chosen strength
    - Save + Apply / Reload / Reset

Widgets write directly to MainWindow attributes (`uniform_B`, `gradient_dir`,
`gradient_mag`, `roll_axis`, `roll_freq`, `arduino1.coil_gains`). The
existing apply_actions() pipeline reads those attributes each frame and
handles the actual serial send. This dock is additive: the pre-existing
manual/joystick widgets keep working alongside it.
"""

from __future__ import annotations

import json
import numpy as np
from PyQt5 import QtCore, QtWidgets

from classes import field_synth


_COIL_LABELS = [
    "C1 top 0°",   "C2 top 120°",  "C3 top 240°",
    "C4 bot 0°",   "C5 bot 120°",  "C6 bot 240°",
]


def _spinbox(minimum, maximum, value, step=1.0, decimals=2, suffix=""):
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(float(minimum), float(maximum))
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(float(value))
    if suffix:
        box.setSuffix(" " + suffix)
    return box


class FieldControlsDock(QtWidgets.QDockWidget):
    """Add-on control panel. Constructor takes the MainWindow so widgets can
    write straight into its attributes without extra signal plumbing."""

    def __init__(self, main_window, calibration_path):
        super().__init__("Field Controls", main_window)
        self.main = main_window
        self.calibration_path = calibration_path

        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea |
            QtCore.Qt.RightDockWidgetArea |
            QtCore.Qt.TopDockWidgetArea |
            QtCore.Qt.BottomDockWidgetArea
        )
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._scroll(self._build_field_tab()),       "Field && Gradient")
        tabs.addTab(self._scroll(self._build_rotation_tab()),    "Rotation")
        tabs.addTab(self._scroll(self._build_calibration_tab()), "Calibration")

        wrapper = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(wrapper)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(tabs)
        self.setWidget(wrapper)

    def _scroll(self, page):
        """Wrap a tab page in a QScrollArea so overflow scrolls instead of
        clipping. Fixes the calibration tab getting cut off inside a
        capped-height dock."""
        area = QtWidgets.QScrollArea()
        area.setWidget(page)
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.NoFrame)
        return area

    # ----------------------------------------------------------- Field tab
    def _build_field_tab(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        # Uniform field group
        gb_uniform = QtWidgets.QGroupBox("Uniform field (B, normalized × 100 %)")
        grid = QtWidgets.QGridLayout(gb_uniform)
        self.bx = _spinbox(-100, 100, 0, suffix="%")
        self.by = _spinbox(-100, 100, 0, suffix="%")
        self.bz = _spinbox(-100, 100, 0, suffix="%")
        grid.addWidget(QtWidgets.QLabel("Bx:"), 0, 0); grid.addWidget(self.bx, 0, 1)
        grid.addWidget(QtWidgets.QLabel("By:"), 1, 0); grid.addWidget(self.by, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Bz:"), 2, 0); grid.addWidget(self.bz, 2, 1)
        for w in (self.bx, self.by, self.bz):
            w.valueChanged.connect(self._on_field_changed)
        v.addWidget(gb_uniform)

        # Gradient group
        gb_grad = QtWidgets.QGroupBox("Gradient (direction of |B| peak, magnitude)")
        gg = QtWidgets.QGridLayout(gb_grad)
        self.gdx = _spinbox(-1, 1, 0, step=0.1, decimals=3)
        self.gdy = _spinbox(-1, 1, 0, step=0.1, decimals=3)
        self.gdz = _spinbox(-1, 1, 1, step=0.1, decimals=3)
        self.gmag = _spinbox(0, 2, 0, step=0.05, decimals=3)
        gg.addWidget(QtWidgets.QLabel("dir X:"), 0, 0); gg.addWidget(self.gdx, 0, 1)
        gg.addWidget(QtWidgets.QLabel("dir Y:"), 1, 0); gg.addWidget(self.gdy, 1, 1)
        gg.addWidget(QtWidgets.QLabel("dir Z:"), 2, 0); gg.addWidget(self.gdz, 2, 1)
        gg.addWidget(QtWidgets.QLabel("magnitude:"), 3, 0); gg.addWidget(self.gmag, 3, 1)
        for w in (self.gdx, self.gdy, self.gdz, self.gmag):
            w.valueChanged.connect(self._on_field_changed)
        v.addWidget(gb_grad)

        # Live readout
        self.currents_label = QtWidgets.QLabel("Currents: (send zeros)")
        self.currents_label.setStyleSheet("font-family: monospace;")
        v.addWidget(self.currents_label)

        v.addStretch()
        return page

    def _on_field_changed(self, _value=None):
        self.main.uniform_B = np.array([
            self.bx.value() / 100.0,
            self.by.value() / 100.0,
            self.bz.value() / 100.0,
        ])
        self.main.gradient_dir = np.array([
            self.gdx.value(), self.gdy.value(), self.gdz.value()
        ])
        self.main.gradient_mag = float(self.gmag.value())

    # -------------------------------------------------------- Rotation tab
    def _build_rotation_tab(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        gb = QtWidgets.QGroupBox("Rotation axis (unit vector)")
        grid = QtWidgets.QGridLayout(gb)
        self.rx = _spinbox(-1, 1, 0, step=0.1, decimals=3)
        self.ry = _spinbox(-1, 1, 0, step=0.1, decimals=3)
        self.rz = _spinbox(-1, 1, 1, step=0.1, decimals=3)
        grid.addWidget(QtWidgets.QLabel("axis X:"), 0, 0); grid.addWidget(self.rx, 0, 1)
        grid.addWidget(QtWidgets.QLabel("axis Y:"), 1, 0); grid.addWidget(self.ry, 1, 1)
        grid.addWidget(QtWidgets.QLabel("axis Z:"), 2, 0); grid.addWidget(self.rz, 2, 1)
        for w in (self.rx, self.ry, self.rz):
            w.valueChanged.connect(self._on_rotation_changed)
        v.addWidget(gb)

        gb2 = QtWidgets.QGroupBox("Rotation frequency")
        h = QtWidgets.QHBoxLayout(gb2)
        self.freq = _spinbox(0, 100, 0, step=0.5, decimals=2, suffix="Hz")
        self.roll_on = QtWidgets.QCheckBox("Roll on")
        h.addWidget(QtWidgets.QLabel("freq:"))
        h.addWidget(self.freq)
        h.addWidget(self.roll_on)
        h.addStretch()
        self.freq.valueChanged.connect(self._on_rotation_changed)
        self.roll_on.toggled.connect(self._on_rotation_changed)
        v.addWidget(gb2)

        v.addStretch()
        return page

    def _on_rotation_changed(self, _value=None):
        self.main.roll_axis = np.array([
            self.rx.value(), self.ry.value(), self.rz.value()
        ])
        self.main.roll_freq = float(self.freq.value()) if self.roll_on.isChecked() else 0.0
        self.main.roll_on = self.roll_on.isChecked()

    # ---------------------------------------------------- Calibration tab
    def _build_calibration_tab(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        info = QtWidgets.QLabel(
            "Adjust each coil's gain until the magnetometer reads the target "
            "field for a fixed test drive. Save + Apply persists to disk and "
            "sets ArduinoHandler.coil_gains for all subsequent sends."
        )
        info.setWordWrap(True)
        v.addWidget(info)

        test_row = QtWidgets.QHBoxLayout()
        test_row.addWidget(QtWidgets.QLabel("Test drive strength:"))
        self.test_strength = _spinbox(0, 1, 0.3, step=0.05, decimals=2)
        test_row.addWidget(self.test_strength)
        stop = QtWidgets.QPushButton("Stop All")
        stop.clicked.connect(self._stop_all)
        test_row.addStretch()
        test_row.addWidget(stop)
        v.addLayout(test_row)

        self.gain_spinboxes = []
        self.channel_spinboxes = []
        self._active_coil_idx = None  # index of the coil currently under Test
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.addWidget(QtWidgets.QLabel("<b>Coil</b>"),    0, 0)
        grid.addWidget(QtWidgets.QLabel("<b>Gain</b>"),    0, 1)
        grid.addWidget(QtWidgets.QLabel("<b>Driver</b>"),  0, 2)
        for i, label in enumerate(_COIL_LABELS):
            grid.addWidget(QtWidgets.QLabel(label), i + 1, 0)
            # Gain spinbox: multiplies whatever the field synth commands for
            # this logical coil. Auto-saved on every change.
            spin = _spinbox(-2, 2, 1.0, step=0.05, decimals=3)
            spin.valueChanged.connect(lambda _, idx=i: self._on_gain_changed(idx))
            grid.addWidget(spin, i + 1, 1)
            self.gain_spinboxes.append(spin)
            # Driver index (0..5): which physical Arduino driver channel this
            # logical coil is wired to. Identity by default. Adjust when the
            # coil-to-driver wiring is swapped in hardware.
            drv = QtWidgets.QSpinBox()
            drv.setRange(0, 5)
            drv.setValue(i)
            drv.valueChanged.connect(lambda _v, idx=i: self._on_channel_changed(idx))
            grid.addWidget(drv, i + 1, 2)
            self.channel_spinboxes.append(drv)
            btn = QtWidgets.QPushButton("Test")
            btn.clicked.connect(lambda _, idx=i: self._test_coil(idx))
            grid.addWidget(btn, i + 1, 3)
        v.addLayout(grid)
        # Live update from the base strength spinbox too, and persist it as
        # part of the state we care about.
        self.test_strength.valueChanged.connect(self._on_strength_changed)

        btn_row = QtWidgets.QHBoxLayout()
        b_save = QtWidgets.QPushButton("Save + Apply")
        b_save.clicked.connect(self._save_and_apply)
        b_reload = QtWidgets.QPushButton("Reload")
        b_reload.clicked.connect(self._reload)
        b_reset = QtWidgets.QPushButton("Reset to 1.0")
        b_reset.clicked.connect(self._reset)
        btn_row.addWidget(b_save)
        btn_row.addWidget(b_reload)
        btn_row.addWidget(b_reset)
        btn_row.addStretch()
        v.addLayout(btn_row)

        v.addStretch()

        # Populate from currently-loaded gains and channel map on the arduino handler
        for spin, g in zip(self.gain_spinboxes, self.main.arduino1.coil_gains):
            spin.blockSignals(True)
            spin.setValue(float(g))
            spin.blockSignals(False)
        for spin, c in zip(self.channel_spinboxes, self.main.arduino1.channel_map):
            spin.blockSignals(True)
            spin.setValue(int(c))
            spin.blockSignals(False)
        return page

    def _fire_active_coil(self):
        """Send strength * gain on whichever coil is currently under Test.
        Called from Test click and from any live spinbox edit."""
        idx = self._active_coil_idx
        if idx is None:
            return
        strength = float(self.test_strength.value())
        gain = float(self.gain_spinboxes[idx].value())
        # Temporarily neutralize arduino1.coil_gains so we don't double-multiply
        # if the operator has already saved a set of gains earlier.
        saved = list(self.main.arduino1.coil_gains)
        self.main.arduino1.coil_gains = [1.0] * 6
        try:
            currents = [0.0] * 6
            currents[idx] = strength * gain
            self.main.arduino1.send(currents, 0.0)
        finally:
            self.main.arduino1.coil_gains = saved

    def _test_coil(self, idx):
        """Click handler for a Test button. Set the active coil, then fire."""
        self._active_coil_idx = idx
        self._fire_active_coil()

    def _on_gain_changed(self, idx):
        # Push the change to the arduino handler immediately so subsequent
        # normal-mode sends (from apply_actions) use it, and auto-save to
        # disk so a force-quit doesn't lose the calibration.
        self.main.arduino1.coil_gains = [float(s.value()) for s in self.gain_spinboxes]
        self._autosave()
        # Live-preview: if this coil is currently under Test, re-fire.
        if self._active_coil_idx == idx:
            self._fire_active_coil()

    def _on_channel_changed(self, _idx):
        # Update the arduino handler's channel_map. Reject non-permutations
        # so we don't send garbage; UI keeps whatever the operator typed.
        cmap = [int(s.value()) for s in self.channel_spinboxes]
        if sorted(cmap) == [0, 1, 2, 3, 4, 5]:
            self.main.arduino1.channel_map = cmap
            self._autosave()
        # If mid-Test, re-fire so the physical coil follows the new map.
        if self._active_coil_idx is not None:
            self._fire_active_coil()

    def _on_strength_changed(self, _value=None):
        # Base strength affects whichever coil is active.
        self._fire_active_coil()

    def _autosave(self):
        gains = [float(s.value()) for s in self.gain_spinboxes]
        cmap = [int(s.value()) for s in self.channel_spinboxes]
        if sorted(cmap) != [0, 1, 2, 3, 4, 5]:
            # Don't persist a broken map; wait for the operator to fix it.
            cmap = None
        try:
            field_synth.save_calibration(self.calibration_path, gains, cmap)
        except (OSError, ValueError):
            pass

    def _stop_all(self):
        self._active_coil_idx = None
        self.main.arduino1.send([0.0] * 6, 0.0)

    def _save_and_apply(self):
        # Kept for muscle memory. Auto-save on change already covers this.
        self._autosave()
        self.main.tbprint(f"Calibration saved to {self.calibration_path}")

    def _reload(self):
        gains = field_synth.load_gains(self.calibration_path)
        cmap = field_synth.load_channel_map(self.calibration_path)
        if gains is None and cmap is None:
            self.main.tbprint(f"No calibration file at {self.calibration_path}")
            return
        if gains is not None:
            for spin, g in zip(self.gain_spinboxes, gains):
                spin.blockSignals(True); spin.setValue(float(g)); spin.blockSignals(False)
            self.main.arduino1.coil_gains = gains
        if cmap is not None:
            for spin, c in zip(self.channel_spinboxes, cmap):
                spin.blockSignals(True); spin.setValue(int(c)); spin.blockSignals(False)
            self.main.arduino1.channel_map = cmap
        self.main.tbprint(f"Calibration reloaded from {self.calibration_path}")

    def _reset(self):
        for spin in self.gain_spinboxes:
            spin.blockSignals(True); spin.setValue(1.0); spin.blockSignals(False)
        for i, spin in enumerate(self.channel_spinboxes):
            spin.blockSignals(True); spin.setValue(i); spin.blockSignals(False)
        self.main.arduino1.coil_gains = [1.0] * 6
        self.main.arduino1.channel_map = [0, 1, 2, 3, 4, 5]
        self._autosave()
