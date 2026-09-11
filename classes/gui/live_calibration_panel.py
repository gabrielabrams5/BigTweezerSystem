"""Legacy-style per-coil live-fire calibration panel.

Ports ``classes/field_tabs.py::_build_calibration_tab`` (from the legacy
stack) onto the new config + HAL + Supervisor plumbing. Complements the
Hall-probe Wizard: Wizard measures Bmap columns end-to-end; this panel is
the muscle-memory workflow — click Test on C3, watch the meter or the
bead, dial the gain up/down in 0.05 steps while it's firing.

Per-coil edits flow to ``config.yaml → calibration.per_coil_gains`` and
``calibration.channel_map``. Gain changes trigger ``rebuild_solver_cb`` so
the running solver picks them up on the next tick. Channel-map changes
are pushed directly into the HAL so the next packet routes correctly.

While a coil is under Test, ``supervisor.calibration_active = True`` — same
pattern as the wizard — so the 200 Hz inner loop's zeros don't overwrite
this panel's 200 ms heartbeat.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets


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


class LiveCalibrationPanel(QtWidgets.QWidget):
    """Per-coil live-fire tuning + channel-map editor.

    Signals
    -------
    gainsChanged() — emitted after a gain (or channel_map) edit lands in
        config. MainWindow wires this to ``rebuild_solver_cb`` so the
        solver picks up new gains on the next tick.
    """

    gainsChanged = QtCore.pyqtSignal()

    def __init__(self,
                 hal,
                 config,
                 supervisor=None,
                 log_fn: Callable[[str], None] = print,
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.hal = hal
        self.config = config
        self.supervisor = supervisor
        self.log = log_fn

        self._active_coil_idx: Optional[int] = None

        # Heartbeat re-fires the active packet at 200 ms so the Arduino's
        # 500 ms soft-watchdog doesn't zero the coil while the operator is
        # reading a meter / watching a bead.
        self._heartbeat = QtCore.QTimer(self)
        self._heartbeat.setInterval(200)
        self._heartbeat.timeout.connect(self._heartbeat_tick)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        info = QtWidgets.QLabel(
            "Live-fire per-coil tuning. Click Test to fire one coil at the "
            "chosen strength × its gain. Adjust the gain spinbox while it's "
            "firing to trim in place; auto-saves to config on every edit. "
            "Use the Wizard tab for a full Bmap sweep with a Hall probe."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        test_row = QtWidgets.QHBoxLayout()
        test_row.addWidget(QtWidgets.QLabel("Test drive strength:"))
        self.test_strength = _spinbox(0.0, 1.0, 0.3, step=0.05, decimals=2)
        self.test_strength.valueChanged.connect(self._on_strength_changed)
        test_row.addWidget(self.test_strength)
        test_row.addStretch()
        stop = QtWidgets.QPushButton("Stop All")
        stop.clicked.connect(self._stop_all)
        test_row.addWidget(stop)
        layout.addLayout(test_row)

        gains0 = list(config.get("calibration.per_coil_gains", [1.0] * 6))
        cmap0 = list(config.get("calibration.channel_map", [0, 1, 2, 3, 4, 5]))

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.addWidget(QtWidgets.QLabel("<b>Coil</b>"),    0, 0)
        grid.addWidget(QtWidgets.QLabel("<b>Gain</b>"),    0, 1)
        grid.addWidget(QtWidgets.QLabel("<b>Driver</b>"),  0, 2)
        grid.addWidget(QtWidgets.QLabel(""),               0, 3)

        self.gain_spinboxes: list[QtWidgets.QDoubleSpinBox] = []
        self.channel_spinboxes: list[QtWidgets.QSpinBox] = []
        for i, label in enumerate(_COIL_LABELS):
            grid.addWidget(QtWidgets.QLabel(label), i + 1, 0)

            g = float(gains0[i]) if i < len(gains0) else 1.0
            spin = _spinbox(-2.0, 2.0, g, step=0.05, decimals=3)
            spin.valueChanged.connect(lambda _v, idx=i: self._on_gain_changed(idx))
            grid.addWidget(spin, i + 1, 1)
            self.gain_spinboxes.append(spin)

            c = int(cmap0[i]) if i < len(cmap0) else i
            drv = QtWidgets.QSpinBox()
            drv.setRange(0, 5)
            drv.setValue(c)
            drv.valueChanged.connect(lambda _v, idx=i: self._on_channel_changed(idx))
            grid.addWidget(drv, i + 1, 2)
            self.channel_spinboxes.append(drv)

            btn = QtWidgets.QPushButton("Test")
            btn.clicked.connect(lambda _v, idx=i: self._test_coil(idx))
            grid.addWidget(btn, i + 1, 3)
        layout.addLayout(grid)

        btn_row = QtWidgets.QHBoxLayout()
        b_reset = QtWidgets.QPushButton("Reset to defaults")
        b_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(b_reset)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch(1)

    # ---- heartbeat + firing --------------------------------------------

    def _own_hal(self) -> None:
        """Take exclusive ownership of the HAL so the inner loop's zeros
        don't overwrite our heartbeat. Mirrors CalibrationWizard._start."""
        if self.supervisor is not None:
            self.supervisor.calibration_active = True

    def _release_hal(self) -> None:
        if self.supervisor is not None:
            self.supervisor.calibration_active = False

    def _test_coil(self, idx: int) -> None:
        self._active_coil_idx = idx
        self._own_hal()
        self._fire_active_coil()
        if not self._heartbeat.isActive():
            self._heartbeat.start()

    def _heartbeat_tick(self) -> None:
        if self._active_coil_idx is None:
            return
        self._fire_active_coil()

    def _fire_active_coil(self) -> None:
        idx = self._active_coil_idx
        if idx is None:
            return
        strength = float(self.test_strength.value())
        gain = float(self.gain_spinboxes[idx].value())
        currents = [0.0] * 6
        currents[idx] = strength * gain
        try:
            self.hal.set_currents(currents, 0.0)
        except Exception as e:
            self.log(f"LiveCalibrationPanel: fire failed: {e}")

    def _stop_all(self) -> None:
        self._active_coil_idx = None
        if self._heartbeat.isActive():
            self._heartbeat.stop()
        try:
            self.hal.set_currents([0.0] * 6, 0.0)
        except Exception as e:
            self.log(f"LiveCalibrationPanel: zero failed: {e}")
        self._release_hal()

    # ---- edit handlers --------------------------------------------------

    def _on_gain_changed(self, idx: int) -> None:
        gains = [float(s.value()) for s in self.gain_spinboxes]
        self.config.set("calibration.per_coil_gains", gains)
        self.gainsChanged.emit()
        # If this coil is currently under Test, re-fire so the operator
        # sees the new gain immediately without waiting for the heartbeat.
        if self._active_coil_idx == idx:
            self._fire_active_coil()

    def _on_channel_changed(self, idx: int) -> None:
        # Apply intermediate keystrokes; log (not block) non-permutations
        # so the operator can edit multiple rows without silent rejection.
        cmap = [int(s.value()) for s in self.channel_spinboxes]
        self.config.set("calibration.channel_map", cmap)
        if sorted(cmap) != [0, 1, 2, 3, 4, 5]:
            self.log(
                f"LiveCalibrationPanel: channel_map {cmap} is not a "
                f"permutation of 0..5 — normal-mode sends will double up"
            )
        # Push straight into the HAL so the next packet routes correctly
        # (HAL snapshots channel_map at construction; won't auto-refresh).
        try:
            self.hal.set_channel_map(cmap)
        except AttributeError:
            # PrintHAL doesn't route; nothing to update.
            pass
        self.gainsChanged.emit()
        if self._active_coil_idx is not None:
            self._fire_active_coil()

    def _on_strength_changed(self, _v: float) -> None:
        if self._active_coil_idx is not None:
            self._fire_active_coil()

    def _reset_defaults(self) -> None:
        for spin in self.gain_spinboxes:
            spin.blockSignals(True)
            spin.setValue(1.0)
            spin.blockSignals(False)
        for i, spin in enumerate(self.channel_spinboxes):
            spin.blockSignals(True)
            spin.setValue(i)
            spin.blockSignals(False)
        self.config.set("calibration.per_coil_gains", [1.0] * 6)
        self.config.set("calibration.channel_map", [0, 1, 2, 3, 4, 5])
        try:
            self.hal.set_channel_map([0, 1, 2, 3, 4, 5])
        except AttributeError:
            pass
        self.gainsChanged.emit()
        if self._active_coil_idx is not None:
            self._fire_active_coil()

    # ---- lifecycle ------------------------------------------------------

    def hideEvent(self, event) -> None:
        # If the operator switches tabs mid-Test, stop firing.
        self._stop_all()
        super().hideEvent(event)
