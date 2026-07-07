"""
Per-coil calibration UI.

Rationale: physically the six tweezer coils in the rig produce different field
strengths for identical PWM duty (winding count / core / driver channel vary).
This dialog lets the operator fire one coil at a time via
ArduinoHandler.send_calibration_pulse(), read a magnetometer, then trim a gain
multiplier for that coil until every coil reaches the same target field (e.g.
2 mT). The gains are stored in ArduinoHandler.coil_gains and applied by the
Arduino firmware on every normal-mode packet, so all downstream field commands
produce a matched field per unit command.

Persistence lives in calibration.json in the repo root.
"""

import json
import os

from PyQt5 import QtCore, QtWidgets


COIL_LABELS = [
    "C1 — Top, tip along −Y",
    "C2 — Top, 30° above +X",
    "C3 — Top, 30° above −X",
    "C4 — Bottom, under C1 (−Y)",
    "C5 — Bottom, under C2 (+X)",
    "C6 — Bottom, under C3 (−X)",
]


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, arduino, tbprint, calibration_path, parent=None):
        super().__init__(parent)
        self.arduino = arduino
        self.tbprint = tbprint
        self.calibration_path = calibration_path
        self.setWindowTitle("Coil Calibration")
        self.setMinimumWidth(560)

        layout = QtWidgets.QVBoxLayout(self)

        # Header instructions
        instr = QtWidgets.QLabel(
            "Adjust each coil's gain until the magnetometer reads the target field "
            "(e.g. 2 mT). Click Test to pulse only that coil. Save persists the values "
            "to calibration.json, which is auto-loaded on next launch."
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        # Common test-strength control (base PWM duty for the test pulse)
        test_row = QtWidgets.QHBoxLayout()
        test_row.addWidget(QtWidgets.QLabel("Test pulse base strength (0.0 – 1.0):"))
        self.test_strength = QtWidgets.QDoubleSpinBox()
        self.test_strength.setRange(0.0, 1.0)
        self.test_strength.setSingleStep(0.05)
        self.test_strength.setDecimals(2)
        self.test_strength.setValue(0.5)
        test_row.addWidget(self.test_strength)
        test_row.addStretch()
        stop_all = QtWidgets.QPushButton("Stop All")
        stop_all.clicked.connect(self._stop_all)
        test_row.addWidget(stop_all)
        layout.addLayout(test_row)

        # Per-coil rows
        self.gain_spinboxes = []
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.addWidget(QtWidgets.QLabel("<b>Coil</b>"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("<b>Gain</b>"), 0, 1)
        grid.addWidget(QtWidgets.QLabel(""), 0, 2)
        for i, label in enumerate(COIL_LABELS):
            grid.addWidget(QtWidgets.QLabel(label), i + 1, 0)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, 2.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(1.0)
            self.gain_spinboxes.append(spin)
            grid.addWidget(spin, i + 1, 1)

            test_btn = QtWidgets.QPushButton("Test")
            test_btn.clicked.connect(lambda _, idx=i: self._test_coil(idx))
            grid.addWidget(test_btn, i + 1, 2)
        layout.addLayout(grid)

        # Save / load / reset buttons
        btn_row = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save + Apply")
        save_btn.clicked.connect(self._save_and_apply)
        load_btn = QtWidgets.QPushButton("Reload from file")
        load_btn.clicked.connect(self._reload_from_file)
        reset_btn = QtWidgets.QPushButton("Reset to 1.0")
        reset_btn.clicked.connect(self._reset_all)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Populate spinboxes from the arduino handler's current gains
        for spin, g in zip(self.gain_spinboxes, self.arduino.coil_gains):
            spin.blockSignals(True)
            spin.setValue(float(g))
            spin.blockSignals(False)

    def _test_coil(self, coil_index: int) -> None:
        """Fire one coil at (base test strength) * (that coil's current gain)."""
        base = float(self.test_strength.value())
        gain = float(self.gain_spinboxes[coil_index].value())
        strength = max(-1.0, min(1.0, base * gain))
        self.arduino.send_calibration_pulse(coil_index, strength)
        self.tbprint(
            f"Calibration: pulsing C{coil_index + 1} at "
            f"base={base:.2f} × gain={gain:.2f} = duty {strength:.3f}"
        )

    def _stop_all(self) -> None:
        self.arduino.send_calibration_all_off()

    def _save_and_apply(self) -> None:
        gains = [float(s.value()) for s in self.gain_spinboxes]
        self.arduino.set_gains(gains)
        try:
            with open(self.calibration_path, "w") as f:
                json.dump({"coil_gains": gains}, f, indent=2)
            self.tbprint(f"Calibration saved to {self.calibration_path}")
        except OSError as e:
            self.tbprint(f"Failed to save calibration: {e}")

    def _reload_from_file(self) -> None:
        gains = load_calibration(self.calibration_path)
        if gains is None:
            self.tbprint(f"No calibration file at {self.calibration_path}")
            return
        for spin, g in zip(self.gain_spinboxes, gains):
            spin.blockSignals(True)
            spin.setValue(float(g))
            spin.blockSignals(False)
        self.arduino.set_gains(gains)

    def _reset_all(self) -> None:
        for spin in self.gain_spinboxes:
            spin.blockSignals(True)
            spin.setValue(1.0)
            spin.blockSignals(False)


def load_calibration(path):
    """Return the 6 gain floats from a calibration.json, or None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        gains = data.get("coil_gains")
        if isinstance(gains, list) and len(gains) == 6:
            return [float(g) for g in gains]
    except (OSError, ValueError):
        pass
    return None
