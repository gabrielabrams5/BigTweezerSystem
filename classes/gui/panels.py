"""Control panels for the new GUI.

Each panel owns its own widgets and pushes changes into either the config
or directly into the MotionController. Panels never touch the HAL — that
authority stays with the Supervisor.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from classes.gui.widgets import (
    CoilBar,
    LabeledDoubleSpinBox,
    Vec3Editor,
    make_group,
    scroll_wrap,
)
from classes.motion_controller import Mode, MotionController, Source


class ModeAPanel(QtWidgets.QWidget):
    """Rolling-pull controls: roll axis, ramp rate, enable button.

    Magnitude + frequency are the shared knobs edited on the Joystick tab;
    this panel shows them read-only and pushes them into the controller on
    Enable. Base direction is shared with Mode B (edit it on the Mode B
    panel or via the joystick); Mode A rolls that direction around
    ``roll_axis``.
    """

    def __init__(self, motion: MotionController, config, parent=None):
        super().__init__(parent)
        self.motion = motion
        self.config = config

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.enable_btn = QtWidgets.QPushButton("Enable Mode A — Rolling Pull")
        self.enable_btn.setCheckable(True)
        self.enable_btn.toggled.connect(self._on_enable)
        layout.addWidget(self.enable_btn)

        # Magnitude + frequency are the shared knobs owned by the Joystick
        # tab — displayed here read-only.
        self.shared_lbl = QtWidgets.QLabel("—")
        self.shared_lbl.setStyleSheet("font-family: monospace; color: #888;")
        layout.addWidget(make_group("Drive (shared)", self.shared_lbl))
        self._refresh_shared()
        try:
            config.on_change(
                "modes.mode_a", lambda *_a: self._refresh_shared())
        except Exception:
            pass

        self.roll_axis = Vec3Editor("Roll axis (auto-normalized)",
                                    -1.0, 1.0, 0.05, 3,
                                    initial=tuple(motion.roll_axis),
                                    parent=self)
        self.roll_axis.valueChanged.connect(
            lambda v: motion.set_roll_axis(v, Source.GUI))
        layout.addWidget(make_group(
            "Roll axis (base direction shared with Mode B)", self.roll_axis))

        self.ramp = LabeledDoubleSpinBox(
            "Frequency ramp rate", 0.0, 30.0, 0.1, 3,
            initial=motion.freq_ramp_rate, suffix="Hz/s", parent=self)
        self.ramp.valueChanged.connect(
            lambda v: setattr(motion, "freq_ramp_rate", float(v)))
        layout.addWidget(make_group("Frequency ramp", self.ramp))

        layout.addStretch(1)

    def _refresh_shared(self) -> None:
        mag = float(self.config.get("modes.mode_a.magnitude_default", 1.0))
        freq = float(self.config.get("modes.mode_a.freq_default", 1.0))
        self.shared_lbl.setText(
            f"Magnitude {mag:.2f} · {freq:.1f} Hz — set on the Joystick tab")

    def _on_enable(self, checked: bool) -> None:
        if checked:
            # Push the shared values in before engaging — otherwise Enable
            # would run at whatever magnitude/frequency the last writer left.
            self.motion.set_magnitude(
                float(self.config.get("modes.mode_a.magnitude_default", 1.0)),
                Source.GUI)
            self.motion.set_frequency(
                float(self.config.get("modes.mode_a.freq_default", 1.0)),
                Source.GUI)
        self.motion.set_mode(Mode.A_ROTATING if checked else Mode.OFF)


class ModeBPanel(QtWidgets.QWidget):
    """Static-pull controls: magnitude + direction vector.

    Note: ``motion.magnitude`` is a single last-writer-wins field, so this
    panel's spinbox can transiently diverge from the shared Mode A value —
    acceptable for a manual bench control.
    """

    def __init__(self, motion: MotionController, config, parent=None):
        super().__init__(parent)
        self.motion = motion
        self.config = config

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.enable_btn = QtWidgets.QPushButton("Enable Mode B — Static Pull")
        self.enable_btn.setCheckable(True)
        self.enable_btn.toggled.connect(self._on_enable)
        layout.addWidget(self.enable_btn)

        self.mag = LabeledDoubleSpinBox(
            "|B| magnitude", 0.0, 1.0, 0.05, 3,
            initial=motion.magnitude, parent=self)
        self.mag.valueChanged.connect(
            lambda v: motion.set_magnitude(v, Source.GUI))
        layout.addWidget(make_group("Field magnitude", self.mag))

        self.direction = Vec3Editor("Pull direction", -1.0, 1.0, 0.05, 3,
                                    initial=tuple(motion.direction), parent=self)
        self.direction.valueChanged.connect(
            lambda v: motion.set_static_direction(v, Source.GUI))
        layout.addWidget(make_group("Direction (auto-normalized)", self.direction))

        layout.addStretch(1)

    def _on_enable(self, checked: bool) -> None:
        self.motion.set_mode(Mode.B_STATIC if checked else Mode.OFF)


class SolverPanel(QtWidgets.QWidget):
    """DLS damping λ, per-coil weights W, null-space objective, and
    saturation policy. Writes back into config; the solver has to be
    rebuilt for changes to take effect."""

    solverChanged = QtCore.pyqtSignal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.robot_combo = QtWidgets.QComboBox()
        self.robot_combo.addItems(["paramagnetic", "soft", "hard"])
        self.robot_combo.setCurrentText(config.get("robot.type", "paramagnetic"))
        self.robot_combo.currentTextChanged.connect(
            lambda t: (config.set("robot.type", t), self.solverChanged.emit()))
        robot_row = QtWidgets.QHBoxLayout()
        robot_row.addWidget(QtWidgets.QLabel("Robot type:"))
        robot_row.addWidget(self.robot_combo, 1)
        robot_widget = QtWidgets.QWidget()
        robot_widget.setLayout(robot_row)
        layout.addWidget(make_group("Physics model", robot_widget))

        self.lam = LabeledDoubleSpinBox(
            "λ (Tikhonov damping)", 0.0, 10.0, 0.001, 4,
            config=config, config_path="solver.lambda", parent=self)
        self.lam.valueChanged.connect(lambda _v: self.solverChanged.emit())
        layout.addWidget(make_group("DLS damping (signed rigs only)", self.lam))

        w_widget = QtWidgets.QWidget()
        w_row = QtWidgets.QHBoxLayout(w_widget)
        w_row.setContentsMargins(0, 0, 0, 0)
        self.w_spins = []
        Ws = config.get("solver.W", [1.0] * 6)
        for k in range(6):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setPrefix(f"C{k + 1}: ")
            spin.setRange(0.01, 10.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(float(Ws[k]))
            spin.valueChanged.connect(self._on_W)
            w_row.addWidget(spin)
            self.w_spins.append(spin)
        layout.addWidget(make_group("Per-coil cost weights (diag W)", w_widget))

        self.null_combo = QtWidgets.QComboBox()
        self.null_combo.addItems(["minimize_gradient", "minimize_power"])
        self.null_combo.setCurrentText(
            config.get("solver.null_objective", "minimize_gradient"))
        self.null_combo.currentTextChanged.connect(
            lambda t: (config.set("solver.null_objective", t),
                       self.solverChanged.emit()))
        n_row = QtWidgets.QHBoxLayout()
        n_row.addWidget(QtWidgets.QLabel("Null objective:"))
        n_row.addWidget(self.null_combo, 1)
        n_widget = QtWidgets.QWidget()
        n_widget.setLayout(n_row)
        layout.addWidget(make_group("Null-space objective (Mode A)", n_widget))

        self.sat_combo = QtWidgets.QComboBox()
        self.sat_combo.addItems(["rescale_preserve_direction", "flag"])
        self.sat_combo.setCurrentText(
            config.get("solver.saturation_policy", "rescale_preserve_direction"))
        self.sat_combo.currentTextChanged.connect(
            lambda t: (config.set("solver.saturation_policy", t),
                       self.solverChanged.emit()))
        s_row = QtWidgets.QHBoxLayout()
        s_row.addWidget(QtWidgets.QLabel("Saturation policy:"))
        s_row.addWidget(self.sat_combo, 1)
        s_widget = QtWidgets.QWidget()
        s_widget.setLayout(s_row)
        layout.addWidget(make_group("Saturation", s_widget))

        layout.addStretch(1)

    def _on_W(self, _v: float) -> None:
        self.config.set(
            "solver.W", [float(s.value()) for s in self.w_spins])
        self.solverChanged.emit()


class SupervisorPanel(QtWidgets.QWidget):
    """Live coil bars, power estimates, saturation flag, big E-STOP.

    Reads from ``Supervisor.snapshot()`` on a fixed interval and reads
    the last commanded currents from the MotionController's listener via
    the callback ``set_currents_display``.
    """

    def __init__(self, supervisor, config, parent=None):
        super().__init__(parent)
        self.supervisor = supervisor
        self.config = config

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # E-STOP.
        self.estop_btn = QtWidgets.QPushButton("EMERGENCY STOP")
        self.estop_btn.setStyleSheet(
            "QPushButton { background-color: #b30000; color: white; "
            "font-weight: bold; padding: 12px; font-size: 14pt; }")
        self.estop_btn.clicked.connect(self._on_estop)
        layout.addWidget(self.estop_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset E-Stop")
        self.reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_btn)

        # Live current bars.
        self.bars = CoilBar()
        layout.addWidget(make_group("Live coil duties", self.bars))

        # Power estimate label per coil.
        self.power_lbls = QtWidgets.QWidget()
        p_row = QtWidgets.QGridLayout(self.power_lbls)
        p_row.setContentsMargins(0, 0, 0, 0)
        self.power_labels = []
        for k in range(6):
            title = QtWidgets.QLabel(f"C{k + 1}")
            val = QtWidgets.QLabel("0.0 W")
            val.setStyleSheet("font-family: monospace;")
            p_row.addWidget(title, 0, k, alignment=QtCore.Qt.AlignCenter)
            p_row.addWidget(val, 1, k, alignment=QtCore.Qt.AlignCenter)
            self.power_labels.append(val)
        layout.addWidget(make_group("Estimated I²R (rolling)", self.power_lbls))

        # Status line
        self.status = QtWidgets.QLabel("Armed")
        self.status.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status)

        layout.addStretch(1)

        # Refresh timer
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)  # 10 Hz UI refresh
        self.timer.timeout.connect(self._refresh)
        self.timer.start()

    # Called by whoever wires the motion controller's on_solved listener.
    def set_currents_display(self, currents) -> None:
        self.bars.update_from(currents)

    def _on_estop(self) -> None:
        self.supervisor.estop()

    def _on_reset(self) -> None:
        self.supervisor.reset_estop()

    def _refresh(self) -> None:
        snap = self.supervisor.snapshot()
        for k in range(6):
            self.power_labels[k].setText(f"{snap['power_estimate_w'][k]:.1f} W")
            if snap["coil_cut"][k]:
                self.power_labels[k].setStyleSheet("color: red; font-family: monospace;")
            else:
                self.power_labels[k].setStyleSheet("font-family: monospace;")
        if snap["estopped"]:
            self.status.setText("E-STOP tripped — press Reset to arm")
            self.status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.status.setText("Armed")
            self.status.setStyleSheet("color: green; font-weight: bold;")


class LogPanel(QtWidgets.QPlainTextEdit):
    """Scrolling text log. ``append_line()`` for external callers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QtGui.QFont("Menlo", 10))
        self.setMaximumBlockCount(2000)  # cap growth

    def append_line(self, line: str) -> None:
        self.appendPlainText(line)
