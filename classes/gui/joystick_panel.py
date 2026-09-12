"""Joystick tab: connection status, live axes/buttons, mapping legend, and an
optional live coil-firing + pull-direction visualization at the top."""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from classes.gui.joystick_bridge import JoystickBridge
from classes.gui.widgets import LabeledDoubleSpinBox, make_group
from classes.motion_controller import Mode


class _AxisBar(QtWidgets.QWidget):
    """Bipolar bar: −1 fills left half, +1 fills right half. Cheap paint."""

    def __init__(self, label: str, bipolar: bool = True, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._bipolar = bipolar
        self.setMinimumHeight(18)
        self._label = label

    def set_value(self, v: float) -> None:
        self._value = max(-1.0, min(1.0, float(v)))
        self.update()

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        rect = self.rect()
        p.fillRect(rect, QtGui.QColor("#202020"))
        w = rect.width()
        h = rect.height()
        if self._bipolar:
            mid = w // 2
            v = self._value
            if v >= 0:
                bar_w = int(mid * v)
                p.fillRect(mid, 0, bar_w, h, QtGui.QColor("#4a90e2"))
            else:
                bar_w = int(mid * -v)
                p.fillRect(mid - bar_w, 0, bar_w, h, QtGui.QColor("#e07a4a"))
            p.setPen(QtGui.QColor("#555"))
            p.drawLine(mid, 0, mid, h)
        else:
            bar_w = int(w * self._value)
            p.fillRect(0, 0, bar_w, h, QtGui.QColor("#4a90e2"))
        p.setPen(QtGui.QColor("#ccc"))
        p.drawText(rect.adjusted(6, 0, -6, 0), QtCore.Qt.AlignVCenter,
                   f"{self._label}: {self._value:+.2f}")


class _ButtonLED(QtWidgets.QLabel):
    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 20)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(str(idx))
        self._idx = idx
        self._on = False
        self._render()

    def set_on(self, on: bool) -> None:
        if on == self._on:
            return
        self._on = on
        self._render()

    def _render(self) -> None:
        color = "#4a90e2" if self._on else "#404040"
        self.setStyleSheet(
            f"background-color: {color}; color: white; "
            f"font-family: monospace; border-radius: 4px;")


class JoystickPanel(QtWidgets.QWidget):

    def __init__(self,
                 bridge: JoystickBridge,
                 coil_viz: Optional[QtWidgets.QWidget] = None,
                 parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.coil_viz = coil_viz
        self.motion = bridge.motion
        self.config = bridge.config

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Coil visualization at the very top (if wired) --------------
        if coil_viz is not None:
            layout.addWidget(make_group(
                "Live coil currents + predicted pull direction", coil_viz))

        # --- Connection controls ---------------------------------------
        conn_row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(bridge.start)
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(bridge.stop)
        self.reconnect_btn = QtWidgets.QPushButton("Reconnect")
        self.reconnect_btn.clicked.connect(bridge.reconnect)
        conn_row.addWidget(self.connect_btn)
        conn_row.addWidget(self.disconnect_btn)
        conn_row.addWidget(self.reconnect_btn)
        conn_widget = QtWidgets.QWidget()
        conn_widget.setLayout(conn_row)
        layout.addWidget(make_group("Connection", conn_widget))

        self.status = QtWidgets.QLabel("disconnected")
        self.status.setStyleSheet(
            "font-family: monospace; padding: 4px;")
        self.status.setToolTip(
            "If your pad is plugged in but this stays on 'no joystick' on "
            "macOS 15+, grant the terminal Input Monitoring permission in "
            "System Settings → Privacy & Security.")
        layout.addWidget(self.status)

        self.enable_check = QtWidgets.QCheckBox(
            "Enabled (unchecked = pygame stays connected but writes are ignored)")
        self.enable_check.setChecked(True)
        self.enable_check.toggled.connect(bridge.set_enabled)
        layout.addWidget(self.enable_check)

        # --- Active mode + stick behavior ------------------------------
        mode_widget = QtWidgets.QWidget()
        mode_lay = QtWidgets.QVBoxLayout(mode_widget)
        mode_lay.setContentsMargins(0, 0, 0, 0)

        self.mode_label = QtWidgets.QLabel("Mode: OFF")
        self.mode_label.setStyleSheet(
            "font-family: monospace; font-weight: bold; padding: 4px; "
            "color: #888;")
        mode_lay.addWidget(self.mode_label)

        stick_row = QtWidgets.QHBoxLayout()
        stick_row.addWidget(QtWidgets.QLabel("Stick engages:"))
        self.rolling_radio = QtWidgets.QRadioButton("Rolling (Mode A)")
        self.static_radio = QtWidgets.QRadioButton("Static pull (Mode B)")
        cur = str(self.config.get("joystick.stick_mode", "rolling"))
        (self.static_radio if cur == "static"
         else self.rolling_radio).setChecked(True)
        self.rolling_radio.toggled.connect(self._on_stick_mode)
        stick_row.addWidget(self.rolling_radio)
        stick_row.addWidget(self.static_radio)
        stick_row.addStretch(1)
        mode_lay.addLayout(stick_row)

        inv_row = QtWidgets.QHBoxLayout()
        self.invert_x_check = QtWidgets.QCheckBox("Invert X")
        self.invert_y_check = QtWidgets.QCheckBox("Invert Y")
        self.invert_x_check.setChecked(
            bool(self.config.get("joystick.invert_x", False)))
        self.invert_y_check.setChecked(
            bool(self.config.get("joystick.invert_y", False)))
        self.invert_x_check.toggled.connect(
            lambda on: self._on_invert("joystick.invert_x", on))
        self.invert_y_check.toggled.connect(
            lambda on: self._on_invert("joystick.invert_y", on))
        inv_row.addWidget(self.invert_x_check)
        inv_row.addWidget(self.invert_y_check)
        inv_row.addStretch(1)
        mode_lay.addLayout(inv_row)

        # Frequency with the right stick centered; pushing right scales it
        # toward modes.mode_a.freq_max_hz, pushing left reverses.
        freq_max = float(self.config.get("modes.mode_a.freq_max_hz", 20.0))
        self.freq_spin = LabeledDoubleSpinBox(
            "Base roll frequency", 0.0, freq_max, 0.1, 2,
            suffix="Hz", config=self.config,
            config_path="modes.mode_a.freq_default")
        mode_lay.addWidget(self.freq_spin)

        self.mag_spin = LabeledDoubleSpinBox(
            "Magnitude", 0.0, 1.0, 0.05, 2,
            config=self.config,
            config_path="modes.mode_a.magnitude_default")
        mode_lay.addWidget(self.mag_spin)

        shared_note = QtWidgets.QLabel(
            "Shared by the joystick, path follower, Frame Cal pulses and "
            "the Mode A tab.")
        shared_note.setStyleSheet("color: #888;")
        shared_note.setWordWrap(True)
        mode_lay.addWidget(shared_note)

        # Coil power dial — limits.output_scale, the final HAL-level scale
        # on every packet (5..100% so the rig can't be silently zeroed).
        power_row = QtWidgets.QHBoxLayout()
        self.power_dial = QtWidgets.QDial()
        self.power_dial.setRange(5, 100)
        self.power_dial.setNotchesVisible(True)
        self.power_dial.setFixedSize(64, 64)
        scale = float(self.config.get("limits.output_scale", 1.0))
        self.power_dial.setValue(int(round(scale * 100)))
        self.power_label = QtWidgets.QLabel("")
        self.power_label.setStyleSheet("font-family: monospace;")
        self._refresh_power_label(self.power_dial.value())
        self.power_dial.valueChanged.connect(self._on_power_dial)
        power_row.addWidget(self.power_dial)
        power_row.addWidget(self.power_label)
        power_row.addStretch(1)
        mode_lay.addLayout(power_row)

        # Debounced persist — spinbox drags fire config.set per step; save
        # once things settle instead of on every tick.
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(750)
        self._save_timer.timeout.connect(self._save_config)
        for w in (self.freq_spin, self.mag_spin):
            w.valueChanged.connect(lambda _v: self._save_timer.start())

        layout.addWidget(make_group("Mode", mode_widget))

        # MotionController isn't a QObject, so poll the mode enum — this
        # tracks every writer (bridge, pad buttons, GUI panels, supervisor).
        self._mode_timer = QtCore.QTimer(self)
        self._mode_timer.setInterval(100)
        self._mode_timer.timeout.connect(self._refresh_mode)
        self._mode_timer.start()

        # --- Live axes -------------------------------------------------
        axes_widget = QtWidgets.QWidget()
        axes_lay = QtWidgets.QVBoxLayout(axes_widget)
        axes_lay.setContentsMargins(0, 0, 0, 0)
        self.axis_bars = {
            "left_x": _AxisBar("L stick X", bipolar=True),
            "left_y": _AxisBar("L stick Y", bipolar=True),
            "right_x": _AxisBar("R stick X", bipolar=True),
            "right_y": _AxisBar("R stick Y", bipolar=True),
            "lt": _AxisBar("LT", bipolar=False),
            "rt": _AxisBar("RT", bipolar=False),
        }
        for key in ("left_x", "left_y", "right_x", "right_y", "lt", "rt"):
            axes_lay.addWidget(self.axis_bars[key])
        layout.addWidget(make_group("Live axes", axes_widget))

        # --- Live buttons ----------------------------------------------
        btn_widget = QtWidgets.QWidget()
        btn_grid = QtWidgets.QGridLayout(btn_widget)
        btn_grid.setContentsMargins(0, 0, 0, 0)
        btn_grid.setSpacing(4)
        self.button_leds: list = []
        for i in range(16):
            led = _ButtonLED(i)
            self.button_leds.append(led)
            btn_grid.addWidget(led, i // 8, i % 8)
        layout.addWidget(make_group("Buttons (index)", btn_widget))

        # --- Mapping legend --------------------------------------------
        legend = QtWidgets.QLabel(
            "<b>Sticks & triggers</b>:<br>"
            "&nbsp;&nbsp;Left stick → travel direction (rolling) / "
            "Bx, By (static pull)<br>"
            "&nbsp;&nbsp;RT / LT → +Bz / −Bz (triggers alone with sticks "
            "centered = static vertical pull)<br>"
            "&nbsp;&nbsp;Right stick X → roll speed: center = base freq, "
            "right = faster, left = reverse<br>"
            "<b>Buttons</b> (Xbox default indices):<br>"
            "&nbsp;&nbsp;A(0): toggle Mode B ↔ OFF<br>"
            "&nbsp;&nbsp;B(1): toggle Mode A ↔ OFF (rotating pull)<br>"
            "&nbsp;&nbsp;X(2)/Y(3): shared Magnitude −0.1 / +0.1<br>"
            "&nbsp;&nbsp;Back(6): E-STOP (needs Start to reset)<br>"
            "&nbsp;&nbsp;Start(7): reset E-STOP<br>"
            "<span style='color: #888;'>Remap: edit config.yaml under "
            "joystick.buttons.* and reconnect.</span>"
        )
        legend.setWordWrap(True)
        layout.addWidget(make_group("Mapping", legend))

        layout.addStretch(1)

        # --- Signal wiring ---------------------------------------------
        bridge.connectionChanged.connect(self._on_conn)
        bridge.axesChanged.connect(self._on_axes)
        bridge.buttonsChanged.connect(self._on_buttons)

    def _save_config(self) -> None:
        try:
            self.config.save()
        except Exception:
            pass

    def _refresh_power_label(self, pct: int) -> None:
        self.power_label.setText(
            f"Coil power: {pct:3d}%\n(HAL output scale — caps every packet)")

    def _on_power_dial(self, pct: int) -> None:
        self._refresh_power_label(pct)
        # config.set fires on_change — MainWindow pushes it into the HAL
        # live; the debounced timer persists it.
        self.config.set("limits.output_scale", pct / 100.0)
        self._save_timer.start()

    def _on_invert(self, path: str, on: bool) -> None:
        self.config.set(path, bool(on))
        try:
            self.config.save()
        except Exception:
            pass

    def _on_stick_mode(self, _checked: bool) -> None:
        mode = "static" if self.static_radio.isChecked() else "rolling"
        self.config.set("joystick.stick_mode", mode)
        try:
            # closeEvent doesn't auto-save; persist the choice now.
            self.config.save()
        except Exception:
            pass

    def _refresh_mode(self) -> None:
        mode = self.motion.mode
        if mode == Mode.A_ROTATING:
            text, color = "Mode: A — Rolling", "#4c8"
        elif mode == Mode.B_STATIC:
            text, color = "Mode: B — Static pull", "#4a90e2"
        else:
            text, color = "Mode: OFF", "#888"
        self.mode_label.setText(text)
        self.mode_label.setStyleSheet(
            "font-family: monospace; font-weight: bold; padding: 4px; "
            f"color: {color};")

    def _on_conn(self, msg: str) -> None:
        self.status.setText(msg)
        # Green when connected, red on failure, neutral otherwise.
        if msg.startswith("connected"):
            self.status.setStyleSheet(
                "font-family: monospace; padding: 4px; "
                "color: #4c8; font-weight: bold;")
        elif msg in ("no joystick", "pygame missing", "disabled"):
            self.status.setStyleSheet(
                "font-family: monospace; padding: 4px; color: #d66;")
        else:
            self.status.setStyleSheet(
                "font-family: monospace; padding: 4px;")

    def _on_axes(self, axes: dict) -> None:
        for key, bar in self.axis_bars.items():
            if key in axes:
                bar.set_value(axes[key])

    def _on_buttons(self, buttons: list) -> None:
        for i, led in enumerate(self.button_leds):
            led.set_on(i < len(buttons) and bool(buttons[i]))
