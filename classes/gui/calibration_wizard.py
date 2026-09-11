"""Guided Bmap calibration wizard for a scalar Hall probe (TD8620),
using per-coil back-of-core measurements.

Measurement model
-----------------

The operator has a scalar / single-axis probe and can only conveniently
measure the field strength at **one location per coil** — the back of that
coil's core. So each measurement yields one number: how strong coil ``j``'s
field is at its own reference point, at duty ``d``.

That's enough to build the full Bmap when combined with the rig's geometry:

  Bmap[:, j] = slope_j · n̂_j

where ``slope_j`` is the fitted "mT per unit duty" for coil ``j`` (from the
scalar sweep) and ``n̂_j`` is coil ``j``'s geometric unit axis (from the
top-ring-0/120/240°, bottom-mirrored layout committed in
``config_example.yaml``). Coils that are wound weaker end up with shorter
Bmap columns; direction always comes from geometry.

**This does not check physical alignment.** If someone rebuilt or rotated a
coil, the geometry side of the equation is wrong and the resulting Bmap
column points in the wrong direction. Update the geometry constants (see
the ``_GEOM_AXES`` block below and ``config_example.yaml``) before running
the wizard in that case.

Modes
-----

* **Quick** (default, 6 measurements): fire each coil at 100% duty, one
  scalar per coil. Bmap[:, j] = reading_j · n̂_j.

* **Full** (24 measurements): four duty settings per coil with a
  through-origin linear regression. Surfaces bad probe placements as high
  fit residuals and validates linearity.

Both end with a per-coil saturation-onset prompt (optional in Quick, one
prompt per coil in Full).

Heartbeat
---------

While a measurement card is up the wizard re-fires the same per-coil
packet every 200 ms. Without this the Arduino soft-watchdog
(``main_3DTweezers.ino``, 500 ms) zeros the coils while the operator is
reading the probe display and every measurement comes back near zero.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np
from PyQt5 import QtCore, QtWidgets


DUTIES_QUICK = [1.0]
DUTIES_FULL = [0.25, 0.5, 0.75, 1.0]

# Geometry-derived coil unit axes. Matches config_example.yaml's Bmap
# direction convention (top ring azimuth 0/120/240 @ 45° above horizontal,
# bottom ring mirrored below). Each column is a unit vector — n̂_j.
_S = np.sqrt(2) / 2
_AZ = np.deg2rad([0.0, 120.0, 240.0])
_GEOM_AXES = np.column_stack([
    *(np.array([_S * np.cos(a), _S * np.sin(a),  _S]) for a in _AZ),
    *(np.array([_S * np.cos(a), _S * np.sin(a), -_S]) for a in _AZ),
])


class CalibrationWizard(QtWidgets.QWidget):

    finished = QtCore.pyqtSignal()

    def __init__(self, hal, config, supervisor=None, log_fn=print, parent=None):
        super().__init__(parent)
        self.hal = hal
        self.config = config
        self.supervisor = supervisor
        self.log = log_fn

        # (coil_idx, duty, scalar_reading_mT)
        self.samples: List[tuple] = []
        self.i_max_per_coil = [1.0] * 6
        self.plan: List[dict] = []
        self.step_idx = 0

        # Heartbeat re-fires the current test packet at 200 ms so the
        # Arduino watchdog stays fed while the operator reads the display.
        self._heartbeat = QtCore.QTimer(self)
        self._heartbeat.setInterval(200)
        self._heartbeat.timeout.connect(self._heartbeat_tick)
        self._active_fire = None  # (coil, duty) or None
        # Which coil is currently energized across sub-steps (reorient →
        # measure(s) → saturation). Only zeros when this changes.
        self._current_coil = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.title = QtWidgets.QLabel(
            "Bmap Calibration Wizard (TD8620 · per-coil probe)")
        self.title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(self.title)

        # --- Setup card --------------------------------------------------
        self.setup_card = QtWidgets.QGroupBox("Setup")
        s_lay = QtWidgets.QVBoxLayout(self.setup_card)
        s_lay.addWidget(QtWidgets.QLabel(
            "For each of the six coils in turn you'll:\n"
            "  1. Place the TD8620 probe tip on the back of that coil's core.\n"
            "  2. Fire the coil — read the display — type the reading.\n"
            "\n"
            "The wizard uses the per-coil strengths together with the rig's\n"
            "known geometry (top ring 0/120/240° @ 45° from vertical, bottom\n"
            "ring mirrored) to build the full 3×6 Bmap. Coil directions come\n"
            "from geometry — if a coil was rebuilt or moved, update the\n"
            "geometry in config_example.yaml before running this wizard."
        ))
        mode_row = QtWidgets.QHBoxLayout()
        self.quick_radio = QtWidgets.QRadioButton("Quick — 6 measurements")
        self.quick_radio.setChecked(True)
        self.full_radio = QtWidgets.QRadioButton(
            "Full sweep — 24 measurements (linearity fit)")
        mode_row.addWidget(self.quick_radio)
        mode_row.addWidget(self.full_radio)
        s_lay.addLayout(mode_row)

        self.start_btn = QtWidgets.QPushButton("Start calibration")
        self.start_btn.clicked.connect(self._start)
        s_lay.addWidget(self.start_btn)

        self.reset_btn = QtWidgets.QPushButton(
            "Reset calibration to geometric defaults")
        self.reset_btn.clicked.connect(self._reset_calibration)
        s_lay.addWidget(self.reset_btn)
        layout.addWidget(self.setup_card)

        # --- Reorient card (per coil) ------------------------------------
        self.reorient_card = QtWidgets.QGroupBox("Move probe")
        r_lay = QtWidgets.QVBoxLayout(self.reorient_card)
        self.reorient_label = QtWidgets.QLabel("—")
        self.reorient_label.setWordWrap(True)
        self.reorient_label.setStyleSheet("font-size: 11pt;")
        r_lay.addWidget(self.reorient_label)
        r_btn_row = QtWidgets.QHBoxLayout()
        self.reorient_back_btn = QtWidgets.QPushButton("◀ Back")
        self.reorient_back_btn.clicked.connect(self._go_back)
        self.reorient_btn = QtWidgets.QPushButton("Probe in position — continue")
        self.reorient_btn.clicked.connect(self._reorient_next)
        r_btn_row.addWidget(self.reorient_back_btn)
        r_btn_row.addWidget(self.reorient_btn)
        r_lay.addLayout(r_btn_row)
        self.reorient_card.setVisible(False)
        layout.addWidget(self.reorient_card)

        # --- Measure card ------------------------------------------------
        self.measure_card = QtWidgets.QGroupBox("Measurement")
        m_lay = QtWidgets.QVBoxLayout(self.measure_card)
        self.measure_label = QtWidgets.QLabel("—")
        self.measure_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        m_lay.addWidget(self.measure_label)
        m_lay.addWidget(QtWidgets.QLabel(
            "Coil is firing now (packet re-sent every 200 ms).\n"
            "Read the TD8620 display and type the value."
        ))
        self.measure_spin = QtWidgets.QDoubleSpinBox()
        self.measure_spin.setRange(-2000.0, 2000.0)
        self.measure_spin.setDecimals(3)
        self.measure_spin.setSingleStep(0.1)
        self.measure_spin.setSuffix(" mT")
        m_lay.addWidget(self.measure_spin)
        m_btn_row = QtWidgets.QHBoxLayout()
        self.measure_back_btn = QtWidgets.QPushButton("◀ Back")
        self.measure_back_btn.clicked.connect(self._go_back)
        self.skip_btn = QtWidgets.QPushButton("Skip")
        self.skip_btn.clicked.connect(self._measure_skip)
        self.next_btn = QtWidgets.QPushButton("Record and next")
        self.next_btn.setDefault(True)
        self.next_btn.clicked.connect(self._measure_next)
        m_btn_row.addWidget(self.measure_back_btn)
        m_btn_row.addWidget(self.skip_btn)
        m_btn_row.addWidget(self.next_btn)
        m_lay.addLayout(m_btn_row)
        self.measure_card.setVisible(False)
        layout.addWidget(self.measure_card)

        # --- Saturation-onset card (Full mode only) ----------------------
        self.sat_card = QtWidgets.QGroupBox("Saturation onset (optional)")
        sat_lay = QtWidgets.QVBoxLayout(self.sat_card)
        self.sat_label = QtWidgets.QLabel("—")
        self.sat_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        sat_lay.addWidget(self.sat_label)
        sat_lay.addWidget(QtWidgets.QLabel(
            "Optional. Enter the duty at which the probe reading stopped\n"
            "climbing linearly with the drive. Leave at 1.0 if it stayed\n"
            "linear all the way to full duty."
        ))
        self.sat_spin = QtWidgets.QDoubleSpinBox()
        self.sat_spin.setRange(0.1, 1.0)
        self.sat_spin.setValue(1.0)
        self.sat_spin.setSingleStep(0.05)
        self.sat_spin.setDecimals(2)
        sat_lay.addWidget(self.sat_spin)
        sat_btn_row = QtWidgets.QHBoxLayout()
        self.sat_back_btn = QtWidgets.QPushButton("◀ Back")
        self.sat_back_btn.clicked.connect(self._go_back)
        self.sat_next = QtWidgets.QPushButton("Record and next coil")
        self.sat_next.clicked.connect(self._sat_next)
        sat_btn_row.addWidget(self.sat_back_btn)
        sat_btn_row.addWidget(self.sat_next)
        sat_lay.addLayout(sat_btn_row)
        self.sat_card.setVisible(False)
        layout.addWidget(self.sat_card)

        # --- Summary card ------------------------------------------------
        self.summary_card = QtWidgets.QGroupBox("Summary")
        sum_lay = QtWidgets.QVBoxLayout(self.summary_card)
        self.summary_label = QtWidgets.QLabel("—")
        self.summary_label.setStyleSheet("font-family: monospace;")
        sum_lay.addWidget(self.summary_label)
        self.finish_btn = QtWidgets.QPushButton("Write to config.yaml")
        self.finish_btn.clicked.connect(self._finish)
        sum_lay.addWidget(self.finish_btn)
        self.summary_card.setVisible(False)
        layout.addWidget(self.summary_card)

        layout.addStretch(1)

    # ---- flow -----------------------------------------------------------

    def _start(self) -> None:
        # Take exclusive ownership of the HAL. Without this the 200 Hz
        # inner-loop Mode.OFF zeros overwrite the wizard's 5 Hz heartbeat.
        if self.supervisor is not None:
            self.supervisor.calibration_active = True

        duties = DUTIES_FULL if self.full_radio.isChecked() else DUTIES_QUICK
        include_sat = self.full_radio.isChecked()  # only in Full mode
        self.samples.clear()
        self.i_max_per_coil = [1.0] * 6

        # Plan: for each coil — one reorient card, all duties as measure
        # cards, and (in Full) a saturation-onset card. Grouped by coil so
        # the operator moves the probe exactly six times.
        self.plan = []
        for coil in range(6):
            self.plan.append({"type": "reorient", "coil": coil})
            for duty in duties:
                self.plan.append({"type": "measure",
                                  "coil": coil,
                                  "duty": duty})
            if include_sat:
                self.plan.append({"type": "saturation", "coil": coil})

        self.step_idx = 0
        self.setup_card.setVisible(False)
        self._show_current_step()

    def _show_current_step(self) -> None:
        self.reorient_card.setVisible(False)
        self.measure_card.setVisible(False)
        self.sat_card.setVisible(False)

        if self.step_idx >= len(self.plan):
            self._compute_and_show_summary()
            return

        step = self.plan[self.step_idx]
        kind = step["type"]
        coil = step["coil"]

        # Only zero when handing off to a different coil. Within one coil's
        # sub-steps the heartbeat keeps firing; _begin_fire just overwrites
        # the active (coil, duty) so duty changes swap in place.
        if coil != self._current_coil:
            self._stop_heartbeat_and_zero()
            self._current_coil = coil

        # Back is disabled at the first step (nothing to undo).
        can_go_back = self.step_idx > 0
        self.reorient_back_btn.setEnabled(can_go_back)
        self.measure_back_btn.setEnabled(can_go_back)
        self.sat_back_btn.setEnabled(can_go_back)

        if kind == "reorient":
            self.reorient_label.setText(
                f"Move the TD8620 probe tip onto the back of coil "
                f"C{coil + 1}'s core.\n\n"
                f"Coil C{coil + 1} is firing at full power now — watch the\n"
                f"reading climb as you seat the probe, then press Continue."
            )
            self.reorient_card.setVisible(True)
            self._begin_fire(coil, 1.0)

        elif kind == "measure":
            self.measure_label.setText(
                f"Coil C{coil + 1}, duty {step['duty']:.2f}"
            )
            self.measure_spin.blockSignals(True)
            self.measure_spin.setValue(0.0)
            self.measure_spin.blockSignals(False)
            self.measure_spin.setFocus()
            self.measure_card.setVisible(True)
            self._begin_fire(coil, step["duty"])

        elif kind == "saturation":
            self.sat_label.setText(
                f"Coil C{coil + 1} — saturation onset duty")
            self.sat_spin.blockSignals(True)
            self.sat_spin.setValue(self.i_max_per_coil[coil])
            self.sat_spin.blockSignals(False)
            self.sat_card.setVisible(True)
            self._begin_fire(coil, 1.0)

    # ---- coil driving + heartbeat --------------------------------------

    def _begin_fire(self, coil: int, duty: float) -> None:
        self._active_fire = (coil, duty)
        self._fire_only(coil, duty)
        if not self._heartbeat.isActive():
            self._heartbeat.start()

    def _heartbeat_tick(self) -> None:
        af = self._active_fire
        if af is None:
            return
        self._fire_only(*af)

    def _stop_heartbeat_and_zero(self) -> None:
        self._active_fire = None
        self._current_coil = None
        if self._heartbeat.isActive():
            self._heartbeat.stop()
        try:
            self.hal.set_currents([0.0] * 6, 0.0)
        except Exception:
            pass

    def _fire_only(self, coil: int, duty: float) -> None:
        currents = [0.0] * 6
        currents[coil] = duty
        try:
            self.hal.set_currents(currents, 0.0)
        except Exception as e:
            self.log(f"CalibrationWizard: fire failed: {e}")

    # ---- button handlers -----------------------------------------------

    def _reorient_next(self) -> None:
        self.step_idx += 1
        self._show_current_step()

    def _measure_skip(self) -> None:
        self.step_idx += 1
        self._show_current_step()

    def _measure_next(self) -> None:
        step = self.plan[self.step_idx]
        if step["type"] != "measure":
            return
        reading = float(self.measure_spin.value())
        self.samples.append((step["coil"], step["duty"], reading))
        self.step_idx += 1
        self._show_current_step()

    def _go_back(self) -> None:
        # Undo one step. If the step we're going back to already recorded
        # something (a measure sample or a saturation entry), roll that back
        # so the operator can re-enter it.
        if self.step_idx == 0:
            return
        prev = self.plan[self.step_idx - 1]
        if prev["type"] == "measure":
            if (self.samples
                    and self.samples[-1][0] == prev["coil"]
                    and self.samples[-1][1] == prev["duty"]):
                self.samples.pop()
        elif prev["type"] == "saturation":
            self.i_max_per_coil[prev["coil"]] = 1.0
        self.step_idx -= 1
        self._show_current_step()

    def _sat_next(self) -> None:
        step = self.plan[self.step_idx]
        if step["type"] != "saturation":
            return
        self.i_max_per_coil[step["coil"]] = float(self.sat_spin.value())
        self.step_idx += 1
        self._show_current_step()

    # ---- fitting -------------------------------------------------------

    def _compute_and_show_summary(self) -> None:
        self._stop_heartbeat_and_zero()

        slope = np.zeros(6)         # mT per unit duty at the coil's back-of-core
        residual = np.zeros(6)      # regression residual (Full mode)
        n_pts = np.zeros(6, dtype=int)

        for coil in range(6):
            pts = [(d, r) for (c, d, r) in self.samples if c == coil]
            n_pts[coil] = len(pts)
            if not pts:
                continue
            X = np.array([p[0] for p in pts], dtype=float)
            Y = np.array([p[1] for p in pts], dtype=float)
            denom = float(X @ X)
            if denom < 1e-12:
                continue
            slope[coil] = float(X @ Y) / denom
            pred = X * slope[coil]
            residual[coil] = float(np.linalg.norm(Y - pred))

        # Bmap column j = slope_j · n̂_j (geometry supplies direction).
        # np.abs(slope) — the operator's probe reads magnitude; if a coil is
        # wound backward the sign has to be flipped via calibration.per_coil_gains
        # (that's still exposed elsewhere in the config).
        Bmap = _GEOM_AXES * np.abs(slope)[np.newaxis, :]

        lines = ["Per-coil strength at back-of-core (mT per unit duty):"]
        for c in range(6):
            lines.append(
                f"  C{c + 1}: {slope[c]:8.3f} mT/duty   "
                f"({n_pts[c]} points, residual {residual[c]:.3f} mT, "
                f"i_max={self.i_max_per_coil[c]:.2f})"
            )
        lines.append("")
        lines.append("Resulting Bmap (mT per unit duty):")
        lines.append(
            "         " + "  ".join(f"C{c + 1}      " for c in range(6)))
        for a, name in enumerate("xyz"):
            row = f"  B{name}   "
            for c in range(6):
                row += f"{Bmap[a, c]:+7.3f}  "
            lines.append(row)

        self.summary_label.setText("\n".join(lines))
        self.summary_card.setVisible(True)
        self._computed_Bmap = Bmap.tolist()

    def _reset_calibration(self) -> None:
        # Restore Bmap to the geometry-only unit-magnitude defaults so the
        # operator can test whether a bad calibration is causing skew. Backs
        # up the current Bmap under a timestamped key first.
        reply = QtWidgets.QMessageBox.question(
            self,
            "Reset calibration?",
            "Overwrite Bmap, per_coil_gains, and i_max_per_coil with the\n"
            "neutral geometric defaults (unit magnitude, geometry directions).\n"
            "The current Bmap will be backed up under a timestamped key so\n"
            "it can be restored from config.yaml. channel_map is left alone.\n\n"
            "Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        prev = self.config.get("calibration.Bmap")
        if prev is not None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.config.set(f"calibration.Bmap_backup_{ts}", prev)
        self.config.set("calibration.Bmap", _GEOM_AXES.tolist())
        self.config.set("calibration.per_coil_gains", [1.0] * 6)
        self.config.set("calibration.i_max_per_coil", [1.0] * 6)
        self.config.set(
            "calibration.date",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (reset)")
        self.config.save()
        self.log(
            "CalibrationWizard: reset Bmap to geometric defaults, "
            "per_coil_gains and i_max_per_coil to [1.0]*6")
        self.finished.emit()

    def _finish(self) -> None:
        if not hasattr(self, "_computed_Bmap"):
            return
        prev = self.config.get("calibration.Bmap")
        if prev is not None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.config.set(f"calibration.Bmap_backup_{ts}", prev)
        self.config.set("calibration.Bmap", self._computed_Bmap)
        self.config.set("calibration.i_max_per_coil",
                        list(self.i_max_per_coil))
        self.config.set("calibration.date",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.config.save()
        self.log("CalibrationWizard: Bmap written to config.yaml")
        self.summary_card.setVisible(False)
        self.title.setText("Calibration Wizard — complete")
        if self.supervisor is not None:
            self.supervisor.calibration_active = False
        self.finished.emit()

    # ---- lifecycle -----------------------------------------------------

    def hideEvent(self, event) -> None:
        # If the operator switches away from the Calibration tab mid-run,
        # stop the heartbeat and zero. Otherwise the coils keep firing
        # unattended. Also release the supervisor pause so the normal
        # control pipeline resumes writing to HAL.
        self._stop_heartbeat_and_zero()
        if self.supervisor is not None:
            self.supervisor.calibration_active = False
        super().hideEvent(event)
