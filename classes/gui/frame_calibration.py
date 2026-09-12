"""Frame calibration: fit a screen→world 2×2 transform from tracked bead motion.

Why this exists
---------------

The joystick's stick input is in world axes (bx=lx, by=-ly), but the operator
observes motion in the camera frame. When the coil rig isn't perfectly aligned
with the microscope, stick-up doesn't produce screen-up — sometimes it produces
screen-right, or a diagonal, or the wrong sign. Hand-tuning ``joystick.xy_swap``
works but is fragile.

This panel measures the mismatch directly. The operator clicks the bead in the
video, then presses one button per world direction (+X, -X, +Y, -Y). For each
press the panel:

1. Snapshots the tracked bead position.
2. Commands the ROLLING recipe in that world direction (Mode A, roll axis
   horizontal ⊥ travel, a few Hz) — paramagnetic beads translate by rolling;
   a weak static pull barely moves them.
3. When the hold timer expires, records the tracker's end position and cuts
   the coils.

Re-running this calibration and pressing Apply is the intended fix whenever
the joystick's X/Y feel swapped or rotated — do not hand-edit the matrix.
The measurement is done in RAW camera coordinates, so the camera view's
display rotation (⟳ button) never invalidates a saved matrix.

Both directions per axis are optional — if only +X is measured, the +X column
comes straight from that displacement. If both are measured, the columns are
computed as ``(d_+ − d_−) / 2`` which averages out drift and doubles S/N.

The fit is the RAW inverse: the measured world→screen-up response ``S`` is
inverted directly (``M_sw ∝ S⁻¹``) and only a SINGLE overall scalar is
applied (largest column norm of ``M_sw`` → 1.0). Per-column normalization
would silently distort directions on any rig whose response is skewed or
anisotropic — commanded motion would come out along ``S·D·S⁻¹·u`` instead
of ``u``. Because relative column magnitudes now matter, **all four pulses
must use the same magnitude, roll frequency, and hold duration** (the panel
records each pulse's parameters and warns if they differ; displacements are
divided by freq×duration so hold-length changes are compensated).

``JoystickBridge`` and ``PathFollowController`` both apply the saved matrix
to screen-up vectors, so one calibration run fixes stick driving and
autonomous path following together.

Reuses the existing PathFollowController tracker (template or neural — same
knob as the Tracker tab) so no new tracking code lives here.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from classes.motion_controller import Mode, Source, roll_axis_for


# Pulse recipes. Rolling is what locomotes a paramagnetic bead and is what
# the direction matrix (calibration.screen_to_world_2x2) is fitted from.
# Static measures Mode B pull, which is used ONLY for its speed scale — the
# direction map is shared, since it is just the camera-vs-coil frame
# rotation and both modes see the same one.
ROLLING = "rolling"
STATIC = "static"

# Static pull speed vs drive magnitude. For a paramagnetic bead F ∝ ∇|B|²
# and B ∝ duty, so F ∝ mag²; the regime is overdamped (Stokes drag, no
# inertia at this scale), so terminal velocity ∝ F ∝ mag². This is a real
# difference from rolling, where speed tracks frequency and is roughly
# magnitude-independent once above the rolling threshold.
STATIC_MAG_EXPONENT = 2.0

_DIR_VECTORS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


# ---- pure fit math (no Qt — unit-testable) ----------------------------


def _measurement_delta(m) -> np.ndarray:
    """Raw-image displacement of one measurement, normalized by the pulse
    gain so hold-length (and, for rolling, frequency) differences compensate.

    Rolling divides by ``freq × dur``: a rolling bead advances one step per
    field revolution, so distance scales with revolutions.

    Static divides by ``dur`` alone — there is no frequency in a Mode B
    pulse. The bead drifts at whatever terminal velocity the pull produces,
    so distance scales with time only and the result is px/s. Magnitude is
    deliberately NOT divided out (see ``STATIC_MAG_EXPONENT``): it is
    recorded and all four pulses are expected to share it.

    Tuple form passes through — the common gain cancels in the final fit
    anyway.
    """
    if isinstance(m, dict):
        d = np.asarray(m["d"], dtype=float)
        gain = float(m.get("dur", 1.0))
        if m.get("recipe", ROLLING) != STATIC:
            gain *= float(m.get("freq", 1.0))
        return d / gain if gain > 1e-12 else d
    return np.asarray(m, dtype=float)


def axis_column(dp, dm) -> Optional[np.ndarray]:
    """World→(image-px) displacement column for one axis.

    ``(d+ - d-) / 2`` when both directions are measured (averages out drift
    and doubles S/N); one-sided fallback; None if neither is measured.
    """
    if dp is not None and dm is not None:
        return (_measurement_delta(dp) - _measurement_delta(dm)) / 2.0
    if dp is not None:
        return _measurement_delta(dp)
    if dm is not None:
        return -_measurement_delta(dm)
    return None


def build_screen_columns(displacements) -> Optional[np.ndarray]:
    """RAW world→screen-up 2×2 response matrix. Columns NOT normalized.

    Image-y grows down, so each measured displacement's y component is
    negated to reach the operator's screen-up frame. Returns None if either
    axis lacks any measurement.
    """
    cx = axis_column(displacements.get("+x"), displacements.get("-x"))
    cy = axis_column(displacements.get("+y"), displacements.get("-y"))
    if cx is None or cy is None:
        return None
    flip = np.array([1.0, -1.0])
    return np.column_stack([cx * flip, cy * flip])


def fit_quality(S: np.ndarray) -> dict:
    """Conditioning / geometry report for a raw response matrix.

    ``sin_sep = |det| / (‖cx‖·‖cy‖)`` is the scale-free column separation —
    an absolute det threshold is meaningless on raw-pixel columns.
    """
    nx = float(np.linalg.norm(S[:, 0]))
    ny = float(np.linalg.norm(S[:, 1]))
    q = {"col_norms": (nx, ny), "ok": True, "reason": ""}
    if nx < 1e-9 or ny < 1e-9:
        axis = "±X" if nx < 1e-9 else "±Y"
        q.update(ok=False,
                 reason=f"world {axis} produced no measurable displacement",
                 det=0.0, sin_sep=0.0, angle_deg=0.0, mirrored=False,
                 anisotropy=float("inf"))
        return q
    det = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
    cosv = float(np.dot(S[:, 0], S[:, 1])) / (nx * ny)
    sin_sep = abs(det) / (nx * ny)
    q["det"] = det
    q["sin_sep"] = sin_sep
    q["angle_deg"] = float(np.degrees(np.arctan2(sin_sep, cosv)))
    q["mirrored"] = det < 0.0
    q["anisotropy"] = max(nx, ny) / min(nx, ny)
    if sin_sep < 0.05:
        q["ok"] = False
        q["reason"] = (
            f"axis responses nearly collinear "
            f"(separation {q['angle_deg']:.1f}°) — re-measure")
    return q


def fit_screen_to_world(displacements):
    """Fit ``calibration.screen_to_world_2x2`` from the measurements.

    Returns ``(M_sw or None, quality dict)``. ``M_sw = inv(S)`` scaled so
    its largest column norm is 1.0 — a single overall scalar, so commanded
    world direction is exactly ``S⁻¹·stick`` for every stick direction
    (direction-true even on skewed / anisotropic / mirrored rigs), and full
    stick along the stiffer screen axis maps to full magnitude. Invariant
    to the common pulse gain (freq × duration × px scale).
    """
    S = build_screen_columns(displacements)
    if S is None:
        return None, {"ok": False, "reason": "incomplete measurements"}
    q = fit_quality(S)
    q["S"] = S
    if not q["ok"]:
        return None, q
    try:
        M = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        q["ok"] = False
        q["reason"] = "singular response matrix"
        return None, q
    s = max(float(np.linalg.norm(M[:, 0])), float(np.linalg.norm(M[:, 1])))
    return (M / s if s > 1e-12 else M), q


class FrameCalibrationPanel(QtWidgets.QWidget):
    """Tab that measures the world→screen frame transform via bead motion."""

    def __init__(self, path_follow, motion, config, joystick_bridge=None,
                 log_fn=print, parent=None):
        super().__init__(parent)
        self.path_follow = path_follow
        self.motion = motion
        self.config = config
        self.joystick_bridge = joystick_bridge
        self._joy_was_enabled: Optional[bool] = None
        self.log = log_fn

        # Per-direction screen displacement (image px, y grows down) over
        # one hold. Keys are "+x", "-x", "+y", "-y".
        #
        # Kept PER RECIPE: rolling and static measurements have different
        # units (px per rev vs px/s) and different physics, so mixing them
        # into one fit would be meaningless. Switching the selector swaps
        # which set is being filled and displayed.
        self._by_recipe: Dict[str, Dict[str, dict]] = {
            ROLLING: {}, STATIC: {}}
        self._recipe: str = ROLLING

        # Hold state machine.
        self._hold_dir: Optional[str] = None
        self._hold_p0: Optional[Tuple[float, float]] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        title = QtWidgets.QLabel("Frame Calibration — align stick to screen")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # --- Setup card --------------------------------------------------
        self.setup_card = QtWidgets.QGroupBox("1. Track the bead")
        s_lay = QtWidgets.QVBoxLayout(self.setup_card)
        s_lay.addWidget(QtWidgets.QLabel(
            "Left-click a stationary bead in the video. The Tracker's robot\n"
            "backend will lock onto it (template by default; neural if you\n"
            "set tracker.robot_tracker_backend = adaptive)."
        ))
        self.status_label = QtWidgets.QLabel("Tracking: no")
        self.status_label.setStyleSheet("font-family: monospace;")
        s_lay.addWidget(self.status_label)
        layout.addWidget(self.setup_card)

        # --- Measure card ------------------------------------------------
        self.measure_card = QtWidgets.QGroupBox("2. Hold each direction")
        m_lay = QtWidgets.QVBoxLayout(self.measure_card)
        m_lay.addWidget(QtWidgets.QLabel(
            "For each of the four horizontal world directions: press the\n"
            "button, coils hold that direction, panel records the tracker's\n"
            "displacement, then cuts. Bz (into/out of focus) is skipped —\n"
            "invisible on a top-down camera."
        ))

        recipe_row = QtWidgets.QHBoxLayout()
        recipe_row.addWidget(QtWidgets.QLabel("Recipe:"))
        self.recipe_rolling = QtWidgets.QRadioButton("Rolling (Mode A)")
        self.recipe_static = QtWidgets.QRadioButton("Static pull (Mode B)")
        self.recipe_rolling.setChecked(True)
        self.recipe_rolling.toggled.connect(self._on_recipe_changed)
        recipe_row.addWidget(self.recipe_rolling)
        recipe_row.addWidget(self.recipe_static)
        recipe_row.addStretch(1)
        m_lay.addLayout(recipe_row)
        self.recipe_note = QtWidgets.QLabel()
        self.recipe_note.setWordWrap(True)
        self.recipe_note.setStyleSheet("color: #888;")
        m_lay.addWidget(self.recipe_note)

        params_row = QtWidgets.QHBoxLayout()
        params_row.addWidget(QtWidgets.QLabel("Hold:"))
        self.dur_spin = QtWidgets.QDoubleSpinBox()
        self.dur_spin.setRange(0.2, 10.0)
        self.dur_spin.setSingleStep(0.1)
        self.dur_spin.setDecimals(1)
        self.dur_spin.setSuffix(" s")
        self.dur_spin.setValue(3.0)
        params_row.addWidget(self.dur_spin)
        # Magnitude + roll frequency are the shared knobs owned by the
        # Joystick tab — shown here read-only.
        self.shared_params = QtWidgets.QLabel("")
        self.shared_params.setStyleSheet("font-family: monospace;")
        params_row.addWidget(self.shared_params)
        params_row.addStretch(1)
        m_lay.addLayout(params_row)
        self._refresh_shared_params()
        try:
            config.on_change(
                "modes.mode_a", lambda *_a: self._refresh_shared_params())
        except Exception:
            pass

        # 4 buttons in a 2×2 grid so ±X and ±Y sit next to each other.
        grid = QtWidgets.QGridLayout()
        self.dir_buttons: Dict[str, QtWidgets.QPushButton] = {}
        for row, (plus_key, minus_key, label_plus, label_minus) in enumerate([
            ("+x", "-x", "Hold world +X", "Hold world −X"),
            ("+y", "-y", "Hold world +Y", "Hold world −Y"),
        ]):
            for col, (key, label) in enumerate([
                (plus_key, label_plus), (minus_key, label_minus),
            ]):
                b = QtWidgets.QPushButton(label)
                b.clicked.connect(lambda _c, k=key: self._start_hold(k))
                grid.addWidget(b, row, col)
                self.dir_buttons[key] = b
        m_lay.addLayout(grid)

        self.result_label = QtWidgets.QLabel(self._result_text())
        self.result_label.setStyleSheet("font-family: monospace;")
        m_lay.addWidget(self.result_label)

        clear_row = QtWidgets.QHBoxLayout()
        self.clear_btn = QtWidgets.QPushButton("Clear all measurements")
        self.clear_btn.clicked.connect(self._clear_measurements)
        clear_row.addWidget(self.clear_btn)
        clear_row.addStretch(1)
        m_lay.addLayout(clear_row)

        layout.addWidget(self.measure_card)

        # --- Apply card --------------------------------------------------
        self.apply_card = QtWidgets.QGroupBox("3. Fit and save")
        a_lay = QtWidgets.QVBoxLayout(self.apply_card)
        self.matrix_label = QtWidgets.QLabel(
            "Fitted matrix appears once at least one direction per axis is measured.")
        self.matrix_label.setStyleSheet("font-family: monospace;")
        a_lay.addWidget(self.matrix_label)
        self.quality_label = QtWidgets.QLabel("")
        self.quality_label.setStyleSheet("font-family: monospace; color: #888;")
        self.quality_label.setWordWrap(True)
        a_lay.addWidget(self.quality_label)
        self.apply_btn = QtWidgets.QPushButton("Apply and save")
        self.apply_btn.clicked.connect(self._apply)
        a_lay.addWidget(self.apply_btn)
        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet("color: red;")
        a_lay.addWidget(self.error_label)
        layout.addWidget(self.apply_card)

        layout.addStretch(1)

        # Poll tracker status at 10 Hz so the operator sees when the click
        # locked on.
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setInterval(100)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()
        self._on_recipe_changed()
        self._refresh_status()

    # ---- recipe ---------------------------------------------------------

    def _displacements(self) -> Dict[str, dict]:
        """Measurements for the recipe currently selected."""
        return self._by_recipe[self._recipe]

    def _on_recipe_changed(self, *_a) -> None:
        self._recipe = (ROLLING if self.recipe_rolling.isChecked()
                        else STATIC)
        if self._recipe == STATIC:
            self.recipe_note.setText(
                "Static measures Mode B pull speed only. It does NOT touch "
                "the direction matrix — that is shared with rolling, because "
                "it is just the camera-vs-coil frame rotation and both modes "
                "see the same one. Saves calibration.static_response_2x2, "
                "which gives the tracker a velocity model while you drive "
                "static. Expect small displacements: static pull is weak."
            )
            self.apply_btn.setText("Apply and save (static response)")
        else:
            self.recipe_note.setText(
                "Rolling is the calibration that matters for driving. It "
                "fits calibration.screen_to_world_2x2 (direction, shared by "
                "the joystick, path follower and static pull) plus the "
                "rolling speed scale."
            )
            self.apply_btn.setText("Apply and save")
        self.error_label.setText("")
        self._refresh_result()
        self._refresh_status()

    # ---- status polling ------------------------------------------------

    def _shared_mag_freq(self) -> Tuple[float, float]:
        mag = float(np.clip(
            self.config.get("modes.mode_a.magnitude_default", 1.0), 0.0, 1.0))
        freq = float(self.config.get("modes.mode_a.freq_default", 1.0))
        return mag, freq

    def _refresh_shared_params(self) -> None:
        mag, freq = self._shared_mag_freq()
        text = f"Uses mag {mag:.2f} @ {freq:.1f} Hz — set on the Joystick tab"
        warn = ""
        if freq < 2.0:
            warn = ("  ⚠ low roll frequency — the bead may not move "
                    "measurably; raise Base roll frequency on the Joystick tab")
        elif mag < 0.5:
            warn = "  ⚠ low magnitude — pulses will be weak"
        self.shared_params.setText(text + warn)
        self.shared_params.setStyleSheet(
            "font-family: monospace;"
            + (" color: #e08a3c;" if warn else " color: #888;"))

    def _tracked_pos(self) -> Optional[Tuple[float, float]]:
        r = getattr(self.path_follow, "robot", None)
        if r is None:
            return None
        return r.last_pos

    def _refresh_status(self) -> None:
        pos = self._tracked_pos()
        if pos is None:
            self.status_label.setText(
                "Tracking: no — left-click the bead in the video")
            for b in self.dir_buttons.values():
                b.setEnabled(False)
        else:
            self.status_label.setText(
                f"Tracking: yes  ({pos[0]:6.1f}, {pos[1]:6.1f}) px")
            live = self._hold_dir is None
            for b in self.dir_buttons.values():
                b.setEnabled(live)

    # ---- hold state machine -------------------------------------------

    def _start_hold(self, direction: str) -> None:
        if getattr(self.path_follow, "running", False):
            self.error_label.setStyleSheet("color: red;")
            self.error_label.setText(
                "Stop path following before calibrating.")
            return
        p0 = self._tracked_pos()
        if p0 is None:
            self.error_label.setText("Cannot measure — no tracked bead.")
            return
        self.error_label.setText("")
        self._hold_dir = direction
        self._hold_p0 = p0
        mag, freq = self._shared_mag_freq()
        self._hold_mag = mag
        self._hold_freq = freq
        self._hold_dur = float(self.dur_spin.value())
        dur_ms = int(self._hold_dur * 1000)
        vec = _DIR_VECTORS[direction]

        # A drifting joystick stick past its deadzone would overwrite our
        # direction/magnitude at 33 Hz mid-pulse. Disable, restore on end.
        if self.joystick_bridge is not None:
            self._joy_was_enabled = self.joystick_bridge.enabled
            self.joystick_bridge.set_enabled(False)

        for b in self.dir_buttons.values():
            b.setEnabled(False)

        self.motion.set_static_direction(vec, Source.ALGORITHM)
        self.motion.set_magnitude(mag, Source.ALGORITHM)
        if self._recipe == STATIC:
            self.log(
                f"FrameCalibration: static pull {direction} at "
                f"mag={mag:.2f} for {dur_ms/1000:.1f}s, start pos={p0}")
            # Mode B ignores roll axis and frequency entirely
            # (motion_controller.tick: B_des = magnitude · direction).
            self.motion.set_mode(Mode.B_STATIC)
        else:
            self.log(
                f"FrameCalibration: rolling {direction} at mag={mag:.2f} "
                f"freq={freq:.1f}Hz for {dur_ms/1000:.1f}s, start pos={p0}")
            # Rolling recipe — same as the joystick: the bead rolls along vec.
            self.motion.set_roll_axis(roll_axis_for(vec[0], vec[1]),
                                      Source.ALGORITHM)
            self.motion.set_frequency(freq, Source.ALGORITHM)
            self.motion.set_mode(Mode.A_ROTATING)
        QtCore.QTimer.singleShot(dur_ms, self._end_hold)

    def _end_hold(self) -> None:
        p1 = self._tracked_pos()
        self.motion.set_mode(Mode.OFF)
        self._restore_joystick()

        p0 = self._hold_p0
        direction = self._hold_dir
        self._hold_p0 = None
        self._hold_dir = None

        if p0 is None or p1 is None:
            self.error_label.setText(
                "Measurement failed — tracker lost the bead mid-hold.")
            self._refresh_status()
            return

        # Raw displacement over the hold, recorded alongside the pulse
        # parameters — the fit divides by freq×duration and warns when
        # magnitude differed between pulses (relative column scale matters
        # for the raw-inverse fit).
        self._displacements()[direction] = {
            "d": (p1[0] - p0[0], p1[1] - p0[1]),
            "mag": self._hold_mag,
            "freq": self._hold_freq,
            "dur": self._hold_dur,
            "recipe": self._recipe,
        }

        self._refresh_result()
        self._refresh_status()

    def _restore_joystick(self) -> None:
        if self.joystick_bridge is not None and \
                self._joy_was_enabled is not None:
            try:
                self.joystick_bridge.set_enabled(self._joy_was_enabled)
            except Exception:
                pass
            self._joy_was_enabled = None

    def _clear_measurements(self) -> None:
        self._displacements().clear()
        self._refresh_result()
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setText("")

    # ---- fit + save ---------------------------------------------------

    def _result_text(self) -> str:
        lines = []
        for key in ("+x", "-x", "+y", "-y"):
            v = self._displacements().get(key)
            if v is None:
                lines.append(f"  {key}: (unmeasured)")
            else:
                d = v["d"] if isinstance(v, dict) else v
                lines.append(f"  {key}: Δpx={d[0]:+.2f}  Δpy={d[1]:+.2f}")
        return "\n".join(lines)

    def _quality_text(self, q: dict) -> Tuple[str, str]:
        """(text, css color) summary of the fit geometry."""
        angle = q.get("angle_deg", 0.0)
        acute = min(angle, 180.0 - angle)
        mirror = ("mirrored (left-handed) — expected on this rig, the "
                  "matrix corrects it" if q.get("mirrored")
                  else "right-handed")
        aniso = q.get("anisotropy", 1.0)
        text = (f"Column separation: {angle:.1f}° · det {q.get('det', 0.0):+.2f} "
                f"({mirror}) · anisotropy {aniso:.2f}×")
        color = "#888"
        if acute < 30.0:
            text += "  — nearly collinear, re-measure"
            color = "#d64"
        elif acute < 55.0:
            text += ("  — columns far from perpendicular; the fit still "
                     "corrects it, but longer holds will be more accurate")
            color = "#e08a3c"
        if aniso > 3.0:
            text += ("  — response very unequal between world axes; "
                     "magnitude will feel direction-dependent")
            color = "#e08a3c" if color == "#888" else color
        mags = {round(float(v.get("mag", 0.0)), 3)
                for v in self._displacements().values() if isinstance(v, dict)}
        if len(mags) > 1:
            text += ("  — ⚠ measurements used different magnitudes; "
                     "relative column scale is meaningful, re-measure")
            color = "#e08a3c" if color == "#888" else color
        return text, color

    def _refresh_result(self) -> None:
        self.result_label.setText(self._result_text())

        M_sw, q = fit_screen_to_world(self._displacements())
        if M_sw is None:
            self.matrix_label.setText(
                q.get("reason") or
                "Fitted matrix appears once at least one direction per axis is measured.")
            self.quality_label.setText("")
            return

        S = q["S"]
        self.matrix_label.setText(
            "S (measured screen response, px per unit hold):\n"
            f"  [[{S[0,0]:+.2f}, {S[0,1]:+.2f}],\n"
            f"   [{S[1,0]:+.2f}, {S[1,1]:+.2f}]]\n\n"
            "M_sw (screen-up → world, applied by joystick + follower):\n"
            f"  [[{M_sw[0,0]:+.3f}, {M_sw[0,1]:+.3f}],\n"
            f"   [{M_sw[1,0]:+.3f}, {M_sw[1,1]:+.3f}]]"
        )
        text, color = self._quality_text(q)
        self.quality_label.setText(text)
        self.quality_label.setStyleSheet(
            f"font-family: monospace; color: {color};")

    def _apply(self) -> None:
        M_sw, q = fit_screen_to_world(self._displacements())
        if M_sw is None:
            self.error_label.setStyleSheet("color: red;")
            self.error_label.setText(
                q.get("reason") or
                "Measure at least one direction per axis (+X or -X, and +Y or -Y).")
            return
        if self._recipe == STATIC:
            self._apply_static(M_sw, q)
            return
        self.config.set("calibration.screen_to_world_2x2", M_sw.tolist())
        # Also persist the RAW response matrix. M_sw above is normalized to
        # direction-only (largest column norm → 1.0), which deliberately
        # throws away the pixel scale. S keeps it: px of travel per unit
        # world command per (Hz × second), since _measurement_delta divides
        # each pulse by freq × dur. That makes it the one calibrated source
        # for "how fast should this bead be moving right now", which the
        # tracker's motion filter uses as its control input.
        S = q.get("S")
        if S is not None:
            self.config.set("calibration.screen_response_2x2",
                            np.asarray(S, dtype=float).tolist())
        self.config.save()
        self.error_label.setStyleSheet("color: green;")
        self.error_label.setText(
            f"Saved. Separation {q['angle_deg']:.1f}°, "
            f"{'mirrored' if q['mirrored'] else 'right-handed'} — joystick "
            f"and path follower now apply the transform.")
        self.log(
            f"FrameCalibration: saved calibration.screen_to_world_2x2 = "
            f"{M_sw.tolist()} (quality: {q['angle_deg']:.1f}°, "
            f"det {q['det']:+.3f})")

    def _apply_static(self, M_sw: np.ndarray, q: dict) -> None:
        """Save the Mode B speed scale only.

        Deliberately does NOT write ``screen_to_world_2x2`` or
        ``screen_response_2x2``: those belong to the rolling calibration and
        writing them here would have the two recipes overwriting each other.
        Direction is shared — it is the camera-vs-coil frame rotation, which
        does not depend on how the bead is being driven.
        """
        S = q.get("S")
        if S is None:
            self.error_label.setStyleSheet("color: red;")
            self.error_label.setText("No response matrix to save.")
            return
        mag, _freq = self._shared_mag_freq()
        mags = [float(v.get("mag", mag))
                for v in self._displacements().values() if isinstance(v, dict)]
        mag_cal = float(np.mean(mags)) if mags else mag
        self.config.set("calibration.static_response_2x2",
                        np.asarray(S, dtype=float).tolist())
        self.config.set("calibration.static_response_mag", mag_cal)
        self.config.save()

        # Free diagnostic: after the exact Mode B synthesis landed, static
        # and rolling should agree on direction. A large disagreement means
        # something else is off (mis-set Bmap, a moved coil) and is worth
        # surfacing rather than silently averaging away.
        agree = self._direction_agreement(M_sw)
        note = ""
        if agree is not None:
            note = f" Static vs rolling direction: {agree:.1f}°."
            if agree > 20.0:
                note += (" ⚠ that is large — expected near 0. Check Bmap / "
                         "re-run the rolling calibration.")
        self.error_label.setStyleSheet(
            "color: #e08a3c;" if (agree or 0.0) > 20.0 else "color: green;")
        self.error_label.setText(
            f"Saved static response (px/s per unit command at mag "
            f"{mag_cal:.2f}). The tracker now predicts bead velocity while "
            f"driving static.{note}")
        self.log(
            f"FrameCalibration: saved calibration.static_response_2x2 = "
            f"{np.asarray(S).tolist()} at mag {mag_cal:.2f}"
            + (f" (direction agreement {agree:.1f}°)"
               if agree is not None else ""))

    def _direction_agreement(self, M_static: np.ndarray) -> Optional[float]:
        """Mean angle between the static and saved-rolling direction maps."""
        try:
            M_roll = np.asarray(
                self.config.get("calibration.screen_to_world_2x2",
                                [[1.0, 0.0], [0.0, 1.0]]),
                dtype=float).reshape(2, 2)
        except Exception:
            return None
        angles = []
        for u in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            a, b = M_roll @ u, M_static @ u
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                return None
            c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            angles.append(np.rad2deg(np.arccos(c)))
        return float(np.mean(angles))

    # ---- lifecycle ----------------------------------------------------

    def hideEvent(self, event) -> None:
        # If the operator switches tabs mid-hold, kill the coils so nothing
        # runs unattended, and give the pad back.
        if self._hold_dir is not None:
            try:
                self.motion.set_mode(Mode.OFF)
            except Exception:
                pass
            self._hold_dir = None
        self._restore_joystick()
        super().hideEvent(event)
