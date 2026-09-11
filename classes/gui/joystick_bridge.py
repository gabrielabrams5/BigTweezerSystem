"""Joystick → MotionController shim.

Polls pygame at ~33 Hz on the Qt event loop and pushes commands into the
existing MotionController API (``Source.JOYSTICK``). No new plumbing on the
motion side — this is purely a translation layer:

  * Left stick + LT/RT → desired travel direction. In rolling mode the
    roll axis is recomputed every poll as horizontal ⊥ travel, so the field
    sweeps the vertical plane containing the travel vector and the bead
    rolls along the substrate toward the stick.
  * Right-stick X → signed rolling frequency: centered = base
    (``modes.mode_a.freq_default``), right = faster (up to
    ``modes.mode_a.freq_max_hz``), left = same curve, reversed.
  * Buttons (edge-triggered) → mode toggle (A rolling ↔ B static), magnitude
    ±0.1, e-stop, reset.

Default behavior: sticks alive → Mode.A_ROTATING (rolling toward the
stick), or Mode.B_STATIC when ``joystick.stick_mode`` is ``"static"``
(switchable live from the Joystick tab). Operator can also toggle modes
via btn_mode_a / btn_mode_b.

Cross-platform: axis indices default from ``platform.system()``; per-rig
overrides come from ``config.yaml`` under ``joystick.*``. pygame is imported
inside :meth:`JoystickBridge.start` so machines without pygame still boot
the app (same pattern used by ``classes/aravis_camera.py``).
"""

from __future__ import annotations

import math
import platform
from typing import Callable

import numpy as np
from PyQt5 import QtCore

from classes.motion_controller import (
    Mode,
    MotionController,
    Source,
    roll_axis_for,
)
from classes.gui.view_transform import stick_display_to_raw


def _os_axis_defaults() -> dict:
    """Axis indices for the current OS. Linux differs from Mac/Windows."""
    if platform.system() == "Linux":
        return dict(axis_left_x=0, axis_left_y=1,
                    axis_right_x=3, axis_right_y=4,
                    axis_lt=2, axis_rt=5)
    return dict(axis_left_x=0, axis_left_y=1,
                axis_right_x=2, axis_right_y=3,
                axis_lt=4, axis_rt=5)


def deadzone(v: float, thr: float) -> float:
    return 0.0 if abs(v) < thr else float(v)


class JoystickBridge(QtCore.QObject):

    connectionChanged = QtCore.pyqtSignal(str)
    axesChanged = QtCore.pyqtSignal(dict)      # {axis_label: float, ...}
    buttonsChanged = QtCore.pyqtSignal(list)   # list[bool] length = n_buttons
    enabledChanged = QtCore.pyqtSignal(bool)

    def __init__(self,
                 motion: MotionController,
                 supervisor,
                 config,
                 log_fn: Callable[[str], None] = print,
                 parent=None):
        super().__init__(parent)
        self.motion = motion
        self.supervisor = supervisor
        self.config = config
        self.log = log_fn

        self._pygame = None
        self._joy = None
        self._enabled = bool(config.get("joystick.enabled", True))

        self.deadzone_thr = float(config.get("joystick.deadzone", 0.15))
        self.poll_hz = float(config.get("joystick.poll_hz", 33))
        # Rolling frequency range. Right-stick-X is signed: centered → base,
        # full right → freq_max_hz, full left → -freq_max_hz.
        self.freq_base_hz = float(config.get("modes.mode_a.freq_default", 1.0))
        self.freq_max_hz = float(config.get("modes.mode_a.freq_max_hz", 20.0))
        # Shared master gain — full stick deflection commands this magnitude.
        self.mag_base = float(config.get("modes.mode_a.magnitude_default", 1.0))
        # Which mode grabbing the left stick engages: "rolling" | "static".
        self._stick_mode = str(config.get("joystick.stick_mode", "rolling"))
        # Operator-preference axis flips, applied to the display-frame
        # stick intent before rotation/calibration.
        self._invert_x = bool(config.get("joystick.invert_x", False))
        self._invert_y = bool(config.get("joystick.invert_y", False))

        d = _os_axis_defaults()
        self.axis_left_x = int(config.get("joystick.axis_left_x", d["axis_left_x"]))
        self.axis_left_y = int(config.get("joystick.axis_left_y", d["axis_left_y"]))
        self.axis_right_x = int(config.get("joystick.axis_right_x", d["axis_right_x"]))
        self.axis_right_y = int(config.get("joystick.axis_right_y", d["axis_right_y"]))
        self.axis_lt = int(config.get("joystick.axis_lt", d["axis_lt"]))
        self.axis_rt = int(config.get("joystick.axis_rt", d["axis_rt"]))

        # Screen→world 2×2 for the (bx, by) horizontal command. Populated
        # by the Frame Calibration tab. Identity default = no remap.
        self._reload_frame_matrix()
        self._reload_view_rotation()
        try:
            config.on_change(
                "calibration.screen_to_world_2x2",
                lambda *_a: self._reload_frame_matrix())
            config.on_change(
                "camera.view_rotation_deg",
                lambda *_a: self._reload_view_rotation())
            config.on_change(
                "joystick.stick_mode",
                lambda *_a: setattr(
                    self, "_stick_mode",
                    str(self.config.get("joystick.stick_mode", "rolling"))))
            config.on_change(
                "modes.mode_a.freq_default",
                lambda *_a: setattr(
                    self, "freq_base_hz",
                    float(self.config.get("modes.mode_a.freq_default", 1.0))))
            config.on_change(
                "modes.mode_a.magnitude_default",
                lambda *_a: setattr(
                    self, "mag_base",
                    float(self.config.get(
                        "modes.mode_a.magnitude_default", 1.0))))
            config.on_change(
                "joystick.invert_x",
                lambda *_a: setattr(
                    self, "_invert_x",
                    bool(self.config.get("joystick.invert_x", False))))
            config.on_change(
                "joystick.invert_y",
                lambda *_a: setattr(
                    self, "_invert_y",
                    bool(self.config.get("joystick.invert_y", False))))
        except Exception:
            # Older Config without on_change — safe to ignore; these just
            # require a restart to update.
            pass

        btn = config.get("joystick.buttons", {}) or {}
        self.btn_mode_b = int(btn.get("mode_b", 0))
        self.btn_mode_a = int(btn.get("mode_a", 1))
        self.btn_mag_down = int(btn.get("mag_down", 2))
        self.btn_mag_up = int(btn.get("mag_up", 3))
        self.btn_estop = int(btn.get("estop", 6))
        self.btn_reset = int(btn.get("reset_estop", 7))

        self._prev_buttons: list = []

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000 / max(1.0, self.poll_hz)))
        self._timer.timeout.connect(self._poll)

    # ---- lifecycle ---------------------------------------------------

    def start(self) -> None:
        if not self._enabled:
            self.connectionChanged.emit("disabled")
            return
        try:
            import pygame  # noqa: PLC0415
        except Exception as e:
            self.log(f"JoystickBridge: pygame import failed: {e}")
            self.connectionChanged.emit("pygame missing")
            return
        self._pygame = pygame
        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            self.log(f"JoystickBridge: pygame init failed: {e}")
            self.connectionChanged.emit("init failed")
            return

        n = pygame.joystick.get_count()
        if n == 0:
            self.log("JoystickBridge: no joystick detected")
            self.connectionChanged.emit("no joystick")
            return

        try:
            joy = pygame.joystick.Joystick(0)
            joy.init()
            self._joy = joy
            name = joy.get_name()
            self.log(f"JoystickBridge: connected — {name}")
            self.connectionChanged.emit(f"connected: {name}")
            self._prev_buttons = [False] * joy.get_numbuttons()
            self._timer.start()
        except Exception as e:
            self.log(f"JoystickBridge: joystick init failed: {e}")
            self.connectionChanged.emit(f"init failed: {e}")

    def stop(self) -> None:
        self._timer.stop()
        if self._joy is not None:
            try:
                self._joy.quit()
            except Exception:
                pass
            self._joy = None
        if self._pygame is not None:
            try:
                self._pygame.joystick.quit()
                self._pygame.quit()
            except Exception:
                pass
            self._pygame = None
        self.connectionChanged.emit("disconnected")

    def reconnect(self) -> None:
        self.stop()
        self.start()

    def _reload_frame_matrix(self) -> None:
        raw = self.config.get(
            "calibration.screen_to_world_2x2", [[1.0, 0.0], [0.0, 1.0]])
        try:
            M = np.asarray(raw, dtype=float).reshape(2, 2)
        except Exception:
            M = np.eye(2)
        self._screen_to_world_2x2 = M

    def _reload_view_rotation(self) -> None:
        # Stick intent is in the DISPLAYED view's frame; the calibration
        # matrix expects the raw camera frame. Compose the view rotation
        # so stick-up always means displayed-up.
        try:
            deg = int(self.config.get("camera.view_rotation_deg", 0)) % 360
        except Exception:
            deg = 0
        if deg not in (0, 90, 180, 270):
            deg = 0
        self._stick_view_2x2 = np.asarray(stick_display_to_raw(deg),
                                          dtype=float)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)
        self.enabledChanged.emit(self._enabled)
        if not on:
            # If the operator toggles the joystick off while a stick is held,
            # cancel whatever mode we forced on.
            self.motion.set_mode(Mode.OFF)

    # ---- poll --------------------------------------------------------

    def _poll(self) -> None:
        pg = self._pygame
        joy = self._joy
        if pg is None or joy is None:
            return
        # Drain events so pygame's internal state stays fresh. We ignore
        # events themselves — get_axis / get_button read the latched state.
        try:
            pg.event.pump()
        except Exception:
            return

        # ---- Axes readout, with per-axis bounds check ------------------
        n_axes = joy.get_numaxes()

        def ax(i: int, default: float = 0.0) -> float:
            if i < 0 or i >= n_axes:
                return default
            try:
                return float(joy.get_axis(i))
            except Exception:
                return default

        raw_lx = ax(self.axis_left_x)
        raw_ly = ax(self.axis_left_y)
        raw_rx = ax(self.axis_right_x)
        raw_ry = ax(self.axis_right_y)
        # Triggers rest at -1 on SDL for both LT and RT by default.
        raw_lt = ax(self.axis_lt, default=-1.0)
        raw_rt = ax(self.axis_rt, default=-1.0)

        lx = deadzone(raw_lx, self.deadzone_thr)
        ly = deadzone(raw_ly, self.deadzone_thr)
        rx = deadzone(raw_rx, self.deadzone_thr)
        ry = deadzone(raw_ry, self.deadzone_thr)
        # No deadzone on triggers — the rest-value is well-defined at -1.
        lt = (raw_lt + 1.0) / 2.0    # [0, 1]
        rt = (raw_rt + 1.0) / 2.0    # [0, 1]

        self.axesChanged.emit({
            "left_x": lx, "left_y": ly,
            "right_x": rx, "right_y": ry,
            "lt": lt, "rt": rt,
        })

        # ---- Buttons, edge-detected -----------------------------------
        n_btn = joy.get_numbuttons()
        buttons = [bool(joy.get_button(i)) for i in range(n_btn)]
        if len(self._prev_buttons) != n_btn:
            self._prev_buttons = [False] * n_btn
        pressed = [(now and not was)
                   for now, was in zip(buttons, self._prev_buttons)]
        self._prev_buttons = buttons
        self.buttonsChanged.emit(buttons)

        if self._enabled:
            self._handle_buttons(pressed)
            self._handle_axes(lx, ly, rx, ry, lt, rt)

    # ---- axis → motion --------------------------------------------------

    def _handle_axes(self, lx: float, ly: float, rx: float, ry: float,
                     lt: float, rt: float) -> None:
        # Left stick → operator screen-frame intent (right = +sx, up = +sy).
        # Invert flags flip this display-frame intent — before rotation and
        # calibration — so "Invert X" always flips screen-left/right exactly
        # as the operator sees it. The matrices below then map into world
        # axes so the bead moves where the stick points.
        sx = lx * (-1.0 if self._invert_x else 1.0)
        sy = (-ly) * (-1.0 if self._invert_y else 1.0)
        # Display-up frame → raw-image-up frame (view rotation) → world.
        R = self._stick_view_2x2
        sx, sy = float(R[0, 0] * sx + R[0, 1] * sy), \
                 float(R[1, 0] * sx + R[1, 1] * sy)
        M = self._screen_to_world_2x2
        hx, hy = float(M[0, 0] * sx + M[0, 1] * sy), \
                 float(M[1, 0] * sx + M[1, 1] * sy)
        # The calibration matrix sets DIRECTION only. Its column magnitudes
        # encode measured rolling distances, but roll displacement scales
        # with freq × time, not linearly with |B| — letting them modulate
        # drive strength starves the rig's "weak" screen axis (low duty,
        # bead never breaks free). Magnitude comes from stick deflection.
        h = math.hypot(hx, hy)
        s_norm = math.hypot(sx, sy)   # rotation-invariant deflection
        if h > 1e-9:
            bx, by = hx / h * s_norm, hy / h * s_norm
        else:
            bx, by = 0.0, 0.0
        # Triggers → vertical (Bz). RT pulls +Z, LT pulls -Z.
        bz = rt - lt

        mag = min(1.0, self.mag_base * math.sqrt(bx * bx + by * by + bz * bz))
        cur_mode = self.motion.mode

        if mag < 0.05:
            # Sticks and triggers all at rest → tell motion we're idle.
            # Only override the mode if the joystick was the last writer,
            # so a GUI-driven mode isn't stomped when the operator releases
            # the sticks.
            if cur_mode != Mode.OFF and self.motion.source == Source.JOYSTICK:
                self.motion.set_mode(Mode.OFF)
            return

        # Non-idle stick / triggers. Update the base direction + magnitude;
        # both Mode A (rolling) and Mode B (static) use these.
        self.motion.set_static_direction([bx, by, bz], Source.JOYSTICK)
        self.motion.set_magnitude(mag, Source.JOYSTICK)

        # Roll axis: horizontal, perpendicular to travel (a = ẑ × t), so
        # the field sweeps the vertical plane containing the travel vector
        # and the bead rolls along the substrate toward the stick.
        # Triggers-only (below min_norm) → zero axis → static vertical pull.
        self.motion.set_roll_axis(roll_axis_for(bx, by, min_norm=0.05),
                                  Source.JOYSTICK)

        # Signed frequency: centered → base, right → faster, left → same
        # curve reversed. Base of 0 degenerates to a pure rx·freq_max map.
        span = max(0.0, self.freq_max_hz - self.freq_base_hz)
        if rx > 0:
            freq_hz = self.freq_base_hz + rx * span
        elif rx < 0:
            freq_hz = -(self.freq_base_hz + (-rx) * span)
        else:
            freq_hz = self.freq_base_hz
        self.motion.set_frequency(freq_hz, Source.JOYSTICK)

        # When sticks come alive, engage the operator-selected mode. If a
        # mode is already active (GUI or pad buttons) we respect it.
        if cur_mode == Mode.OFF:
            self.motion.set_mode(Mode.B_STATIC if self._stick_mode == "static"
                                 else Mode.A_ROTATING)

    # ---- buttons → motion / supervisor ----------------------------------

    def _handle_buttons(self, pressed: list) -> None:
        def hit(idx: int) -> bool:
            return 0 <= idx < len(pressed) and pressed[idx]

        if hit(self.btn_estop):
            self.supervisor.estop()
            return

        if hit(self.btn_reset):
            self.supervisor.reset_estop()

        if hit(self.btn_mode_b):
            new = (Mode.OFF if self.motion.mode == Mode.B_STATIC
                   else Mode.B_STATIC)
            self.motion.set_mode(new)

        if hit(self.btn_mode_a):
            new = (Mode.OFF if self.motion.mode == Mode.A_ROTATING
                   else Mode.A_ROTATING)
            self.motion.set_mode(new)

        if hit(self.btn_mag_up):
            self._bump_mag(+0.1)

        if hit(self.btn_mag_down):
            self._bump_mag(-0.1)

    def _bump_mag(self, delta: float) -> None:
        # Nudge the SHARED magnitude so the joystick, follower, Frame Cal
        # and Mode A tab all follow. config.set fires the on_change listener
        # that updates self.mag_base.
        v = float(np.clip(self.mag_base + delta, 0.0, 1.0))
        self.config.set("modes.mode_a.magnitude_default", v)
        try:
            self.config.save()
        except Exception:
            pass
        # Take effect immediately, even mid-hold.
        self.motion.set_magnitude(v, Source.JOYSTICK)
