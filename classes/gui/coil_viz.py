"""Live coil-firing + pull-direction visualization for the Joystick tab.

Two sub-panels drawn from scratch with :class:`QtGui.QPainter`:

* **Top-down (XY)** — six coil discs at azimuths 0/120/240° for the top ring
  and mirrored for the bottom ring. Fill intensity ∝ |current[j]|. Top-ring
  coils are filled discs, bottom-ring coils are ring outlines so the operator
  can tell them apart at a glance. A magenta arrow from the origin shows the
  projection of ``Bmap @ currents`` onto the XY plane — the direction the
  paramagnetic bead is being pulled toward.

* **Side (XZ)** — the same six coils drawn as their (x, z) projection, so top
  ring coils sit above the workspace line and bottom-ring coils below. Same
  intensity + magenta arrow convention along the (x, z) components of the
  field.

Repaints are throttled to ~30 Hz. ``update_state`` can be safely called from
the ~200 Hz inner loop; only every ~6th call schedules an actual repaint.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from classes.motion_controller import Mode


# ------------------------------------------------------------------ geometry
# Kept local to this widget by design. Matches classes/gui/calibration_wizard.py
# ::_GEOM_AXES conventionally: top ring 0/120/240° azimuth, 45° above
# horizontal, bottom ring same azimuths but 45° below.

_TILT_COS = math.cos(math.radians(45.0))  # ≈ 0.707
_TILT_SIN = math.sin(math.radians(45.0))  # ≈ 0.707 (equal by design)

_AZIMUTHS = [math.radians(a) for a in (0.0, 120.0, 240.0)]

# Order C1..C6, matching classes/field_solver.py.
_COIL_POS = np.column_stack([
    # Top ring: x = cos*tilt, y = sin*tilt, z = +tilt
    *(np.array([_TILT_COS * math.cos(a), _TILT_COS * math.sin(a),  _TILT_SIN])
      for a in _AZIMUTHS),
    # Bottom ring: same but z inverted
    *(np.array([_TILT_COS * math.cos(a), _TILT_COS * math.sin(a), -_TILT_SIN])
      for a in _AZIMUTHS),
])


# ------------------------------------------------------------------ helpers

def _lerp_color(t: float,
                cold: QtGui.QColor = QtGui.QColor("#333333"),
                hot: QtGui.QColor = QtGui.QColor("#00d4ff")) -> QtGui.QColor:
    """Colour from ``cold`` to ``hot`` as t goes 0 → 1."""
    t = max(0.0, min(1.0, float(t)))
    r = int(round(cold.red() + t * (hot.red() - cold.red())))
    g = int(round(cold.green() + t * (hot.green() - cold.green())))
    b = int(round(cold.blue() + t * (hot.blue() - cold.blue())))
    return QtGui.QColor(r, g, b)


# ------------------------------------------------------------------ panels

class _CoilPanel(QtWidgets.QWidget):
    """Single 2D projection panel. Two axes are chosen at construction:

    ``ax_x`` selects the coil-position dimension used for horizontal display
    (0, 1, or 2 → x, y, z). ``ax_y`` selects the vertical dimension. Same
    two axes are used for the projected field arrow.
    """

    def __init__(self,
                 title: str,
                 ax_x: int,
                 ax_y: int,
                 ring_marker_style: str = "auto",
                 xy_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 parent=None):
        super().__init__(parent)
        self.title = title
        self.ax_x = ax_x
        self.ax_y = ax_y
        self.setMinimumSize(180, 180)
        self._currents = np.zeros(6)
        self._field = np.zeros(3)
        # Optional 2D transform applied to (ax_x, ax_y) values before drawing.
        # Used by the top-down panel to route through the frame-calibration
        # matrix so the viz matches the microscope's screen frame.
        self._xy_transform = xy_transform

    def set_state(self, currents: np.ndarray, field: np.ndarray) -> None:
        self._currents = np.asarray(currents, dtype=float).reshape(6)
        self._field = np.asarray(field, dtype=float).reshape(3)

    def _project(self, vec3: np.ndarray) -> tuple:
        """Extract (ax_x, ax_y) components, optionally apply xy_transform."""
        vx = float(vec3[self.ax_x])
        vy = float(vec3[self.ax_y])
        if self._xy_transform is not None:
            out = self._xy_transform(np.array([vx, vy], dtype=float))
            return float(out[0]), float(out[1])
        return vx, vy

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()
        p.fillRect(rect, QtGui.QColor("#181818"))

        # Title
        p.setPen(QtGui.QColor("#bbbbbb"))
        p.setFont(QtGui.QFont(self.font().family(), 9, QtGui.QFont.Bold))
        p.drawText(rect.adjusted(6, 4, -6, 0),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, self.title)

        # Compute drawing area (leave room for title).
        pad = 24
        area = rect.adjusted(pad, pad, -pad, -pad)
        if area.width() <= 0 or area.height() <= 0:
            return
        cx = area.center().x()
        cy = area.center().y()
        radius = min(area.width(), area.height()) / 2.0

        # Faint workspace outline
        p.setPen(QtGui.QPen(QtGui.QColor("#333333"), 1, QtCore.Qt.DashLine))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        # Coils
        # Normalize duty for intensity mapping. Cap at 1.0 (post-saturation values).
        max_duty = max(1e-9, float(np.max(np.abs(self._currents))))
        for j in range(6):
            pos = _COIL_POS[:, j]
            sx, sy = self._project(pos)
            x = cx + radius * sx
            # Qt y grows downward — flip so up is "toward viewer" naturally.
            y = cy - radius * sy

            duty = float(self._currents[j])
            intensity = min(1.0, abs(duty) / max(1.0, max_duty))
            fill = _lerp_color(intensity)

            is_top_ring = j < 3
            coil_r = 12
            if is_top_ring:
                p.setBrush(fill)
                p.setPen(QtGui.QPen(QtGui.QColor("#8899aa"), 1))
                p.drawEllipse(QtCore.QPointF(x, y), coil_r, coil_r)
            else:
                # Bottom ring: ring outline with thick pen coloured by intensity.
                pen = QtGui.QPen(fill, 3)
                p.setPen(pen)
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(QtCore.QPointF(x, y), coil_r, coil_r)

            # Coil label
            p.setPen(QtGui.QColor("#dddddd"))
            p.setFont(QtGui.QFont(self.font().family(), 8))
            p.drawText(
                QtCore.QRectF(x - coil_r, y - coil_r,
                              2 * coil_r, 2 * coil_r),
                QtCore.Qt.AlignCenter,
                f"C{j + 1}",
            )

        # Origin dot
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#dddddd"))
        p.drawEllipse(QtCore.QPointF(cx, cy), 3, 3)

        # Field arrow: magenta. Projected + optionally transformed to match
        # the top-down panel's camera-frame layout.
        fx, fy = self._project(self._field)
        fmag = math.hypot(fx, fy)
        if fmag > 1e-9:
            # Scale arrow so full-magnitude field reaches ~85% of radius.
            # `_field` is in "Bmap-units × duty" which for calibrated Bmaps
            # (mT / unit-duty) can be tens of mT. Normalize by the largest
            # component of the current field.
            arrow_scale = 0.85 * radius / max(1e-9, fmag)
            end_x = cx + fx * arrow_scale
            end_y = cy - fy * arrow_scale  # flip so up is +axis in world

            arrow_color = QtGui.QColor("#ff44dd")
            pen = QtGui.QPen(arrow_color, 3)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(end_x, end_y))

            # Arrowhead: two short lines rotated ±30° from the shaft.
            angle = math.atan2(end_y - cy, end_x - cx)
            head_len = 10
            for da in (math.radians(150), math.radians(-150)):
                hx = end_x + head_len * math.cos(angle + da)
                hy = end_y + head_len * math.sin(angle + da)
                p.drawLine(QtCore.QPointF(end_x, end_y),
                           QtCore.QPointF(hx, hy))

        p.end()


class CoilVisualization(QtWidgets.QWidget):
    """Two side-by-side coil-schematic panels + a legend footer."""

    def __init__(self, Bmap: np.ndarray, motion=None, config=None, parent=None):
        super().__init__(parent)
        self._Bmap = np.asarray(Bmap, dtype=float).reshape(3, 6)
        self._motion = motion
        self._config = config
        self._currents = np.zeros(6)
        self._last_paint = 0.0
        # Repaint at most every 33 ms (≈30 Hz) even if update_state
        # is being hammered by the 200 Hz inner loop.
        self._min_interval_s = 1.0 / 30.0

        # World → screen 2×2. Applied to the top-down panel only so its layout
        # matches the microscope's camera frame. Kept as identity when no
        # Frame Cal has been run.
        self._M_ws = np.eye(2)
        self._reload_frame_matrix()
        if config is not None:
            try:
                config.on_change(
                    "calibration.screen_to_world_2x2",
                    lambda *_a: self._on_frame_matrix_changed())
            except Exception:
                pass  # older Config without on_change — restart-only.

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        panels_row = QtWidgets.QHBoxLayout()
        # Top-down: X horizontal, Y vertical. Applies the frame-calibration
        # transform so a stick-up command shows the arrow pointing up on this
        # panel and the bead in the scope moves screen-up.
        self.top_down = _CoilPanel(
            "Top-down (XY)", ax_x=0, ax_y=1,
            xy_transform=self._transform_xy)
        # Side: X horizontal, Z vertical. Frame cal is horizontal-only; Z is
        # camera-invisible, so this panel stays in world axes.
        self.side = _CoilPanel("Side (XZ)", ax_x=0, ax_y=2)
        panels_row.addWidget(self.top_down)
        panels_row.addWidget(self.side)
        layout.addLayout(panels_row)

        legend = QtWidgets.QLabel(
            "<span style='color:#00d4ff;'>Blue fill</span> = coil duty. "
            "Top-ring coils are filled discs; bottom-ring coils are outlines. "
            "<span style='color:#ff44dd;'>Magenta arrow</span> = commanded "
            "pull direction (matches your stick / Mode B command)."
        )
        legend.setWordWrap(True)
        legend.setStyleSheet("color: #aaaaaa; font-size: 10pt; padding: 2px;")
        layout.addWidget(legend)

        self.setMinimumHeight(240)

    # -------- public API ----------------------------------------

    def update_state(self, currents) -> None:
        """Store the latest sent currents. Schedules a repaint at most every
        ~33 ms so the inner loop can push at 200 Hz without ever spending
        useful CPU on paint calls."""
        self._currents = np.asarray(currents, dtype=float).reshape(6)
        now = time.monotonic()
        if now - self._last_paint < self._min_interval_s:
            return
        self._last_paint = now
        self._push_to_panels()

    def set_bmap(self, Bmap) -> None:
        """Called from the solver-rebuild path when calibration changes so
        the pull-direction arrow stays honest."""
        self._Bmap = np.asarray(Bmap, dtype=float).reshape(3, 6)
        # Force one refresh with the new Bmap.
        self._last_paint = 0.0
        self._push_to_panels()

    # -------- internal ------------------------------------------

    def _reload_frame_matrix(self) -> None:
        """Recompute M_ws = inv(M_sw). Called at init and on config change."""
        if self._config is None:
            self._M_ws = np.eye(2)
            return
        raw = self._config.get(
            "calibration.screen_to_world_2x2", [[1.0, 0.0], [0.0, 1.0]])
        try:
            M_sw = np.asarray(raw, dtype=float).reshape(2, 2)
            self._M_ws = np.linalg.inv(M_sw)
        except Exception:
            self._M_ws = np.eye(2)

    def _on_frame_matrix_changed(self) -> None:
        self._reload_frame_matrix()
        # Force a repaint even if the throttle just fired.
        self._last_paint = 0.0
        self._push_to_panels()

    def _transform_xy(self, xy: np.ndarray) -> np.ndarray:
        return self._M_ws @ xy

    def _current_B_des(self) -> np.ndarray:
        """Compute commanded field vector — matches MotionController.tick()."""
        m = self._motion
        if m is None:
            return np.zeros(3)
        mode = getattr(m, "mode", None)
        if mode == Mode.B_STATIC:
            return float(m.magnitude) * np.asarray(m.direction, dtype=float)
        if mode == Mode.A_ROTATING:
            # Mirror MotionController: R(roll_axis, phase) @ direction.
            from classes.motion_controller import _rotation_matrix
            phase = float(getattr(m, "phase", 0.0))
            R = _rotation_matrix(m.roll_axis, phase)
            return float(m.magnitude) * (R @ np.asarray(m.direction, dtype=float))
        return np.zeros(3)

    def _push_to_panels(self) -> None:
        # Arrow tracks commanded B_des (smooth, matches operator intent);
        # coil intensities still show what actually fired.
        field = self._current_B_des()
        self.top_down.set_state(self._currents, field)
        self.side.set_state(self._currents, field)
        self.top_down.update()
        self.side.update()
