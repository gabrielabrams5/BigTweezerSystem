"""Camera source + display for the new GUI.

Two moving parts:

  * :class:`VideoSource` — a QThread that owns the underlying capture
    object (aravis / EasyPySpin / cv2.VideoCapture) and emits a
    ``frameReady`` signal per grab. The capture backend is picked at
    connect time in the same order the legacy GUI used:
    aravis → EasyPySpin → cv2 webcam → video file.

  * :class:`VideoWidget` — a QLabel that renders the latest frame
    letterboxed to fit its box. Centre-aligned so black bars are
    symmetric (the legacy build shipped with an off-centre bug there).

The pair is deliberately independent of the tracker / algorithm code —
the new GUI ships with manual field control first; feedback tracking
gets bolted on later.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from classes.gui import view_transform

# cv2 rotation codes for the display-only view rotation (CW degrees).
_ROT_CV = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


# Try to import the FLIR Spinnaker binding; keep it optional.
try:
    import EasyPySpin  # noqa: F401
    _HAS_EASYPYSPIN = True
except Exception:
    _HAS_EASYPYSPIN = False


def _try_aravis(printer) -> Optional[object]:
    try:
        from classes import aravis_camera
    except Exception as e:
        printer(f"aravis backend import failed: {e}")
        if "'gi'" in str(e) or "No module named" in str(e):
            printer(
                "  ^ PyGObject not installed for this Python interpreter. "
                "On macOS the Homebrew bottle installs it into python3.13. "
                "Try:  /opt/homebrew/bin/python3.13 main.py"
            )
        return None
    try:
        if not aravis_camera.is_available():
            printer("aravis: no camera enumerated on the bus")
            return None
        cap = aravis_camera.AravisCameraCapture(printer=printer)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FPS, 19)
        printer("Camera: connected via aravis")
        return cap
    except Exception as e:
        printer(f"aravis backend failed: {e}")
        return None


def _try_easypyspin(printer) -> Optional[object]:
    if not _HAS_EASYPYSPIN:
        printer("EasyPySpin not installed; skipping")
        return None
    try:
        import EasyPySpin
        cap = EasyPySpin.VideoCapture(0)
        if not cap.isOpened():
            printer("EasyPySpin: could not open camera 0")
            return None
        cap.set(cv2.CAP_PROP_AUTO_WB, True)
        cap.set(cv2.CAP_PROP_FPS, 19)
        # Force BGR8; PixelFormat is read-only mid-stream, so end
        # acquisition, set format, restart.
        try:
            import PySpin
            cam = cap.cam
            try: cam.EndAcquisition()
            except Exception: pass
            try:
                cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
            except Exception as e:
                printer(f"EasyPySpin PixelFormat=BGR8 skipped: {e}")
            try: cam.BeginAcquisition()
            except Exception: pass
        except ImportError:
            pass
        printer("Camera: connected via EasyPySpin (FLIR)")
        return cap
    except Exception as e:
        printer(f"EasyPySpin backend failed: {e}")
        return None


def _try_cv2_webcam(printer) -> Optional[object]:
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            printer("Camera: connected via cv2 (webcam / UVC)")
            return cap
    except Exception as e:
        printer(f"cv2 backend failed: {e}")
    return None


def _try_video_file(path: str, printer) -> Optional[object]:
    try:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            printer(f"Camera: playing video file {os.path.basename(path)}")
            return cap
        printer(f"Could not open video file {path}")
    except Exception as e:
        printer(f"Video file open failed: {e}")
    return None


def open_camera(source: str = "auto",
                video_path: str = "",
                printer=print):
    """Open a capture object per the ``source`` selector.

    ``source``:
        ``"auto"``    — aravis → EasyPySpin → cv2 webcam (default).
        ``"aravis"``  — force aravis (fail if not available).
        ``"easypyspin"`` — force EasyPySpin.
        ``"webcam"``  — force cv2.VideoCapture(0).
        ``"file"``    — force video file (uses ``video_path``).
    """
    source = (source or "auto").lower()
    if source == "file":
        return _try_video_file(video_path, printer)
    if source == "aravis":
        return _try_aravis(printer)
    if source == "easypyspin":
        return _try_easypyspin(printer)
    if source == "webcam":
        return _try_cv2_webcam(printer)
    # auto
    for fn in (_try_aravis, _try_easypyspin, _try_cv2_webcam):
        cap = fn(printer)
        if cap is not None:
            return cap
    printer("Camera: no backend succeeded; running headless")
    return None


class VideoSource(QtCore.QThread):
    """Owns the capture object; emits frames on a signal.

    Live sources (aravis / EasyPySpin / webcam) run in "as-fast-as-driver"
    mode — the underlying capture already blocks until the next frame.
    Video files honour their embedded FPS via a short sleep.
    """

    # (frame, capture_time_monotonic_s). The timestamp is taken the
    # instant read() returns: consumers that derive dt from arrival time
    # instead fold queue and inference latency into every interval.
    frameReady = QtCore.pyqtSignal(np.ndarray, float)
    statusChanged = QtCore.pyqtSignal(str)

    def __init__(self, printer=print, parent=None):
        super().__init__(parent)
        self.printer = printer
        self.cap = None
        self._running = False
        self._file_path = ""
        self._is_file = False
        self._fps = 30.0

    def open(self, source: str = "auto", video_path: str = "") -> bool:
        self.close()
        self.cap = open_camera(source=source,
                               video_path=video_path,
                               printer=self.printer)
        if self.cap is None:
            self.statusChanged.emit("no source")
            return False
        self._is_file = (source == "file")
        self._file_path = video_path
        try:
            self._fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        except Exception:
            self._fps = 30.0
        self._running = True
        self.start()
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.statusChanged.emit(f"{w}×{h} @ {self._fps:.0f} fps")
        return True

    def close(self) -> None:
        self._running = False
        if self.isRunning():
            self.wait(1000)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.statusChanged.emit("disconnected")

    def run(self) -> None:
        # Throttle to (at most) the camera's own FPS so we don't flood the
        # Qt signal queue. For aravis, ``read()`` returns immediately with
        # whatever frame is currently latched — without a sleep here we spin
        # at CPU-max rate and starve the main thread. Cap the display rate
        # at ~30 Hz so full-frame renders don't dominate the event loop.
        display_fps = min(30.0, max(1.0, self._fps))
        target_dt = 1.0 / display_fps
        last_frame_id = None
        while self._running and self.cap is not None:
            t0 = time.monotonic()
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            t_capture = time.monotonic()
            if not ok or frame is None:
                if self._is_file:
                    self._running = False
                    break
                time.sleep(0.01)
                continue
            # Skip emitting if the aravis backend hasn't produced a new frame
            # since the last iteration — otherwise the main thread does a
            # full cvtColor + QImage + scaled for nothing.
            fid = id(frame)
            if fid != last_frame_id:
                last_frame_id = fid
                self.frameReady.emit(frame, t_capture)
            # Pace the loop.
            elapsed = time.monotonic() - t0
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)


# Robot overlay colours. Amber matches the IN_PROGRESS cell hue so the
# "needs attention" reading is consistent across the overlay.
_ROBOT_OK = "#2ecc71"
_ROBOT_CONTESTED = "#f39c12"


class VideoWidget(QtWidgets.QLabel):
    """QLabel that renders a BGR ndarray, letterboxed to fit.

    Owns its own paintEvent so it can draw overlays (tracker boxes,
    trajectories, look-ahead markers) glued to the same coordinate frame
    as the video. Mouse clicks are converted from widget space to
    original-frame pixel space via the stored render transform, so
    callers get back true image coordinates regardless of window resize
    or letterbox padding.
    """

    leftClicked = QtCore.pyqtSignal(int, int)
    rightClicked = QtCore.pyqtSignal(int, int)
    middleClicked = QtCore.pyqtSignal(int, int)

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color: #101010; color: #888;")
        self.setText("no camera")
        self.setMinimumSize(320, 240)
        self._config = config
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_t: float = 0.0
        self._displayed_pixmap: Optional[QtGui.QPixmap] = None
        self._render_scale: float = 1.0                 # display px → widget px
        self._render_offset: tuple = (0, 0)              # letterbox top-left
        self._frame_size: tuple = (0, 0)                 # (w, h) of the source
        self._overlay: dict = {}

        # Display-only view rotation (0/90/180/270 CW). The tracker and all
        # click signals stay in raw camera coordinates.
        rot = 0
        if config is not None:
            try:
                rot = int(config.get("camera.view_rotation_deg", 0)) % 360
            except Exception:
                rot = 0
        self._rotation_deg = rot if rot in (0, 90, 180, 270) else 0

        self._rotate_btn = QtWidgets.QToolButton(self)
        self._rotate_btn.setText("⟳ 90°")
        self._rotate_btn.setToolTip("Rotate view 90° (display only)")
        self._rotate_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._rotate_btn.setStyleSheet(
            "QToolButton { background: rgba(0, 0, 0, 120); color: #ddd; "
            "border: 1px solid #555; border-radius: 3px; padding: 2px 6px; }"
            "QToolButton:hover { background: rgba(60, 60, 60, 180); }")
        self._rotate_btn.clicked.connect(self._on_rotate_clicked)
        self._rotate_btn.raise_()
        self._place_rotate_btn()

        # Record button (top-left). The RecordingController owns behavior —
        # this widget just hosts the button.
        self.record_btn = QtWidgets.QToolButton(self)
        self.record_btn.setText("⏺")
        self.record_btn.setToolTip("Record annotated view (.mp4)")
        self.record_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.record_btn.setStyleSheet(
            "QToolButton { background: rgba(0, 0, 0, 120); color: #ddd; "
            "border: 1px solid #555; border-radius: 3px; padding: 2px 6px; }"
            "QToolButton:hover { background: rgba(60, 60, 60, 180); }")
        self.record_btn.raise_()
        self._place_record_btn()

    # ---- inputs ----------------------------------------------------

    def set_frame(self, frame: np.ndarray, t_capture: float = 0.0) -> None:
        self._last_frame = frame
        self._last_frame_t = float(t_capture)
        self._prepare_pixmap()
        self.update()

    def set_overlay(self, overlay: dict) -> None:
        self._overlay = overlay or {}
        self.update()

    # ---- lifecycle -------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_rotate_btn()
        self._place_record_btn()
        if self._last_frame is not None:
            self._prepare_pixmap()
            self.update()

    def render_video_image(self) -> Optional[QtGui.QImage]:
        """Video area + overlays as a QImage — no child buttons, no
        letterbox bars. Used by the recorder; pixel-identical to the
        on-screen view (the overlay painter is translated so widget-coord
        drawing lands on the pixmap origin)."""
        pm = self._displayed_pixmap
        if pm is None:
            return None
        img = QtGui.QImage(pm.size(), QtGui.QImage.Format_RGB32)
        p = QtGui.QPainter(img)
        p.drawPixmap(0, 0, pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.translate(-self._render_offset[0], -self._render_offset[1])
        try:
            self._paint_overlay(p)
        except Exception:
            pass
        p.end()
        return img

    def _place_rotate_btn(self) -> None:
        hint = self._rotate_btn.sizeHint()
        self._rotate_btn.move(self.width() - hint.width() - 8, 8)

    def _place_record_btn(self) -> None:
        self.record_btn.move(8, 8)

    def _on_rotate_clicked(self) -> None:
        self._rotation_deg = (self._rotation_deg + 90) % 360
        if self._config is not None:
            try:
                self._config.set("camera.view_rotation_deg",
                                 self._rotation_deg)
                self._config.save()
            except Exception:
                pass
        if self._last_frame is not None:
            self._prepare_pixmap()
        self.update()

    def _prepare_pixmap(self) -> None:
        frame = self._last_frame
        if frame is None:
            return
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        self._frame_size = (w, h)                        # always the RAW size

        rot = self._rotation_deg
        if rot in _ROT_CV:
            frame = cv2.rotate(frame, _ROT_CV[rot])
        dh, dw = frame.shape[:2]                         # display-space dims

        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        scale = min(widget_w / dw, widget_h / dh)

        if scale < 0.9:
            new_w = max(1, int(round(dw * scale)))
            new_h = max(1, int(round(dh * scale)))
            downsampled = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            downsampled = frame
            new_w, new_h = dw, dh

        rgb = cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB)
        self._rgb_buf = np.ascontiguousarray(rgb)
        img = QtGui.QImage(
            self._rgb_buf.data, new_w, new_h, new_w * 3,
            QtGui.QImage.Format_RGB888)
        self._displayed_pixmap = QtGui.QPixmap.fromImage(img)
        # Actual drawn-px per display-px — when the pixmap isn't resized
        # (scale >= 0.9) it draws at 1:1, so the fractional fit scale would
        # skew overlays and clicks.
        self._render_scale = new_w / dw
        pw, ph = self._displayed_pixmap.width(), self._displayed_pixmap.height()
        self._render_offset = ((widget_w - pw) // 2, (widget_h - ph) // 2)

    # ---- paint -----------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#101010"))
        if self._displayed_pixmap is None:
            p.setPen(QtGui.QColor("#888"))
            p.drawText(
                self.rect(), QtCore.Qt.AlignCenter, self.text())
            p.end()
            return
        p.drawPixmap(
            self._render_offset[0], self._render_offset[1],
            self._displayed_pixmap)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint_overlay(p)
        except Exception:
            pass
        p.end()

    # ---- overlay ---------------------------------------------------

    def _f2w(self, x: float, y: float) -> tuple:
        w, h = self._frame_size
        dx, dy = view_transform.frame_to_display(
            x, y, w, h, self._rotation_deg)
        return (
            self._render_offset[0] + dx * self._render_scale,
            self._render_offset[1] + dy * self._render_scale,
        )

    def _paint_overlay(self, p: QtGui.QPainter) -> None:
        ov = self._overlay
        if not ov:
            return

        # Mask overlays first, so tracker boxes and paths draw over them.
        robot_mask = ov.get("robot_mask")
        if robot_mask is not None:
            self._draw_mask(p, robot_mask, (220, 70, 70))
        cell_mask = ov.get("cell_mask")
        if cell_mask is not None:
            self._draw_mask(p, cell_mask, (70, 200, 220))

        # Robot bbox + trajectory
        robot = ov.get("robot")
        if robot and robot.get("pos") is not None:
            # Amber + bold while contested: the tracker has two equally
            # good candidates and is coasting on its filter rather than
            # trusting the correlation peak. Green means unambiguous.
            contested = bool(robot.get("contested"))
            self._draw_object(
                p, robot,
                colour=QtGui.QColor(_ROBOT_CONTESTED if contested
                                    else _ROBOT_OK),
                label="R?" if contested else "R",
                target_colour=QtGui.QColor("#f1c40f"),
                trajectory_colour=QtGui.QColor("#3498db"),
                bold_box=contested)

        # Cell bboxes + trajectories
        cell_hues = [
            QtGui.QColor("#e67e22"), QtGui.QColor("#9b59b6"),
            QtGui.QColor("#1abc9c"), QtGui.QColor("#e74c3c"),
            QtGui.QColor("#8e44ad"), QtGui.QColor("#16a085"),
            QtGui.QColor("#d35400"), QtGui.QColor("#2980b9"),
        ]
        status_colours = {
            "PENDING": QtGui.QColor("#888888"),
            "IN_PROGRESS": QtGui.QColor("#f39c12"),
            "DONE": QtGui.QColor("#2ecc71"),
            "LOST": QtGui.QColor("#c0392b"),
        }
        active_idx = ov.get("active_cell_idx", -1)
        for cell in ov.get("cells", []):
            if cell.get("pos") is None:
                continue
            box_col = status_colours.get(cell.get("status", "PENDING"),
                                         QtGui.QColor("#888"))
            traj_col = cell_hues[cell["index"] % len(cell_hues)]
            label = f"C{cell['index'] + 1}"
            self._draw_object(
                p, cell, colour=box_col, label=label,
                target_colour=QtGui.QColor("#f1c40f"),
                trajectory_colour=traj_col,
                bold_box=(cell["index"] == active_idx),
            )

        # Push-mode arrows: from robot to current robot_target (approach or push)
        if ov.get("operation_mode") == "push_cells" and ov.get("running"):
            self._draw_push_arrows(p, ov)

    def _draw_mask(self, p: QtGui.QPainter, mask: np.ndarray,
                   rgb: tuple) -> None:
        """Draw a binary mask as a semi-transparent tint over the video.

        ``mask`` is expected at the original frame resolution. It's resized
        (INTER_NEAREST — the mask is binary, we want to preserve edges) to
        the displayed pixmap dimensions, then converted to an RGBA image
        where "on" pixels take the given colour at ~35% alpha and "off"
        pixels are fully transparent.
        """
        if mask is None or self._displayed_pixmap is None:
            return
        # The mask raster bypasses _f2w — rotate it the same way the frame
        # was rotated in _prepare_pixmap.
        if self._rotation_deg in _ROT_CV:
            mask = cv2.rotate(mask, _ROT_CV[self._rotation_deg])
        try:
            mh, mw = mask.shape[:2]
        except Exception:
            return
        pw = self._displayed_pixmap.width()
        ph = self._displayed_pixmap.height()
        if pw <= 0 or ph <= 0:
            return
        if (mw, mh) != (pw, ph):
            small = cv2.resize(mask, (pw, ph),
                               interpolation=cv2.INTER_NEAREST)
        else:
            small = mask
        rgba = np.zeros((ph, pw, 4), dtype=np.uint8)
        on = small > 0
        r, g, b = rgb
        rgba[..., 0][on] = r
        rgba[..., 1][on] = g
        rgba[..., 2][on] = b
        rgba[..., 3][on] = 90
        # Keep a reference so the QImage buffer stays valid until draw.
        self._mask_rgba_buf = np.ascontiguousarray(rgba)
        img = QtGui.QImage(
            self._mask_rgba_buf.data, pw, ph, pw * 4,
            QtGui.QImage.Format_RGBA8888)
        p.drawImage(
            self._render_offset[0], self._render_offset[1], img)

    def _draw_object(self,
                     p: QtGui.QPainter,
                     obj: dict,
                     colour: QtGui.QColor,
                     label: str,
                     target_colour: QtGui.QColor,
                     trajectory_colour: QtGui.QColor,
                     bold_box: bool = False) -> None:
        pos = obj["pos"]
        crop = obj.get("crop_length", 40)
        side = 2 * crop * self._render_scale
        wx, wy = self._f2w(pos[0], pos[1])
        pen_w = 3 if bold_box else 2
        p.setPen(QtGui.QPen(colour, pen_w))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(wx - side / 2, wy - side / 2, side, side))
        # Label
        p.setPen(colour)
        p.setFont(QtGui.QFont("Menlo", 9, QtGui.QFont.Bold))
        p.drawText(QtCore.QPointF(wx - side / 2 + 4, wy - side / 2 - 4),
                   label)

        # Trajectory
        traj = obj.get("trajectory", [])
        target_idx = obj.get("target_idx", 0)
        if traj:
            p.setPen(QtGui.QPen(trajectory_colour, 2))
            path = QtGui.QPainterPath()
            first_wx, first_wy = self._f2w(*traj[0])
            path.moveTo(first_wx, first_wy)
            for tx, ty in traj[1:]:
                pwx, pwy = self._f2w(tx, ty)
                path.lineTo(pwx, pwy)
            p.drawPath(path)
            # Dots + numbering
            for i, (tx, ty) in enumerate(traj):
                pwx, pwy = self._f2w(tx, ty)
                is_target = (i == target_idx)
                col = target_colour if is_target else trajectory_colour
                p.setBrush(col)
                p.setPen(QtCore.Qt.NoPen)
                r = 6 if is_target else 4
                p.drawEllipse(QtCore.QPointF(pwx, pwy), r, r)
                p.setPen(QtGui.QColor("#ffffff"))
                p.setFont(QtGui.QFont("Menlo", 8))
                p.drawText(QtCore.QPointF(pwx + 8, pwy - 4),
                           str(i + 1))

    def _draw_push_arrows(self, p: QtGui.QPainter, ov: dict) -> None:
        robot = ov.get("robot")
        if robot is None or robot.get("pos") is None:
            return
        active_idx = ov.get("active_cell_idx", -1)
        cells = ov.get("cells", [])
        if not (0 <= active_idx < len(cells)):
            return
        cell = cells[active_idx]
        if cell.get("pos") is None:
            return
        traj = cell.get("trajectory", [])
        if cell.get("target_idx", 0) >= len(traj):
            return
        goal = traj[cell["target_idx"]]
        cx, cy = cell["pos"]
        rx, ry = robot["pos"]

        # Cell → goal arrow (magenta)
        p.setPen(QtGui.QPen(QtGui.QColor("#ff44dd"), 2, QtCore.Qt.DashLine))
        wc = self._f2w(cx, cy)
        wg = self._f2w(*goal)
        p.drawLine(QtCore.QPointF(*wc), QtCore.QPointF(*wg))
        # Robot → cell (solid magenta)
        p.setPen(QtGui.QPen(QtGui.QColor("#ff44dd"), 3))
        wr = self._f2w(rx, ry)
        p.drawLine(QtCore.QPointF(*wr), QtCore.QPointF(*wc))

    # ---- mouse -----------------------------------------------------

    def _widget_to_frame(self, x: int, y: int):
        if self._render_scale < 1e-9 or self._displayed_pixmap is None:
            return None
        dx = (x - self._render_offset[0]) / self._render_scale
        dy = (y - self._render_offset[1]) / self._render_scale
        fw, fh = self._frame_size
        fx, fy = view_transform.display_to_frame(
            dx, dy, fw, fh, self._rotation_deg)
        if fx < 0 or fy < 0 or fx >= fw or fy >= fh:
            return None
        return int(round(fx)), int(round(fy))

    def mousePressEvent(self, event) -> None:
        coords = self._widget_to_frame(event.x(), event.y())
        if coords is None:
            return
        fx, fy = coords
        if event.button() == QtCore.Qt.LeftButton:
            self.leftClicked.emit(fx, fy)
        elif event.button() == QtCore.Qt.RightButton:
            self.rightClicked.emit(fx, fy)
        elif event.button() == QtCore.Qt.MiddleButton:
            self.middleClicked.emit(fx, fy)


class CameraPanel(QtWidgets.QWidget):
    """Sidebar panel: source selection, connect/disconnect, exposure, status."""

    def __init__(self, source: VideoSource, parent=None):
        super().__init__(parent)
        self.source = source

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Source combo
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Source:"))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["auto", "aravis", "easypyspin", "webcam", "file"])
        row.addWidget(self.source_combo, 1)
        layout.addLayout(row)

        # File path row (only visible when source == file)
        f_row = QtWidgets.QHBoxLayout()
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("Video file path")
        f_row.addWidget(self.file_edit, 1)
        self.file_btn = QtWidgets.QPushButton("…")
        self.file_btn.setFixedWidth(28)
        self.file_btn.clicked.connect(self._pick_file)
        f_row.addWidget(self.file_btn)
        f_widget = QtWidgets.QWidget()
        f_widget.setLayout(f_row)
        layout.addWidget(f_widget)

        # Connect / disconnect
        btn_row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.source.close)
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.disconnect_btn)
        layout.addLayout(btn_row)

        # Status label
        self.status = QtWidgets.QLabel("disconnected")
        self.status.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.status)

        # Exposure spin (only takes effect on live sources).
        exp_row = QtWidgets.QHBoxLayout()
        exp_row.addWidget(QtWidgets.QLabel("Exposure (µs):"))
        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(10, 200000)
        self.exp_spin.setValue(5000)
        self.exp_spin.valueChanged.connect(self._on_exposure)
        exp_row.addWidget(self.exp_spin, 1)
        layout.addLayout(exp_row)

        # FPS spin
        fps_row = QtWidgets.QHBoxLayout()
        fps_row.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 200)
        self.fps_spin.setValue(19)
        self.fps_spin.valueChanged.connect(self._on_fps)
        fps_row.addWidget(self.fps_spin, 1)
        layout.addLayout(fps_row)

        layout.addStretch(1)

        self.source.statusChanged.connect(self.status.setText)

    def _pick_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose video file", "", "Video (*.mp4 *.mov *.avi *.mkv)")
        if path:
            self.file_edit.setText(path)
            self.source_combo.setCurrentText("file")

    def _on_connect(self) -> None:
        source = self.source_combo.currentText()
        video_path = self.file_edit.text().strip()
        ok = self.source.open(source=source, video_path=video_path)
        if not ok:
            self.status.setText("connect failed — see log")

    def _on_exposure(self, v: int) -> None:
        if self.source.cap is not None:
            try:
                self.source.cap.set(cv2.CAP_PROP_EXPOSURE, int(v))
            except Exception:
                pass

    def _on_fps(self, v: int) -> None:
        if self.source.cap is not None:
            try:
                self.source.cap.set(cv2.CAP_PROP_FPS, int(v))
            except Exception:
                pass
