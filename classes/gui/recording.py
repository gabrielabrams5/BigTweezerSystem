"""Annotated-view video recording.

The ⏺ button on the video widget records exactly what the operator sees —
the displayed (possibly rotated) frame plus overlays — to an .mp4.

Design:

  * A fixed-rate grab timer (``recording.fps``, PreciseTimer) on the GUI
    thread renders the video area via ``VideoWidget.render_video_image()``
    and hands BGR frames to a worker. Grab rate == writer fps, so the file
    plays back at wall-clock speed by construction — no camera-fps
    guessing.
  * Encoding (the expensive part) runs on a dedicated QThread so the
    shared 200 Hz inner control loop never pays for it. A bounded backlog
    drops frames if the encoder falls behind.
  * Output size is latched at start; later window resizes or ⟳ rotations
    are aspect-fit onto the latched canvas rather than distorting.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui

from classes.config_loader import REPO_ROOT


_IDLE_STYLE = (
    "QToolButton { background: rgba(0, 0, 0, 120); color: #ddd; "
    "border: 1px solid #555; border-radius: 3px; padding: 2px 6px; }"
    "QToolButton:hover { background: rgba(60, 60, 60, 180); }")
_REC_STYLE = (
    "QToolButton { background: rgba(180, 20, 20, 200); color: white; "
    "border: 1px solid #a33; border-radius: 3px; padding: 2px 6px; }"
    "QToolButton:hover { background: rgba(210, 40, 40, 220); }")


# ---- pure helpers (headless-testable) ---------------------------------


def resolve_output_dir(config=None, create: bool = True) -> str:
    """Recording output folder.

    ``recording.output_dir`` overrides; empty → platform default
    (Windows: D:/Microrobots/Tracking Data; else <repo>/Data — anchored on
    REPO_ROOT, not the CWD, unlike the legacy stack).
    """
    override = ""
    if config is not None:
        override = str(config.get("recording.output_dir", "") or "")
    if override:
        path = os.path.expanduser(override)
    elif platform.system() == "Windows":
        path = os.path.join("D:/", "Microrobots", "Tracking Data")
    else:
        path = os.path.join(REPO_ROOT, "Data")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def qimage_to_bgr(img: QtGui.QImage) -> np.ndarray:
    """QImage → contiguous BGR uint8 ndarray (owns its memory)."""
    img = img.convertToFormat(QtGui.QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    try:
        nbytes = img.sizeInBytes()
    except AttributeError:      # Qt < 5.10
        nbytes = img.byteCount()
    ptr.setsize(nbytes)
    buf = np.frombuffer(ptr, dtype=np.uint8).reshape(h, img.bytesPerLine())
    # QImage rows are 4-byte aligned — strip the stride padding.
    rgb = buf[:, : w * 3].reshape(h, w, 3)
    # RGB → BGR; the copy also detaches from the QImage buffer before the
    # array crosses threads.
    return np.ascontiguousarray(rgb[:, :, ::-1])


def fit_to_size(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Aspect-preserving fit of ``frame`` onto a black canvas of ``size``.

    ``size`` is (w, h). Exact-size frames pass through untouched. Handles
    mid-recording window resizes and ⟳ rotations (w/h swap) without
    squashing the picture — cv2.VideoWriter silently drops mismatched
    frames otherwise.
    """
    tw, th = int(size[0]), int(size[1])
    fh, fw = frame.shape[:2]
    if (fw, fh) == (tw, th):
        return frame
    scale = min(tw / fw, th / fh)
    nw = max(1, int(round(fw * scale)))
    nh = max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# ---- worker (lives on its own QThread) --------------------------------


class VideoRecorderWorker(QtCore.QObject):
    """Owns the cv2.VideoWriter; every slot runs on the record thread."""

    wrote = QtCore.pyqtSignal()                # backlog accounting
    stopped = QtCore.pyqtSignal(str, int)      # (path, n_frames_written)
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._writer: Optional[cv2.VideoWriter] = None
        self._path = ""
        self._size: Tuple[int, int] = (0, 0)
        self._count = 0

    @QtCore.pyqtSlot(str, object, float)
    def start(self, path: str, size: object, fps: float) -> None:
        w, h = int(size[0]), int(size[1])
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not writer.isOpened():
            self.error.emit(f"could not open video writer at {path}")
            return
        self._writer = writer
        self._path = path
        self._size = (w, h)
        self._count = 0

    @QtCore.pyqtSlot(np.ndarray)
    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        self._writer.write(fit_to_size(frame, self._size))
        self._count += 1
        self.wrote.emit()

    @QtCore.pyqtSlot()
    def stop(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        self.stopped.emit(self._path, self._count)
        self._path = ""
        self._count = 0


# ---- controller (GUI thread) ------------------------------------------


class RecordingController(QtCore.QObject):
    """Wires the video widget's ⏺ button to a threaded mp4 writer."""

    _startReq = QtCore.pyqtSignal(str, object, float)
    _stopReq = QtCore.pyqtSignal()
    frameGrabbed = QtCore.pyqtSignal(np.ndarray)

    _MAX_BACKLOG = 4

    def __init__(self, video, config, log_fn: Callable[[str], None] = print,
                 parent=None):
        super().__init__(parent)
        self.video = video
        self.config = config
        self.log = log_fn

        self._recording = False
        self._t0 = 0.0
        self._pending = 0
        self._dropped = 0

        self._thread = QtCore.QThread(self)
        self._worker = VideoRecorderWorker()
        self._worker.moveToThread(self._thread)
        self._startReq.connect(self._worker.start)
        self._stopReq.connect(self._worker.stop)
        self.frameGrabbed.connect(self._worker.write)
        self._worker.wrote.connect(self._on_wrote)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.error.connect(self._on_error)
        self._thread.start()

        self._grab_timer = QtCore.QTimer(self)
        self._grab_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._grab_timer.timeout.connect(self._on_grab)

        self._elapsed_timer = QtCore.QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed)

        video.record_btn.clicked.connect(self.toggle)

    # ---- start / stop ------------------------------------------------

    def toggle(self) -> None:
        if self._recording:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self._recording:
            return
        img = self.video.render_video_image()
        if img is None:
            self.log("Recording: no video frame yet — connect a camera first")
            return
        # mp4v (yuv420) wants even dimensions.
        w = img.width() & ~1
        h = img.height() & ~1
        if w < 2 or h < 2:
            self.log("Recording: video area too small")
            return
        fps = float(self.config.get("recording.fps", 20.0))
        fps = max(1.0, min(60.0, fps))
        path = os.path.join(
            resolve_output_dir(self.config),
            time.strftime("rec_%Y%m%d-%H%M%S") + ".mp4")

        self._recording = True
        self._t0 = time.monotonic()
        self._pending = 0
        self._dropped = 0
        self._startReq.emit(path, (w, h), fps)
        self._grab_timer.setInterval(max(1, int(round(1000.0 / fps))))
        self._grab_timer.start()
        self._elapsed_timer.start()
        self.video.record_btn.setStyleSheet(_REC_STYLE)
        self.video.record_btn.setText("⏹ 00:00")
        self.log(f"Recording started → {path} ({w}x{h} @ {fps:g} fps)")

    def stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._grab_timer.stop()
        self._elapsed_timer.stop()
        self._stopReq.emit()

    def shutdown(self) -> None:
        """Finalize any running recording before the app tears down."""
        self._grab_timer.stop()
        self._elapsed_timer.stop()
        if self._recording:
            self._recording = False
            # Deliver the flush synchronously — a queued stop followed by
            # quit() can race (quit doesn't drain the queue).
            QtCore.QMetaObject.invokeMethod(
                self._worker, "stop", QtCore.Qt.BlockingQueuedConnection)
        self._thread.quit()
        self._thread.wait(2000)

    # ---- internals ----------------------------------------------------

    def _on_grab(self) -> None:
        if not self._recording:
            return
        if self._pending >= self._MAX_BACKLOG:
            self._dropped += 1
            return
        img = self.video.render_video_image()
        if img is None:
            return   # source disconnected mid-recording; skip the tick
        self._pending += 1
        self.frameGrabbed.emit(qimage_to_bgr(img))

    def _on_wrote(self) -> None:
        self._pending = max(0, self._pending - 1)

    def _refresh_elapsed(self) -> None:
        s = int(time.monotonic() - self._t0)
        self.video.record_btn.setText(f"⏹ {s // 60:02d}:{s % 60:02d}")

    def _on_stopped(self, path: str, n: int) -> None:
        msg = f"Recording saved: {path} ({n} frames)"
        if self._dropped:
            msg += f" — {self._dropped} frames dropped (encoder backlog)"
        self.log(msg)
        btn = self.video.record_btn
        btn.setText("✓ saved")
        QtCore.QTimer.singleShot(3000, self._reset_button)

    def _on_error(self, msg: str) -> None:
        self.log(f"Recording error: {msg}")
        self._recording = False
        self._grab_timer.stop()
        self._elapsed_timer.stop()
        self._reset_button()

    def _reset_button(self) -> None:
        if self._recording:
            return
        btn = self.video.record_btn
        btn.setText("⏺")
        btn.setStyleSheet(_IDLE_STYLE)
