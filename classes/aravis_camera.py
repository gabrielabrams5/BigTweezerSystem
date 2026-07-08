"""macOS-friendly FLIR camera backend using aravis + PyGObject.

FLIR Blackfly S / other U3V machine-vision cameras don't speak UVC, so
macOS's AVFoundation (and therefore cv2.VideoCapture) can't see them.
The aravis project (Homebrew: `brew install aravis pygobject3`) provides
a GenICam / U3V library that works on macOS.

This module wraps aravis in a class that looks enough like
cv2.VideoCapture that tracker_class.VideoThread can use it without
changes. Only the properties and methods actually used elsewhere in
the app are implemented:
    set(cv2.CAP_PROP_FPS, fps)
    set(cv2.CAP_PROP_EXPOSURE, us)
    set(cv2.CAP_PROP_AUTO_WB, bool)  -- no-op, camera is monochrome
    isOpened()
    get(cv2.CAP_PROP_FRAME_WIDTH|HEIGHT|FPS|FRAME_COUNT)
    read() -> (bool, np.ndarray)  -- BGR frame
    release()

Streams frames continuously in a background thread and returns the most
recent one on each read(). That matches cv2.VideoCapture's semantics of
"give me the latest frame" better than aravis's default pop-a-buffer
model, and lets the tracker's frame loop run at its own pace.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


def is_available() -> bool:
    """Return True iff aravis + PyGObject are importable AND at least one
    aravis camera is currently visible on the bus."""
    try:
        import gi
        gi.require_version("Aravis", "0.8")
        from gi.repository import Aravis
        Aravis.update_device_list()
        return Aravis.get_n_devices() > 0
    except Exception:
        return False


class AravisCameraCapture:
    """cv2.VideoCapture-lookalike backed by aravis."""

    def __init__(self, device_id: Optional[str] = None):
        # Local import so importing this module on Windows/Linux (where
        # aravis isn't installed) doesn't explode; caller is expected to
        # gate creation on is_available().
        import gi
        gi.require_version("Aravis", "0.8")
        from gi.repository import Aravis
        self._Aravis = Aravis

        self._opened = False
        self._cam = None
        self._stream = None
        self._payload = 0
        self._width = 0
        self._height = 0
        self._fps = 30.0
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        try:
            self._cam = Aravis.Camera.new(device_id)
        except Exception:
            return

        try:
            self._width, self._height = self._cam.get_sensor_size()
            # Keep whatever pixel format the camera boots in -- setting it
            # here needs the camera not to be streaming (fine) but some
            # setups have already-open handles that make SetValue fail
            # with USB3Vision access-denied. We debayer / promote at read
            # time so this backend works with Mono8 or BayerRG8 either way.
            self._payload = self._cam.get_payload()
            self._pixel_format_str = self._cam.get_pixel_format_as_string()
            self._stream = self._cam.create_stream(None, None)
            for _ in range(6):
                self._stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))
            self._cam.start_acquisition()
            self._opened = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self._opened = False

    # ------------------------------------------------------ cv2.VideoCapture API

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop, value) -> bool:
        if not self._opened:
            return False
        try:
            if prop == cv2.CAP_PROP_FPS:
                self._cam.set_frame_rate(float(value))
                self._fps = float(value)
                return True
            if prop == cv2.CAP_PROP_EXPOSURE:
                # Camera takes microseconds; the app already passes us that.
                self._cam.set_exposure_time(float(value))
                return True
            if prop == cv2.CAP_PROP_AUTO_WB:
                # Monochrome sensor: no-op, but pretend success so callers
                # don't panic.
                return True
        except Exception:
            return False
        return False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return int(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return int(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0  # live stream -- no total
        return 0.0

    def read(self):
        """Return (ret, frame). Frame is BGR uint8."""
        if not self._opened:
            return False, None
        with self._latest_lock:
            frame = self._latest_frame
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cam is not None:
            try:
                self._cam.stop_acquisition()
            except Exception:
                pass
        self._opened = False

    # ------------------------------------------------------ internals

    def _buffer_to_bgr(self, buf) -> Optional[np.ndarray]:
        """Copy an aravis buffer into a fresh BGR uint8 numpy array."""
        try:
            data = buf.get_data()
        except Exception:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size != self._width * self._height and arr.size < self._width * self._height:
            # Truncated / bad buffer; skip.
            return None
        # For Bayer, aravis exposes the raw single-channel array. We debayer
        # only if the format string obviously starts with "Bayer"; otherwise
        # treat as Mono8 (or its multi-byte relatives).
        try:
            arr = arr[: self._width * self._height].reshape(
                (self._height, self._width)
            )
        except Exception:
            return None

        fmt = self._pixel_format_str.lower() if self._pixel_format_str else ""
        if fmt.startswith("bayer"):
            # Debayer by convention. FLIR default is BG so try BayerBG2BGR;
            # this is a heuristic and may need per-model tuning.
            try:
                return cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)
            except Exception:
                pass
        # Mono / anything else -> replicate grayscale into three channels
        # so downstream cv2.cvtColor(BGR2GRAY) doesn't crash.
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    def _run(self):
        Aravis = self._Aravis
        while not self._stop_event.is_set():
            try:
                buf = self._stream.try_pop_buffer()
            except Exception:
                buf = None
            if buf is None:
                time.sleep(0.005)
                continue
            try:
                if buf.get_status() == Aravis.BufferStatus.SUCCESS:
                    frame = self._buffer_to_bgr(buf)
                    if frame is not None:
                        with self._latest_lock:
                            self._latest_frame = frame
            finally:
                # Return the buffer to the pool so aravis can reuse it
                self._stream.push_buffer(buf)
