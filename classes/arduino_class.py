"""ArduinoHandler for the 3D-tweezer 7-float protocol.

Packet: [I1, I2, I3, I4, I5, I6, acoustic_freq]

All six currents are signed PWM duty in [-1, 1]. The Arduino calls set*()
directly on each; sign chooses H-bridge polarity. Python side handles all
field synthesis via classes/field_synth.py.

Also exposes a `send_field(...)` compatibility method that takes the
legacy Bx/By/Bz + gradient + roll intent and synthesizes currents on the
fly. This is transitional -- used during the gui_functions refactor so
the old call sites keep working, then deleted in migration Step 4.
"""

from __future__ import annotations

import time
from typing import Sequence

from pySerialTransfer import pySerialTransfer as txfer
from pySerialTransfer.pySerialTransfer import InvalidSerialPort


class ArduinoHandler:
    PACKET_LABEL = "[I1, I2, I3, I4, I5, I6, acoustic_freq]"

    def __init__(self, printer):
        self.conn = None
        self.port = None
        self.printer = printer
        # Per-coil calibration gains applied to any send(). Owned here so
        # the calibration tab has one obvious place to write. Default 1.0
        # is neutral. Negative values invert an individual coil's polarity
        # (fix for a coil wound backward at build time).
        self.coil_gains = [1.0] * 6

    def connect(self, port) -> None:
        """Open a SerialTransfer connection. Idempotent and non-fatal on failure."""
        if port is None:
            self.printer("No port specified for Arduino, skipping connection")
            return
        if self.conn is not None:
            self.printer(
                f"Connection already initialized at port {self.port}, new port {port} ignored"
            )
            return
        try:
            self.conn = txfer.SerialTransfer(port)
            self.port = port
            self.conn.open()
            time.sleep(1)
            self.printer(f"Arduino Connection initialized using port {port}")
        except InvalidSerialPort:
            self.printer(f"Could not connect to arduino at {port}: invalid port")
            self.conn = None
            self.port = None
        except PermissionError as e:
            self.printer(
                f"Access denied to {port} ({e}). Close Arduino IDE Serial Monitor "
                f"or any other program using that port, then restart the app."
            )
            self.conn = None
            self.port = None
        except Exception as e:
            self.printer(f"Could not connect to arduino at {port}: {type(e).__name__}: {e}")
            self.conn = None
            self.port = None

    def send(self, currents: Sequence[float], acoustic_freq: float = 0.0) -> None:
        """Send the 7-float packet: 6 signed coil currents + acoustic freq.

        Applies self.coil_gains element-wise, then clamps to [-1, 1]. Passing
        pre-calibrated currents from the caller is fine (gains default to
        1.0 so this is a no-op unless the calibration tab changed them).
        """
        currents = list(currents)
        if len(currents) != 6:
            self.printer(f"send: expected 6 currents, got {len(currents)}")
            return
        currents = [max(-1.0, min(1.0, float(c) * float(g)))
                    for c, g in zip(currents, self.coil_gains)]
        data = [round(c, 3) for c in currents] + [float(acoustic_freq)]
        if self.conn is None:
            self.printer("No Connection:  " + self.PACKET_LABEL + " = " + str(data))
        else:
            message = self.conn.tx_obj(data)
            self.conn.send(message)
            # macOS's serial driver buffers small writes and doesn't push them
            # to the wire until the buffer fills. At the tracker's ~15 Hz send
            # rate that never happens naturally, so the Arduino sees nothing
            # even though every send() succeeds from Python's side. Force it.
            try:
                self.conn.connection.flush()
            except Exception:
                pass
            self.printer("Data Sent:  " + self.PACKET_LABEL + " = " + str(data))

    def send_field(self, Bx, By, Bz,
                   gradient_dir=(0.0, 0.0, 1.0), gradient_mag=0.0,
                   roll_axis=(0.0, 0.0, 1.0), roll_freq=0.0,
                   t=0.0, acoustic_freq=0.0,
                   gains=None) -> None:
        """High-level field intent -> per-coil currents via field_synth.

        This is the recommended entry point for gui_functions. It hides the
        matrix math behind a keyword-argument interface that mirrors the
        operator's mental model (uniform field, gradient, roll, acoustic).
        """
        # Import here so this module doesn't require numpy at import time on
        # test / offline runs where field_synth is exercised separately.
        from classes import field_synth
        I = field_synth.synthesize(
            uniform_B=(Bx, By, Bz),
            gradient_dir=gradient_dir,
            gradient_mag=gradient_mag,
            roll_axis=roll_axis,
            roll_freq_hz=roll_freq,
            t=t,
            gains=gains,
        )
        self.send(I, acoustic_freq)

    def close(self) -> None:
        if self.conn is not None:
            self.printer(f"Closing connection at port {self.port}")
            self.send([0.0] * 6, 0.0)
            self.conn.close()


if __name__ == "__main__":
    # Smoke test: fire each coil in turn, then zero.
    import sys
    PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem11301"
    arduino = ArduinoHandler(print)
    arduino.connect(PORT)
    time.sleep(1)

    for i in range(6):
        currents = [0.0] * 6
        currents[i] = 0.3
        print(f"Firing coil C{i + 1} at 30% duty for 1 s")
        arduino.send(currents, 0.0)
        time.sleep(1.0)

    arduino.send([0.0] * 6, 0.0)
    arduino.close()
