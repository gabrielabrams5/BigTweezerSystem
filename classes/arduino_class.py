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

    def __init__(self, printer, baud: int = 500000):
        self.conn = None
        self.port = None
        self.printer = printer
        self.baud = int(baud)
        # Per-coil calibration gains applied to any send(). Default 1.0 is
        # neutral. Negative values invert an individual coil's polarity.
        self.coil_gains = [1.0] * 6
        # Channel permutation: channel_map[i] = which physical driver logical
        # coil i is wired to. Default identity. Set from calibration.json.
        self.channel_map = [0, 1, 2, 3, 4, 5]

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
            self.conn = txfer.SerialTransfer(port, baud=self.baud)
            self.port = port
            self.conn.open()
            time.sleep(1)
            self.printer(
                f"Arduino Connection initialized using port {port} "
                f"at {self.baud} baud"
            )
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

        Applies self.coil_gains element-wise, then clamps. Range depends on
        field_synth.PULL_ONLY: under pull-only rigs we clip to [0, 1] so
        negative-gain-flipped currents don't try to reverse polarity (which
        would still attract the paramagnetic bead toward that coil).
        """
        currents = list(currents)
        if len(currents) != 6:
            self.printer(f"send: expected 6 currents, got {len(currents)}")
            return
        # Late import to avoid a circular hard-dep at import time.
        from classes.field_synth import PULL_ONLY
        lo = 0.0 if PULL_ONLY else -1.0
        # 1) Apply per-coil gains + clamp
        currents = [max(lo, min(1.0, float(c) * float(g)))
                    for c, g in zip(currents, self.coil_gains)]
        # 2) Route through the channel map so a wiring swap between the
        #    Arduino driver channels and physical coil positions is
        #    corrected in software. currents[i] is the *logical* current for
        #    coil i; sent[channel_map[i]] is the physical wire it goes to.
        sent = [0.0] * 6
        for i in range(6):
            sent[self.channel_map[i]] = currents[i]
        data = [round(c, 3) for c in sent] + [float(acoustic_freq)]
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
