"""hal — hardware abstraction between control logic and the Arduino.

Everything above HAL (FieldSolver, MotionController, Supervisor, GUI) talks
in terms of a 6-vector of per-coil duties and an acoustic frequency. HAL
routes those through the channel map and hands them to ArduinoHandler.

Two implementations:

  * ``ArduinoHAL`` — real serial connection. Auto-discovers the port from
    ``serial.port_glob`` in config if none is provided.
  * ``PrintHAL`` — no hardware; prints packets to a log callback. Used by
    unit tests and for GUI development without a rig attached.

Both expose an identical interface so the rest of the stack doesn't care
which one is behind them.
"""

from __future__ import annotations

import glob
import time
from typing import Callable, Optional

import numpy as np

from classes.arduino_class import ArduinoHandler


class BaseHAL:
    def set_currents(self, i: np.ndarray, acoustic_freq: float = 0.0) -> None:
        raise NotImplementedError

    def read_temps(self) -> np.ndarray:
        """Placeholder for future thermistor support. Returns zeros."""
        return np.zeros(6)

    def estop(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PrintHAL(BaseHAL):
    """No-hardware HAL. Prints every packet to the log callback.

    Used in unit tests and when running the GUI on a laptop with no rig
    attached. Faithfully mimics ArduinoHAL's side-effects on the log so
    the operator sees the same "Data Sent:" lines.
    """

    def __init__(self, printer: Callable[[str], None] = print,
                 output_scale: float = 1.0):
        self.printer = printer
        self.last_currents = np.zeros(6)
        self.last_acoustic = 0.0
        self.estopped = False
        self.output_scale = float(output_scale)

    def set_currents(self, i: np.ndarray, acoustic_freq: float = 0.0) -> None:
        if self.estopped:
            return
        i = np.asarray(i, dtype=float).reshape(6) * self.output_scale
        self.last_currents = i.copy()
        self.last_acoustic = float(acoustic_freq)
        rounded = [round(float(c), 3) for c in i]
        self.printer(
            f"PrintHAL: I = {rounded}, acoustic = {acoustic_freq:.1f} Hz"
        )

    def set_output_scale(self, scale: float) -> None:
        self.output_scale = float(scale)

    def estop(self) -> None:
        self.estopped = True
        self.last_currents = np.zeros(6)
        self.printer("PrintHAL: E-STOP — currents zeroed")


class ArduinoHAL(BaseHAL):
    """Wraps ArduinoHandler with channel-map/gains routing.

    Note the per-coil gains are applied here **on top of** whatever
    FieldSolver already applied. This is redundant when both share the
    same config, so callers should set ``apply_gains_in_solver=True`` in
    the solver and pass identity gains here — or vice versa. HAL defaults
    to identity so the solver owns the gain math.
    """

    def __init__(self,
                 port: Optional[str] = None,
                 baud: int = 500000,
                 port_glob: str = "/dev/cu.usbmodem*",
                 channel_map: Optional[list] = None,
                 gains: Optional[list] = None,
                 output_scale: float = 1.0,
                 printer: Callable[[str], None] = print):
        self.printer = printer
        self.arduino = ArduinoHandler(printer, baud=baud)
        self.baud = baud
        self.channel_map = list(channel_map or [0, 1, 2, 3, 4, 5])
        self.gains = list(gains or [1.0] * 6)
        self.output_scale = float(output_scale)
        # Sync into the underlying handler so its send() path uses them too.
        self.arduino.channel_map = self.channel_map
        self.arduino.coil_gains = self.gains
        self.estopped = False

        resolved_port = port or self._autodiscover(port_glob)
        if resolved_port:
            self.arduino.connect(resolved_port)
        else:
            self.printer(
                f"ArduinoHAL: no port matched glob {port_glob!r}; "
                f"running in disconnected mode"
            )

    @staticmethod
    def _autodiscover(port_glob: str) -> Optional[str]:
        matches = sorted(glob.glob(port_glob))
        return matches[0] if matches else None

    def set_channel_map(self, channel_map: list) -> None:
        self.channel_map = list(channel_map)
        self.arduino.channel_map = self.channel_map

    def set_gains(self, gains: list) -> None:
        self.gains = list(gains)
        self.arduino.coil_gains = self.gains

    def set_output_scale(self, scale: float) -> None:
        self.output_scale = float(scale)

    def set_currents(self, i: np.ndarray, acoustic_freq: float = 0.0) -> None:
        if self.estopped:
            return
        # ArduinoHandler.send does channel_map + gains + clip + flush.
        scaled = np.asarray(i, dtype=float).reshape(6) * self.output_scale
        self.arduino.send(list(scaled), float(acoustic_freq))

    def estop(self) -> None:
        self.estopped = True
        self.printer("ArduinoHAL: E-STOP — zeroing coils")
        try:
            self.arduino.send([0.0] * 6, 0.0)
        except Exception as e:
            self.printer(f"ArduinoHAL: e-stop send failed: {e}")

    def reset_estop(self) -> None:
        """Operator-ack after e-stop. Must be an explicit call to arm again."""
        self.estopped = False

    def close(self) -> None:
        try:
            self.arduino.close()
        except Exception:
            pass


def build_from_config(config, printer: Callable[[str], None] = print,
                      force_print_only: bool = False) -> BaseHAL:
    """Return an ArduinoHAL if serial is available, else PrintHAL.

    ``force_print_only=True`` skips the serial attempt entirely — used by
    tests and by the GUI's "practice mode."
    """
    output_scale = float(config.get("limits.output_scale", 1.0))
    if force_print_only:
        return PrintHAL(printer=printer, output_scale=output_scale)
    port_glob = config.get("serial.port_glob", "/dev/cu.usbmodem*")
    baud = int(config.get("serial.baud", 500000))
    channel_map = config.get("calibration.channel_map", [0, 1, 2, 3, 4, 5])
    # Gains applied here are identity; solver owns the multiplication so
    # its residual calculation stays honest.
    return ArduinoHAL(
        port=None,
        baud=baud,
        port_glob=port_glob,
        channel_map=channel_map,
        gains=[1.0] * 6,
        output_scale=output_scale,
        printer=printer,
    )
