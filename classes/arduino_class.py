from pySerialTransfer import pySerialTransfer as txfer
from pySerialTransfer.pySerialTransfer import InvalidSerialPort
import time

class ArduinoHandler:
    """
    Handles connections and messaging to an Arduino.

    Attributes:
        conn:   PySerialTransfer connection; has None value when no successsful
                connection has been made
        port:   name of connection port currently being used; has None value when
                no successful port has been used
    """

    def __init__(self, printer):
        self.conn = None
        self.port = None
        self.printer = printer
        # Per-coil gain multipliers (applied on Arduino in normal-mode send()).
        # 1.0 means no scaling. Populated by the Calibration tab / load_gains().
        self.coil_gains = [1.0] * 6

        

    def connect(self, port: str) -> None:
        """
        Initializes a connection to an arduino at a specified port. If successful,
        the conn and port attributes are updated. If the port is unavailable or
        already claimed (e.g. Arduino IDE Serial Monitor still open, or a stale
        Python process holding the handle on Windows), we log and leave
        self.conn = None so the GUI can still run in offline mode.
        """
        if port is None:
            self.printer("No port specified for Arduino, skipping connection")
            return
        if self.conn is not None:
            self.printer(f"Connection already initialized at port {self.port}, new port {port} ignored")
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
   
    # Packet layout: order MUST match main_3DTweezers.ino action[0..16] unpacking.
    #   [0..9]   normal field packet
    #   [10]     calibration_mode (0 = normal, 1 = direct per-coil PWM)
    #   [11..16] coil_values[6] (gain multipliers in normal mode; direct PWM 0..1 in calibration mode)
    PACKET_LABEL = "[Bx, By, Bz, alpha, gamma, freq, psi, gradient, equal_field, acoustic_freq, cal_mode, c1..c6]"

    def _tx(self, data, tag):
        if self.conn is None:
            self.printer(f"No Connection:  {self.PACKET_LABEL} = {data}")
        else:
            message = self.conn.tx_obj(data)
            self.conn.send(message)
            self.printer(f"{tag}:  {self.PACKET_LABEL} = {data}")

    def send(self, Bx, By, Bz, alpha, gamma, freq, psi, gradient_status, equal_field_status, acoustic_freq) -> None:
        """Normal-mode 17-float packet. Uses self.coil_gains as the per-coil multipliers."""
        data = [
            round(float(Bx), 3), round(float(By), 3), round(float(Bz), 3),
            round(float(alpha), 3), round(float(gamma), 3), round(float(freq), 3),
            round(float(psi), 3),
            float(gradient_status), float(equal_field_status), float(acoustic_freq),
            0.0,  # calibration_mode
        ] + [float(g) for g in self.coil_gains]
        self._tx(data, "Data Sent")

    def send_calibration_pulse(self, coil_index: int, strength: float, acoustic_freq: float = 0.0) -> None:
        """Fire exactly one coil at `strength` (0.0..1.0) with all others off.
        Used by the Calibration tab to isolate one coil for magnetometer readings.
        `coil_index` is 0..5 for C1..C6.
        """
        if not 0 <= coil_index < 6:
            self.printer(f"send_calibration_pulse: coil_index {coil_index} out of range (0..5)")
            return
        strength = max(-1.0, min(1.0, float(strength)))
        coil_values = [0.0] * 6
        coil_values[coil_index] = strength
        data = [0.0] * 7 + [0.0, 0.0, float(acoustic_freq), 1.0] + coil_values
        self._tx(data, f"Cal Pulse C{coil_index + 1}={strength}")

    def send_calibration_all_off(self) -> None:
        """Zero all coils while still in calibration mode. Used to stop a pulse."""
        data = [0.0] * 7 + [0.0, 0.0, 0.0, 1.0] + [0.0] * 6
        self._tx(data, "Cal Off")

    def set_gains(self, gains) -> None:
        """Update per-coil normal-mode gain multipliers. Expects an iterable of 6 floats."""
        gains = list(gains)
        if len(gains) != 6:
            self.printer(f"set_gains: expected 6 values, got {len(gains)}")
            return
        self.coil_gains = [float(g) for g in gains]
        self.printer(f"Coil gains set to {self.coil_gains}")
    
    
    def close(self) -> None:
        """
        Closes the current connection, if applicable

        Args:
            None
        Returns:
            None
        """
        if self.conn is not None:
         
            self.printer(f"Closing connection at port {self.port}")
            self.send(0,0,0,0,0,0,0,0,0,0)
            self.conn.close()
   
            


if __name__ == "__main__":

    def tbprint(text):
        #print to textbox
        print(text)


    PORT = "/dev/cu.usbmodem11301"
    arduino = ArduinoHandler(tbprint)
    arduino.connect(PORT)
    time.sleep(1)

    arduino.send(0,0,0,0,0,0,0,0,0,0)
    print("sending")
    time.sleep(5)
    arduino.send(0,0,0,0,0,0,0,0,0,0)
    print("zeroing")
    arduino.close()
    
    
