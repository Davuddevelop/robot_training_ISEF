"""
serial_comm.py — Serial link between the laptop and the Petoi NyBoard.

Protocol: OpenCat ASCII serial (115200 baud, 8N1).
  'd' token: direct servo control.
       Format: "d <slot> <deg> <slot> <deg> ... \\n"
       The NyBoard moves each listed servo to the given degree immediately.
  'v' token: request one IMU reading.
       The NyBoard responds with a single line containing yaw, pitch, roll
       and gyroscope values.

All hardware-talking code lives here.  The control loop never touches
serial bytes directly — it only calls the methods below.
"""

import time
import numpy as np

try:
    import serial
except ImportError:
    raise ImportError(
        "pyserial is not installed.\n"
        "Run:  D:\\robot_venv\\Scripts\\pip.exe install pyserial"
    )


class NyBoardComm:
    """
    Handles the serial link to one NyBoard.

    Usage:
        comm = NyBoardComm("COM3")
        comm.send_joint_angles([10, 14, 8, 12, 11, 15, 9, 13], [0]*8)
        imu  = comm.request_imu()
        comm.close()
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05):
        """
        Open the serial port and wait for the NyBoard to boot.

        Args:
            port   : COM port string, e.g. "COM3" on Windows.
            baud   : must match the NyBoard firmware setting (default 115200).
            timeout: how long to wait for any single serial read (seconds).
        """
        print(f"  Connecting to NyBoard on {port} at {baud} baud …")
        self._ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)          # NyBoard needs ~1–2 s to boot after serial open
        self._ser.reset_input_buffer()
        print("  Connected.")

    # ---------------------------------------------------------------------- send

    def send_joint_angles(self, slot_ids: list, angles_deg: list):
        """
        Send a direct servo command using the OpenCat 'd' token.

        Args:
            slot_ids  : list of NyBoard servo slot numbers (integers).
            angles_deg: list of target angles in degrees (integers).

        Example command string sent over serial:
            "d 10 15 14 -20 8 10 12 -10 11 -15 15 20 9 -10 13 5\\n"
        """
        pairs = []
        for sid, ang in zip(slot_ids, angles_deg):
            pairs.extend([str(sid), str(ang)])
        cmd = "d " + " ".join(pairs) + "\n"
        self._ser.write(cmd.encode("ascii"))

    def stand_still(self):
        """Command all leg servos to 0° (neutral standing pose)."""
        from src.robot.calibration import neutral_servo_degrees
        slots, angles = neutral_servo_degrees()
        self.send_joint_angles(slots, angles)

    # ---------------------------------------------------------------------- read

    def request_imu(self) -> dict | None:
        """
        Ask the NyBoard for one IMU reading and parse the response.

        Returns a dict with keys: roll, pitch, yaw (radians)
                                   gx, gy, gz (radians/s, body frame)
        Returns None if the NyBoard did not respond or the line can't be parsed.

        Expected NyBoard response format (one line):
            "E: <yaw_deg> <pitch_deg> <roll_deg>  G: <gx_dps> <gy_dps> <gz_dps>"
        If your firmware outputs a different format, adjust _parse_imu_line() below.
        """
        self._ser.write(b"v\n")
        line = self._ser.readline().decode("ascii", errors="ignore").strip()
        return _parse_imu_line(line)

    # ---------------------------------------------------------------------- close

    def close(self):
        """Return the robot to standing pose and release the serial port."""
        try:
            self.stand_still()
            time.sleep(0.3)
        finally:
            self._ser.close()
            print("  Serial port closed.")


# -------------------------------------------------------------------------- helpers

def _parse_imu_line(line: str) -> dict | None:
    """
    Parse one line of NyBoard IMU output.

    Expected format:
        "E: 15.2 -3.1 2.4  G: 0.12 -0.05 0.33"
        where E values are yaw, pitch, roll in DEGREES
        and G values are gyro x, y, z in DEGREES/SECOND.

    Returns None if parsing fails (malformed line, timeout, etc.).

    NOTE: If your NyBoard firmware prints a different format, change
    only this function — nothing else needs to change.
    """
    try:
        parts = line.split()
        e_idx = parts.index("E:")
        yaw_deg   = float(parts[e_idx + 1])
        pitch_deg = float(parts[e_idx + 2])
        roll_deg  = float(parts[e_idx + 3])

        gx_dps = gy_dps = gz_dps = 0.0
        if "G:" in parts:
            g_idx  = parts.index("G:")
            gx_dps = float(parts[g_idx + 1])
            gy_dps = float(parts[g_idx + 2])
            gz_dps = float(parts[g_idx + 3])

        return {
            "roll":  np.radians(roll_deg),
            "pitch": np.radians(pitch_deg),
            "yaw":   np.radians(yaw_deg),
            "gx":    np.radians(gx_dps),
            "gy":    np.radians(gy_dps),
            "gz":    np.radians(gz_dps),
        }
    except (ValueError, IndexError):
        return None
