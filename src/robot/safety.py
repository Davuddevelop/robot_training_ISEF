"""
safety.py — Safety checks for real-robot operation.

Every control step passes through this module before a command is sent.
If any check fails, the control loop stops and the robot stands still.

Rules:
  1. If the robot tilts too far → stop (it has fallen or is about to).
  2. If the user presses Ctrl+C → stop gracefully (servo to neutral first).
  3. If IMU data is missing for too many steps in a row → stop (sensor fault).
"""

import numpy as np
from src.sim.config import HARDWARE


class SafetyMonitor:
    """
    Checks each control step for unsafe conditions.

    Usage:
        safety = SafetyMonitor()
        if not safety.is_safe(imu_data):
            break  # stop the loop
    """

    def __init__(self, max_missing_imu: int = 10):
        """
        Args:
            max_missing_imu: how many consecutive steps without IMU data
                             before we call it a sensor fault and stop.
        """
        self._fall_angle     = HARDWARE["fall_stop_angle"]  # radians
        self._stop_requested = False
        self._missing_imu    = 0
        self._max_missing    = max_missing_imu

    # ------------------------------------------------------------------ public

    def is_safe(self, imu_data: dict | None) -> bool:
        """
        Returns True if it is safe to send the next command.

        Call this every control step. It combines all checks.
        """
        if self._stop_requested:
            return False

        if not self._check_imu_available(imu_data):
            return False

        if imu_data is not None and not self._check_upright(imu_data):
            return False

        return True

    def request_stop(self):
        """Call this to trigger a clean stop from another thread."""
        self._stop_requested = True

    # ------------------------------------------------------------------ private

    def _check_imu_available(self, imu_data: dict | None) -> bool:
        if imu_data is None:
            self._missing_imu += 1
            if self._missing_imu >= self._max_missing:
                print(f"Safety: IMU missing for {self._missing_imu} steps — stopping.")
                return False
        else:
            self._missing_imu = 0
        return True

    def _check_upright(self, imu_data: dict) -> bool:
        """
        Check if the robot is within the safe tilt angle.

        We use the larger of |roll| and |pitch| as the tilt measure.
        If either exceeds fall_stop_angle, the robot is considered fallen.
        """
        tilt = max(abs(imu_data.get("roll", 0.0)),
                   abs(imu_data.get("pitch", 0.0)))
        if tilt > self._fall_angle:
            deg = np.degrees(tilt)
            print(f"Safety: tilt {deg:.1f}° exceeds limit "
                  f"{np.degrees(self._fall_angle):.0f}° — stopping.")
            return False
        return True
