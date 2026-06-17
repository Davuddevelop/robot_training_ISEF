"""
calibration.py — Convert simulation joint angles to real NyBoard servo commands.

The simulation and the real robot use different units and different zero-points:
  Sim:  radians, offset from 0.56 rad neutral, specific joint order, specific directions.
  Real: degrees, offset from the servo's calibrated standing position, NyBoard slot IDs.

This module is the translation layer between the two.
"""

import numpy as np

from src.sim.config import ACTION_LIMIT, CALIBRATION

# Unpack once at module load (faster than dict lookup every step).
SLOTS     = CALIBRATION["servo_slots"]           # NyBoard slot for each of our 8 joints
SIGNS     = np.array(CALIBRATION["direction_signs"], dtype=np.float64)
TRIM      = np.array(CALIBRATION["trim_deg"],        dtype=np.float64)
MAX_DEG   = CALIBRATION["max_joint_offset_deg"]


def action_to_servo_degrees(action: np.ndarray):
    """
    Convert a raw policy action ([-1, 1] per joint) to servo degree commands.

    Args:
        action: shape (8,), dtype float32 — the policy's output before any scaling.

    Returns:
        slot_ids  : list[int] — NyBoard servo slot numbers to command.
        angles_deg: list[int] — angle (degrees) for each slot.

    What this does step by step:
      1. Scale action from [-1, 1] to ±ACTION_LIMIT radians  (same as sim does).
      2. Convert radians → degrees.
      3. Flip signs for joints where sim and servo directions are opposite.
      4. Add per-servo trim (fine-tuning after physical calibration).
      5. Clip to the hardware-safe range.
      6. Round to integer degrees (NyBoard takes integers).
    """
    offset_rad = action * ACTION_LIMIT                   # step 1: ±0.5 rad
    offset_deg = np.degrees(offset_rad)                  # step 2: ±28.6°
    adjusted   = SIGNS * offset_deg + TRIM               # steps 3–4
    clipped    = np.clip(adjusted, -MAX_DEG, MAX_DEG)    # step 5
    angles_int = [int(round(a)) for a in clipped]        # step 6
    return SLOTS, angles_int


def neutral_servo_degrees():
    """
    Returns (slot_ids, angles) that place every joint at neutral (0° offset).
    Use this to stand the robot still before or after a run.
    """
    return SLOTS, [0] * 8
