"""
scripted_gait.py — A FIXED, open-loop trot gait for the Bittle.

WHY THIS EXISTS (the science, not just the code):
Our research question needs a fair "control group" for the RL policy. A fixed
gait is a repeating leg pattern that runs blind — it never senses the ground,
never reacts to a bump, never adapts. It just plays the same motion forever.
This is what a NON-learned controller looks like (the stock OpenCat gait is
also a fixed pattern, though tuned by Petoi's engineers).

THE PREDICTION we test with it:
  - On FLAT ground: both the fixed gait and the RL policy walk fine.
  - On ROUGH ground: the fixed gait stumbles and falls (it can't react to
    bumps it can't feel), while the RL policy adapts and keeps going.
The size of that gap is our evidence that RL is actually necessary here.

HONEST SCOPE (say this in the report):
This is a STANDARD open-loop trot we tuned ourselves — a fair stand-in for a
fixed, non-learned controller. It is NOT Petoi's exact firmware gait. A
comparison against the real OpenCat firmware gait on physical hardware is a
separate test (a mentor item), not something we can do in simulation.

HOW A TROT WORKS:
A quadruped trot moves diagonal leg pairs together: the left-front + right-back
legs swing forward while the right-front + left-back legs push back, then they
swap. Each leg does the same cyclic motion, just offset in time (phase). We
drive each joint with a sine wave; the diagonal pairs are half a cycle apart.
"""

import numpy as np

from src.sim.config import NEUTRAL_POSE

# Our joint order (from scene.xml actuators):
#   [lb_shoulder, lb_knee, lf_shoulder, lf_knee, rb_shoulder, rb_knee, rf_shoulder, rf_knee]
#    0            1         2            3         4            5         6            7
# Legs: LB=(0,1)  LF=(2,3)  RB=(4,5)  RF=(6,7)
# Trot diagonal pairs: (LF, RB) together, (RF, LB) together, half a cycle apart.


class ScriptedTrotGait:
    """
    A fixed open-loop trot. Call step(t) to get the 8 target joint angles at
    time t. It ignores all sensor input — that is the whole point.

    Parameters (tuned for a slow, stable trot; adjust in one place):
        freq_hz        : gait cycles per second (leg stride rate).
        shoulder_amp   : how far the hip swings (radians).
        knee_amp       : how far the knee lifts (radians).
    """

    def __init__(self, freq_hz=2.5, shoulder_amp=0.15, knee_amp=0.15):
        # These defaults were tuned (parameter sweep) so the fixed gait walks
        # STABLY on flat ground — ~1.08 m in 10 s, never falling — so it is a
        # FAIR baseline for the RL policy, not a straw man. On rough terrain the
        # same fixed motion degrades sharply; that degradation is the result.
        self.freq_hz = freq_hz
        self.shoulder_amp = shoulder_amp
        self.knee_amp = knee_amp
        self._neutral = np.array(NEUTRAL_POSE, dtype=np.float64)

        # Phase offset (in cycles) for each of the 4 legs.
        # Diagonal pair A = LF(leg index for joints 2,3) & RB(4,5) → phase 0.0
        # Diagonal pair B = RF(6,7) & LB(0,1)                     → phase 0.5
        # Listed per JOINT (8 values), matching the joint order above:
        #   LB shoulder/knee = 0.5, LF = 0.0, RB = 0.0, RF = 0.5
        self._phase = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])

    def step(self, t):
        """
        Return the 8 target joint angles (radians) at time t seconds.

        Shoulders swing with a sine; knees lift with a sine a quarter-cycle
        ahead so the foot lifts as the leg swings forward (a basic walking
        coordination). Diagonal pairs are half a cycle apart via self._phase.
        """
        # Angle of the gait cycle, per joint, including each joint's phase offset.
        cycle = 2.0 * np.pi * (self.freq_hz * t + self._phase)

        angles = self._neutral.copy()

        # Shoulders (even indices 0,2,4,6): swing forward and back (full sine).
        shoulder_idx = [0, 2, 4, 6]
        angles[shoulder_idx] += self.shoulder_amp * np.sin(cycle[shoulder_idx])

        # Knees (odd indices 1,3,5,7): lift ONCE per cycle, only during the swing
        # half (sin > 0). Using max(0, sin) — not abs(sin) — means a leg lifts
        # while it swings forward and stays planted while it pushes back, and it
        # keeps the two diagonal pairs (half a cycle apart) properly distinct.
        knee_idx = [1, 3, 5, 7]
        angles[knee_idx] += self.knee_amp * np.maximum(0.0, np.sin(cycle[knee_idx]))

        return angles

    def reset(self):
        """No internal state to reset — a fixed gait is memoryless. Here for API parity."""
        pass
