"""
control_loop.py — Real-time control loop for the physical Bittle.

This is the core of Phase 3.  It does exactly what the simulation does
in bittle_env.py — build the 23-element observation, run the policy,
apply the action — but on real hardware instead of a virtual model.

Loop: read IMU → build obs → normalize → predict → send servos → sleep → repeat
Rate: 50 Hz (CONTROL_TIMESTEP = 0.02 s), matching the simulation training rate.
"""

import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.sim.config import (
    OBS_DIM, ACTION_DIM, ACTION_LIMIT, CONTROL_TIMESTEP, HARDWARE, COMMAND,
)
from src.sim.bittle_env import BittleEnv
from src.robot.calibration import action_to_servo_degrees
from src.robot.serial_comm import NyBoardComm
from src.robot.safety import SafetyMonitor


# -------------------------------------------------------------------------- IMU

def imu_to_projected_gravity(imu: dict) -> np.ndarray:
    """
    Convert real IMU roll and pitch angles to the 'projected gravity' vector.

    This must match exactly what the simulation computes in _projected_gravity().
    The policy was trained on this representation — if we give it something
    different here, it will behave as if the robot is in an orientation it
    has never seen.

    The projected gravity vector is [0, 0, -1] when perfectly upright, and
    tilts away from that as the robot leans.  It is immune to yaw drift,
    which makes it robust for real-world use.

    Math:
        gravity_world = [0, 0, -1]
        gravity_body  = R_body^T @ gravity_world
        For ZYX Euler (roll applied last):
            gx = -sin(pitch)
            gy =  sin(roll) * cos(pitch)
            gz = -cos(roll) * cos(pitch)
    """
    pitch, roll = imu["pitch"], imu["roll"]
    sp, cp = np.sin(pitch), np.cos(pitch)
    sr, cr = np.sin(roll),  np.cos(roll)
    return np.array([-sp, sr * cp, -cr * cp], dtype=np.float32)


# -------------------------------------------------------------------------- controller

class RealRobotController:
    """
    Loads a trained policy and runs it on the physical Bittle.

    The structure mirrors BittleEnv closely so the observation the policy
    receives here is as close as possible to what it saw during training.
    """

    def __init__(self, model_path, vecnorm_path, port: str):
        """
        Args:
            model_path  : path to the .zip policy file (without extension).
            vecnorm_path: path to vecnormalize.pkl saved during training.
            port        : serial port string, e.g. "COM3".
        """
        print("Loading policy …")
        self._model = PPO.load(str(model_path))

        # VecNormalize statistics MUST match training exactly.
        # We load them into a dummy env (no simulation runs — it's only used
        # for its observation mean/std arrays).
        dummy = DummyVecEnv([lambda: BittleEnv(domain_rand=False)])
        self._norm = VecNormalize.load(str(vecnorm_path), dummy)
        self._norm.training   = False   # don't update running stats during eval
        self._norm.norm_reward = False  # we don't need reward here
        print("Policy and normalization stats loaded.")

        # Hardware
        self._comm   = NyBoardComm(port, baud=HARDWARE["baud_rate"])
        self._safety = SafetyMonitor()

        # Per-episode state — mirror of BittleEnv's internal bookkeeping
        self._step_count          = 0
        self._prev_action         = np.zeros(ACTION_DIM, dtype=np.float32)
        self._prev_action_for_obs = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_command_rad    = np.zeros(ACTION_DIM, dtype=np.float32)

        # Forward speed (m/s) to ask the policy for. Defaults to the middle of
        # the range it was trained on. This is a throttle you can change at
        # runtime -- the policy was trained across a range of commands, so it
        # should obey a different one without retraining. Values far outside
        # COMMAND["vx_range"] were never seen in training and will not work.
        self.vx_command = float(np.mean(COMMAND["vx_range"]))

    # ------------------------------------------------------------------ obs

    def _build_obs(self, imu: dict | None) -> np.ndarray:
        """
        Build the 23-element observation vector from real sensor data.

        Layout (must match OBS_LAYOUT in config.py):
            [0:3]   projected gravity   (from IMU roll/pitch)
            [3:6]   angular velocity     (from IMU gyroscope)
            [6:14]  last commanded joint angles (radians, absolute)
            [14:22] action from two steps ago
            [22]    gait phase timer (0→1→0→1…)
            [23]    commanded forward speed (m/s)

        THIS MUST MATCH bittle_env._get_obs() EXACTLY. If the real robot builds
        its observation even slightly differently from the simulator the policy
        was trained in, the policy is being fed something it has never seen and
        sim-to-real fails for a reason that has nothing to do with physics.
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        if imu is not None:
            obs[0:3] = imu_to_projected_gravity(imu)
            obs[3:6] = [imu["gx"], imu["gy"], imu["gz"]]
        else:
            # Fallback: pretend robot is perfectly upright.
            obs[0:3] = [0.0, 0.0, -1.0]

        obs[6:14]  = self._last_command_rad
        obs[14:22] = self._prev_action_for_obs
        obs[22]    = (self._step_count * CONTROL_TIMESTEP) % 1.0
        # The speed we are asking the robot for. On hardware this is OUR choice,
        # not a measurement, so it needs no sensor -- set self.vx_command to
        # drive the robot faster or slower without retraining.
        obs[23]    = self.vx_command

        return obs

    # ------------------------------------------------------------------ loop

    def run(self, n_steps: int = 500):
        """
        Run the policy on the real robot for n_steps control steps.

        Default n_steps=500 → 500 × 0.02 s = 10 seconds of walking.

        Press Ctrl+C at any time for an emergency stop.
        The robot returns to the standing pose before the program exits.
        """
        dt = CONTROL_TIMESTEP  # 0.02 s = 20 ms per step

        print()
        print(f"Running for {n_steps} steps ({n_steps * dt:.0f} s). "
              "Press Ctrl+C to emergency-stop.")
        print()

        self._step_count          = 0
        self._prev_action[:]         = 0
        self._prev_action_for_obs[:] = 0
        self._last_command_rad[:]    = 0

        try:
            for step in range(n_steps):
                t_start = time.perf_counter()

                # ── 1. Read IMU ──────────────────────────────────────────
                imu = self._comm.request_imu()

                # ── 2. Safety check ──────────────────────────────────────
                if not self._safety.is_safe(imu):
                    print(f"Stopped at step {step} by safety monitor.")
                    break

                # ── 3. Build observation ─────────────────────────────────
                raw_obs  = self._build_obs(imu)
                norm_obs = self._norm.normalize_obs(raw_obs)

                # ── 4. Run policy ────────────────────────────────────────
                action, _ = self._model.predict(norm_obs, deterministic=True)

                # ── 5. Send to servos ────────────────────────────────────
                slots, angles_deg = action_to_servo_degrees(action)
                self._comm.send_joint_angles(slots, angles_deg)

                # ── 6. Bookkeeping ───────────────────────────────────────
                self._prev_action_for_obs = self._prev_action.copy()
                self._prev_action         = action.copy()
                self._last_command_rad    = action * ACTION_LIMIT
                self._step_count         += 1

                # ── 7. Sleep to hit 50 Hz ────────────────────────────────
                elapsed = time.perf_counter() - t_start
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                elif step % 50 == 0 and elapsed > dt * 1.1:
                    print(f"  Step {step}: loop slow by {(elapsed - dt)*1000:.1f} ms")

        except KeyboardInterrupt:
            print("\nCtrl+C received — stopping safely.")

        finally:
            print("Returning to standing pose …")
            self._comm.stand_still()
            time.sleep(0.5)
            print("Done.")

    # ------------------------------------------------------------------ cleanup

    def close(self):
        """Release the serial port."""
        self._comm.close()
