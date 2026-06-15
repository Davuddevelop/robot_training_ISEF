"""
bittle_env.py — The Gymnasium environment for the Bittle quadruped.

This file is the heart of the simulation. It wraps the MuJoCo physics
simulator in the Gymnasium interface that Stable-Baselines3 expects.

STRUCTURE: A Gymnasium environment is a Python class with exactly these methods:
    __init__  — sets up the environment (runs once)
    reset()   — starts a new episode, returns first observation
    step()    — takes one action, returns (observation, reward, done, info)
    close()   — cleanup

You will study the Gymnasium API in Phase 3 of your learning roadmap.
When you do, come back here — it will make immediate sense.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Our config file — all the numbers we defined live there.
from src.sim.config import (
    OBS_DIM, ACTION_DIM, ACTION_LIMIT,
    NEUTRAL_POSE, REWARD, DOMAIN_RAND,
    SIM_TIMESTEP, CONTROL_TIMESTEP, EPISODE_LENGTH_STEPS,
    BITTLE_MODEL_PATH,
)


class BittleEnv(gym.Env):
    """
    Gymnasium environment for the Petoi Bittle quadruped robot in MuJoCo.

    This class inherits from gym.Env, which means it MUST implement:
        reset(), step(), close()
    and MUST define:
        self.observation_space
        self.action_space

    Think of gym.Env as a contract: SB3 calls these methods in a specific
    order and expects specific return types. We fulfill that contract here.
    """

    # Gymnasium needs this to know what kind of rendering we support.
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, domain_rand=True):
        """
        Sets up the environment. Called once before training starts.

        Args:
            render_mode: "human" opens a viewer window; None = headless (faster).
            domain_rand: if True, randomizes physics parameters each episode.
                         Set to False for evaluation / real-robot deployment.
        """
        super().__init__()

        self.render_mode = render_mode
        self.domain_rand = domain_rand

        # --- OBSERVATION SPACE ---
        # Tells SB3 what shape and range observations have.
        # Box = a multi-dimensional array of continuous values.
        # low/high = the minimum and maximum value for each element.
        # dtype = float32 (32-bit float — standard for neural networks)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # --- ACTION SPACE ---
        # 8 joint angle offsets, each bounded to [-ACTION_LIMIT, +ACTION_LIMIT].
        self.action_space = spaces.Box(
            low=-ACTION_LIMIT,
            high=ACTION_LIMIT,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        # --- INTERNAL STATE ---
        # These track what's happening during an episode.
        self._step_count = 0               # how many steps taken in current episode
        self._prev_action = np.zeros(ACTION_DIM)   # action from last step (for smoothness penalty)
        self._gait_phase = 0.0             # counts 0→1→0→1... to provide rhythm signal

        # --- MuJoCo MODEL ---
        # Not loaded yet — we load it lazily on first reset() to keep __init__ fast.
        # TODO (Phase 2): load MuJoCo model here once you understand the MuJoCo API.
        self._mj_model = None
        self._mj_data = None
        self._viewer = None

    def reset(self, seed=None, options=None):
        """
        Starts a new episode. Called at the beginning of every training episode.

        Steps:
          1. Reset the MuJoCo simulation to a standing pose.
          2. If domain randomization is on, randomize physics parameters.
          3. Add small random noise to the starting joint angles.
          4. Return the first observation.

        Returns:
            observation (np.ndarray): shape (OBS_DIM,) — what the robot "sees"
            info (dict): extra diagnostic info (can be empty)
        """
        super().reset(seed=seed)

        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM)
        self._gait_phase = 0.0

        # TODO (Phase 2): implement MuJoCo reset using mujoco.mj_resetData()
        # TODO (Phase 2): apply domain randomization if self.domain_rand is True
        # TODO (Phase 2): apply initial pose noise from DOMAIN_RAND["initial_pose"]

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """
        Executes one control step. Called at every timestep during training.

        The loop (SB3 calls this repeatedly):
          1. Receive 'action' (8 joint offsets) from the policy network.
          2. Convert to absolute joint angles: neutral_pose + action.
          3. Send to MuJoCo actuators.
          4. Step the physics forward (10 physics steps = 1 control step).
          5. Read new sensor data from MuJoCo.
          6. Compute the reward.
          7. Check if the episode ended (fallen or time limit).
          8. Return everything to SB3.

        Args:
            action (np.ndarray): shape (ACTION_DIM,) — joint angle offsets from policy

        Returns:
            observation  (np.ndarray): what the robot sees now
            reward       (float):      score for this step
            terminated   (bool):       True if robot fell — episode ends
            truncated    (bool):       True if time limit reached — episode ends
            info         (dict):       diagnostics (we log forward velocity here)
        """
        # Clip action to valid range (safety: don't damage real servos)
        action = np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)

        # TODO (Phase 2): compute target_angles = NEUTRAL_POSE + action
        # TODO (Phase 2): set mj_data.ctrl to target_angles
        # TODO (Phase 2): step physics with mujoco.mj_step() × (CONTROL_TIMESTEP / SIM_TIMESTEP) times
        # TODO (Phase 2): optionally add control latency from DOMAIN_RAND["control_latency"]

        self._step_count += 1
        self._gait_phase = (self._step_count * CONTROL_TIMESTEP) % 1.0

        reward = self._compute_reward(action)
        terminated = self._is_fallen()
        truncated = self._step_count >= EPISODE_LENGTH_STEPS
        observation = self._get_obs()

        self._prev_action = action.copy()

        info = {
            "step": self._step_count,
            # TODO (Phase 2): log forward_velocity from mj_data
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Reads sensor data from MuJoCo and packages it into the observation vector.

        Observation layout (see config.py OBS_LAYOUT for indices):
          [0:3]   IMU orientation (roll, pitch, yaw)
          [3:6]   IMU angular velocity
          [6:14]  last commanded joint angles (previous action in absolute terms)
          [14:22] action from two steps ago
          [22]    gait phase timer

        Returns:
            obs (np.ndarray): shape (OBS_DIM,) dtype float32
        """
        # TODO (Phase 2): read actual values from mj_data.sensordata / mj_data.qpos / mj_data.qvel
        # TODO (Phase 2): add IMU noise from DOMAIN_RAND["imu_noise"] if domain_rand is True

        # Placeholder: zeros until MuJoCo is wired up.
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[22] = self._gait_phase
        return obs

    def _compute_reward(self, action):
        """
        Computes the scalar reward for the current step.

        Reward = (forward_velocity × coeff) + alive_bonus
                 - (tilt_penalty if tilted)
                 - (action_size_penalty × ||action||)
                 - (smoothness_penalty × ||action - prev_action||)

        Each term comes from config.py REWARD — you control the weights there.

        Returns:
            reward (float)
        """
        # TODO (Phase 2): read forward_velocity from mj_data (x-axis velocity of base)
        forward_velocity = 0.0  # placeholder

        reward = 0.0

        # Forward progress (main objective)
        reward += REWARD["forward_velocity_coeff"] * forward_velocity

        # Alive bonus (encourages staying upright)
        reward += REWARD["alive_bonus"]

        # Tilt penalty (discourages falling)
        # TODO (Phase 2): read roll and pitch from mj_data and check against threshold
        # if abs(roll) > REWARD["tilt_threshold"] or abs(pitch) > REWARD["tilt_threshold"]:
        #     reward += REWARD["tilt_penalty"]

        # Action size penalty (proxy for energy/torque)
        reward += REWARD["action_size_penalty"] * np.sum(np.square(action))

        # Smoothness penalty (discourages jitter)
        reward += REWARD["action_smoothness_penalty"] * np.sum(np.square(action - self._prev_action))

        return float(reward)

    def _is_fallen(self):
        """
        Returns True if the robot has fallen (tilt exceeds fall_angle).
        Ends the episode early — no point continuing if the robot is on its side.

        Returns:
            fallen (bool)
        """
        # TODO (Phase 2): read roll and pitch from mj_data
        # return abs(roll) > REWARD["fall_angle"] or abs(pitch) > REWARD["fall_angle"]
        return False  # placeholder — never falls until MuJoCo is wired up

    def render(self):
        """Opens the MuJoCo viewer window to watch the robot."""
        # TODO (Phase 2): initialize mujoco.viewer.launch_passive() on first call
        pass

    def close(self):
        """Cleanup: close the viewer if it was opened."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
