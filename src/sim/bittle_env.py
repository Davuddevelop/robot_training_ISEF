"""
bittle_env.py — The Gymnasium environment for the Bittle quadruped.

This file is the heart of the simulation. It wraps the MuJoCo physics
simulator in the Gymnasium interface that Stable-Baselines3 expects.

STRUCTURE: A Gymnasium environment is a Python class with exactly these methods:
    __init__  — sets up the environment (runs once)
    reset()   — starts a new episode, returns first observation
    step()    — takes one action, returns (observation, reward, done, info)
    close()   — cleanup

The robot faces +Y (its head sits at +Y in the model), so "forward" is +Y.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from src.sim.config import (
    OBS_DIM, ACTION_DIM, ACTION_LIMIT,
    NEUTRAL_POSE, REWARD, DOMAIN_RAND,
    SIM_TIMESTEP, CONTROL_TIMESTEP, EPISODE_LENGTH_STEPS,
    BITTLE_MODEL_PATH,
)

# The model faces +Y, so forward progress is movement along the world Y axis.
FORWARD_AXIS = 1  # 0 = x, 1 = y, 2 = z


class BittleEnv(gym.Env):
    """Gymnasium environment for the Petoi Bittle quadruped robot in MuJoCo."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, domain_rand=False):
        """
        Args:
            render_mode: "human" opens a viewer window; None = headless (faster).
            domain_rand: if True, randomizes physics each episode (for the
                         robustness experiments). Default False = clean baseline,
                         which is also experimental condition (1) "no randomization".
        """
        super().__init__()
        self.render_mode = render_mode
        self.domain_rand = domain_rand

        # --- Spaces (the contract with SB3) ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # Policy outputs in [-1, 1] (what SB3 learns best with); we scale to
        # ±ACTION_LIMIT radians inside step(). Keeps the network's job simple.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        # --- Load the MuJoCo model (our scene.xml: body + servos + sensors) ---
        self._mj_model = mujoco.MjModel.from_xml_path(str(BITTLE_MODEL_PATH))
        self._mj_model.opt.timestep = SIM_TIMESTEP
        self._mj_data = mujoco.MjData(self._mj_model)

        # How many physics steps make one control step (0.02 / 0.002 = 10).
        self._n_substeps = int(round(CONTROL_TIMESTEP / SIM_TIMESTEP))

        # Cache the ids we read every step (looking them up by name is slow).
        self._root_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "root")
        self._home_key = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._gyro_adr = self._sensor_adr("angular_velocity")

        self._neutral = np.array(NEUTRAL_POSE, dtype=np.float64)
        self._ctrl_low = self._mj_model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self._mj_model.actuator_ctrlrange[:, 1].copy()

        # Save the "nominal" physics values so domain randomization always
        # scales from the original, never from an already-randomized value.
        self._base_body_mass = self._mj_model.body_mass.copy()
        self._base_force = self._mj_model.actuator_forcerange.copy()
        self._floor_geom = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._base_friction = self._mj_model.geom_friction.copy()

        # --- Episode state ---
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM)       # action from the previous step
        self._prev_action_for_obs = np.zeros(ACTION_DIM)
        self._last_command = self._neutral.copy()      # absolute angles we last commanded
        self._gait_phase = 0.0
        self._latency_steps = 0
        self._command_buffer = []

        self._viewer = None

    # ------------------------------------------------------------------ helpers
    def _sensor_adr(self, name):
        """Return (start_index, length) of a named sensor inside data.sensordata."""
        sid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(self._mj_model.sensor_adr[sid])
        dim = int(self._mj_model.sensor_dim[sid])
        return adr, dim

    def _projected_gravity(self):
        """
        Direction of gravity ('down') expressed in the robot's body frame.

        data.xmat is the 3x3 rotation of the root body (body axes in world).
        The body's up-axis in world is its 3rd column = (xmat[2], xmat[5], xmat[8]).
        Gravity in the body frame is therefore (-xmat[2], -xmat[5], -xmat[8]):
        [0, 0, -1] when upright, tilting away as the robot leans.
        """
        xmat = self._mj_data.xmat[self._root_id]
        return np.array([-xmat[2], -xmat[5], -xmat[8]], dtype=np.float64)

    def _uprightness(self):
        """xmat[8] = how aligned the body-up axis is with world-up. 1.0 = perfectly upright."""
        return float(self._mj_data.xmat[self._root_id][8])

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM)
        self._prev_action_for_obs = np.zeros(ACTION_DIM)
        self._last_command = self._neutral.copy()
        self._gait_phase = 0.0

        # Start from the home keyframe (neutral standing pose).
        mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, self._home_key)

        if self.domain_rand:
            self._apply_domain_randomization()

        # Small random offset on each starting joint angle (qpos[7:15] = 8 joints).
        if self.domain_rand and DOMAIN_RAND["initial_pose"]["enabled"]:
            std = DOMAIN_RAND["initial_pose"]["std"]
            self._mj_data.qpos[7:7 + ACTION_DIM] += self.np_random.normal(0, std, ACTION_DIM)

        # Set up the control-latency delay buffer for this episode.
        self._latency_steps = 0
        if self.domain_rand and DOMAIN_RAND["control_latency"]["enabled"]:
            lo, hi = DOMAIN_RAND["control_latency"]["range"]
            latency_secs = self.np_random.uniform(lo, hi)
            self._latency_steps = int(round(latency_secs / CONTROL_TIMESTEP))
        self._command_buffer = [self._neutral.copy()] * (self._latency_steps + 1)

        # Recompute derived quantities (positions, sensors) after our edits.
        mujoco.mj_forward(self._mj_model, self._mj_data)

        return self._get_obs(), {}

    def _apply_domain_randomization(self):
        """Randomize physics for this episode. These are the research variables."""
        m, rng = self._mj_model, self.np_random

        if DOMAIN_RAND["friction"]["enabled"]:
            lo, hi = DOMAIN_RAND["friction"]["range"]
            m.geom_friction[self._floor_geom, 0] = rng.uniform(lo, hi)

        if DOMAIN_RAND["body_mass"]["enabled"]:
            lo, hi = DOMAIN_RAND["body_mass"]["range"]
            m.body_mass[:] = self._base_body_mass * rng.uniform(lo, hi)

        if DOMAIN_RAND["motor_strength"]["enabled"]:
            lo, hi = DOMAIN_RAND["motor_strength"]["range"]
            m.actuator_forcerange[:] = self._base_force * rng.uniform(lo, hi)

    # ------------------------------------------------------------------ step
    def step(self, action):
        # Policy gives [-1, 1]; scale to a ±ACTION_LIMIT-radian offset.
        action = np.clip(action, -1.0, 1.0)
        offset = action * ACTION_LIMIT

        # Target absolute angles = neutral stance + the offset.
        target = np.clip(self._neutral + offset, self._ctrl_low, self._ctrl_high)

        # Control latency: push the new command in, use the delayed one.
        self._command_buffer.append(target)
        applied = self._command_buffer.pop(0)

        # Drive the servos and advance physics by one control step.
        self._mj_data.ctrl[:] = applied
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._mj_model, self._mj_data)

        # Bookkeeping for the observation (we feed COMMANDED angles, not measured).
        self._step_count += 1
        self._gait_phase = (self._step_count * CONTROL_TIMESTEP) % 1.0
        self._prev_action_for_obs = self._prev_action
        self._last_command = target

        reward = self._compute_reward(action)
        terminated = self._is_fallen()
        truncated = self._step_count >= EPISODE_LENGTH_STEPS

        forward_velocity = float(self._mj_data.qvel[FORWARD_AXIS])
        self._prev_action = action.copy()

        info = {
            "step": self._step_count,
            "forward_velocity": forward_velocity,
            "forward_position": float(self._mj_data.qpos[FORWARD_AXIS]),
            "height": float(self._mj_data.qpos[2]),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ obs
    def _get_obs(self):
        # Orientation: projected gravity (3), optionally noised like a real IMU.
        gravity = self._projected_gravity()
        ang_vel = self._mj_data.sensordata[self._gyro_adr[0]:
                                           self._gyro_adr[0] + self._gyro_adr[1]].copy()
        if self.domain_rand and DOMAIN_RAND["imu_noise"]["enabled"]:
            std = DOMAIN_RAND["imu_noise"]["std"]
            gravity = gravity + self.np_random.normal(0, std, 3)
            ang_vel = ang_vel + self.np_random.normal(0, std, 3)

        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[0:3] = gravity
        obs[3:6] = ang_vel
        obs[6:14] = self._last_command           # last commanded joint angles
        obs[14:22] = self._prev_action_for_obs   # the action before that
        obs[22] = self._gait_phase
        return obs

    # ------------------------------------------------------------------ reward
    def _compute_reward(self, action):
        forward_velocity = float(self._mj_data.qvel[FORWARD_AXIS])
        lateral_velocity = float(self._mj_data.qvel[0])   # X axis = sideways

        reward = REWARD["forward_velocity_coeff"] * forward_velocity
        reward += REWARD["alive_bonus"]

        # Lateral drift: squared so small drifts are cheap, large crab-walks are costly.
        reward += REWARD["lateral_velocity_penalty"] * lateral_velocity ** 2

        # Tilt penalty: how far from upright are we? (angle between body-up and world-up)
        tilt_angle = np.arccos(np.clip(self._uprightness(), -1.0, 1.0))
        if tilt_angle > REWARD["tilt_threshold"]:
            reward += REWARD["tilt_penalty"]

        # Energy proxy: penalize large offsets. Jitter proxy: penalize fast changes.
        reward += REWARD["action_size_penalty"] * np.sum(np.square(action))
        reward += REWARD["action_smoothness_penalty"] * np.sum(np.square(action - self._prev_action))
        return float(reward)

    def _is_fallen(self):
        tilt_angle = np.arccos(np.clip(self._uprightness(), -1.0, 1.0))
        too_tilted = tilt_angle > REWARD["fall_angle"]
        too_low = self._mj_data.qpos[2] < 0.04   # torso basically on the ground
        return bool(too_tilted or too_low)

    # ------------------------------------------------------------------ render
    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
