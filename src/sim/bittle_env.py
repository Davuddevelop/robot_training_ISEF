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
    NEUTRAL_POSE, REWARD, DOMAIN_RAND, TERRAIN, COMMAND,
    SIM_TIMESTEP, CONTROL_TIMESTEP, EPISODE_LENGTH_STEPS,
    BITTLE_MODEL_PATH, BITTLE_ROUGH_MODEL_PATH,
)

# The model faces +Y, so forward progress is movement along the world Y axis.
FORWARD_AXIS = 1  # 0 = x, 1 = y, 2 = z


class BittleEnv(gym.Env):
    """Gymnasium environment for the Petoi Bittle quadruped robot in MuJoCo."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, domain_rand=False, terrain=False,
                 terrain_difficulty=None):
        """
        Args:
            render_mode: "human" opens a viewer window; None = headless (faster).
            domain_rand: if True, randomizes physics each episode (for the
                         robustness experiments). Default False = clean baseline,
                         which is also experimental condition (1) "no randomization".
            terrain: if True, load the heightfield scene (scene_rough.xml) and
                     generate uneven ground each episode. False = flat scene.xml
                     (the experimental control).
            terrain_difficulty: 0.0 = flat, 1.0 = max bumps. If None, uses
                     TERRAIN["difficulty"] from config. The training curriculum
                     overrides this via set_terrain_difficulty().
        """
        super().__init__()
        self.render_mode = render_mode
        self.domain_rand = domain_rand
        self.terrain = terrain
        self._terrain_difficulty = (
            TERRAIN["difficulty"] if terrain_difficulty is None else terrain_difficulty
        )

        # --- Spaces (the contract with SB3) ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # Policy outputs in [-1, 1] (what SB3 learns best with); we scale to
        # ±ACTION_LIMIT radians inside step(). Keeps the network's job simple.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        # --- Load the MuJoCo model. Rough scene has the heightfield; flat does not. ---
        model_path = BITTLE_ROUGH_MODEL_PATH if terrain else BITTLE_MODEL_PATH
        self._mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self._mj_model.opt.timestep = SIM_TIMESTEP
        self._mj_data = mujoco.MjData(self._mj_model)

        # Heightfield bookkeeping (only when terrain is on).
        if self.terrain:
            self._hfield_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_HFIELD, "rough")
            self._hfield_nrow = int(self._mj_model.hfield_nrow[self._hfield_id])
            self._hfield_ncol = int(self._mj_model.hfield_ncol[self._hfield_id])
            self._hfield_radius_x = float(self._mj_model.hfield_size[self._hfield_id][0])
            self._hfield_radius_y = float(self._mj_model.hfield_size[self._hfield_id][1])

            # Debris (rubble) geoms declared in scene_rough.xml: debris_0..debris_N-1.
            # We look each one up by name once here; every reset just repositions them.
            self._debris_ids = []
            i = 0
            while True:
                gid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"debris_{i}")
                if gid == -1:
                    break
                self._debris_ids.append(gid)
                i += 1

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
        # Forward speed (m/s) this episode is being asked for. Re-sampled every
        # reset, and fed to the policy as obs[23] -- a reward for matching a
        # target the network cannot see would be unlearnable noise.
        self._vx_command = float(np.mean(COMMAND["vx_range"]))
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

    # ------------------------------------------------------------------ terrain
    def set_terrain_difficulty(self, difficulty):
        """
        Set the terrain difficulty (0.0 = flat, 1.0 = max bumps).

        The training curriculum calls this to make the ground harder as the
        robot improves. Takes effect on the NEXT reset (bumps regenerate then).
        """
        self._terrain_difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def get_terrain_difficulty(self):
        return self._terrain_difficulty

    def _generate_terrain(self):
        """
        Fill the heightfield grid with fresh random bumps for this episode.

        Steps:
          1. Draw random noise in [0, 1] for each of the nrow x ncol cells.
          2. Smooth it — FEWER passes as difficulty rises, so easy/early
             terrain is gentle rolling ground (learnable warmup) and hard/late
             terrain is genuinely jagged rock (less blur = sharper, more
             irregular surface). At difficulty 0 the smoothing choice doesn't
             matter anyway (step 3 zeroes it out).
          3. Scale by terrain_difficulty: 0.0 wipes it flat, 1.0 = full height.
          4. Flatten a spawn patch at the centre so the robot starts on level
             ground and doesn't topple before it can take a step.
          5. Write the grid into the model and push it to the renderer.

        MuJoCo stores hfield_data as values in [0, 1]; the actual bump height in
        metres is that value times z_top (set in scene_rough.xml).
        """
        nrow, ncol = self._hfield_nrow, self._hfield_ncol

        # 1. Random noise.
        raw = self.np_random.uniform(0.0, 1.0, size=(nrow, ncol))

        # 2. Build the fully-smoothed version ONCE, then blend towards the raw
        # noise as difficulty climbs. Texture goes from gentle rolling ground at
        # difficulty 0 to jagged rock at difficulty 1.
        #
        # This replaces `passes = round(smoothing_passes * (1 - difficulty))`,
        # which was a STEP function: round() gave 3 passes below d≈0.17, 2 up to
        # 0.5, 1 up to 0.84, and 0 above 0.84. So the ground barely changed for
        # most of the curriculum and then jumped brutally at one specific rung --
        # measured mean ground slope went 5.4° at d=0.8 to 15.1° at d=0.9, nearly
        # TRIPLING in a single 0.1 curriculum step. Training runs stalled at
        # exactly d=0.90, which was not a coincidence.
        #
        # Blending makes difficulty a smooth, monotone axis while leaving both
        # endpoints untouched: d=0 is still flat, d=1 is still raw noise. The
        # terrain is not made easier -- the SCALE is made linear, so "difficulty
        # 0.5" now means something halfway between the extremes.
        smooth = raw.copy()
        for _ in range(TERRAIN["smoothing_passes"]):
            smooth = self._smooth(smooth)

        d = self._terrain_difficulty
        field = (1.0 - d) * self._normalise01(smooth) + d * self._normalise01(raw)

        # 3. Scale amplitude by difficulty (this is what makes d=0 perfectly flat).
        field *= d

        # 4. Flatten a spawn patch at the grid centre (robot starts here).
        cr, cc = nrow // 2, ncol // 2
        pad = max(2, nrow // 12)
        field[cr - pad:cr + pad, cc - pad:cc + pad] = 0.0

        # 5. Write to the model. hfield_data is a flat (nrow*ncol,) array.
        adr = int(self._mj_model.hfield_adr[self._hfield_id])
        self._mj_model.hfield_data[adr:adr + nrow * ncol] = field.flatten().astype(np.float32)

        # Push updated terrain to the viewer if one is open.
        if self._viewer is not None:
            try:
                mujoco.mjr_uploadHField(self._mj_model, self._viewer._sim.context,
                                        self._hfield_id)
            except Exception:
                pass

        # Scatter rubble chunks on top — this is what gives "destroyed house"
        # terrain its actual obstacles, not just rolling ground. Uses the same
        # (pre-flatten) height grid so chunks rest roughly on the surface.
        if TERRAIN["debris"]["enabled"] and self._debris_ids:
            self._place_debris(field)

    def _place_debris(self, field):
        """
        Scatter rubble chunks onto the terrain (or hide them, if not needed).

        Count and size both scale with terrain_difficulty — 0 chunks at
        difficulty 0.0 (flat baseline stays a clean experimental control),
        up to config["max_count"] chunks at difficulty 1.0. Chunks not in use
        this episode are parked far away, underground, so they can't be
        touched or seen.

        Each active chunk gets a random (x, y) on the terrain patch — away
        from the flattened spawn area so the robot always starts clear — and
        its height is read off the SAME height grid we just generated, so it
        rests roughly on the ground instead of floating or clipping through it.
        """
        cfg = TERRAIN["debris"]
        active = int(round(self._terrain_difficulty * cfg["max_count"]))
        size_lo, size_hi = cfg["size_range"]
        rx, ry = self._hfield_radius_x, self._hfield_radius_y
        nrow, ncol = field.shape
        max_bump = TERRAIN["max_bump_height"]
        spawn_clear_radius = 0.3  # metres — keep clear of the robot's start point

        for idx, gid in enumerate(self._debris_ids):
            if idx >= active:
                # Not needed this episode: hide it far away, underground.
                self._mj_model.geom_pos[gid] = [50.0 + idx, 0.0, -5.0]
                continue

            # Random (x, y), resampling a few times to avoid the spawn patch.
            x, y = 0.0, 0.0
            for _attempt in range(8):
                x = self.np_random.uniform(-rx * 0.85, rx * 0.85)
                y = self.np_random.uniform(-ry * 0.85, ry * 0.85)
                if np.hypot(x, y) > spawn_clear_radius:
                    break

            # Sample this episode's terrain height under (x, y) so the chunk
            # sits on the ground rather than floating above or buried below it.
            col = int(np.clip((x / rx + 1.0) / 2.0 * (ncol - 1), 0, ncol - 1))
            row = int(np.clip((y / ry + 1.0) / 2.0 * (nrow - 1), 0, nrow - 1))
            ground_z = float(field[row, col]) * max_bump

            half_extents = self.np_random.uniform(size_lo, size_hi, size=3)
            half_extents[2] = min(half_extents[2], 0.03)  # keep chunks step-over-able

            self._mj_model.geom_size[gid] = half_extents
            self._mj_model.geom_pos[gid] = [x, y, ground_z + half_extents[2]]

            # geom_rbound is the radius of the bounding sphere MuJoCo's broadphase
            # uses to decide which pairs are even WORTH checking for contact. It is
            # computed once at model-compile time from the size in the XML, and is
            # NOT recalculated when we resize a geom at runtime like we just did.
            # Leaving it stale means a chunk we grew is still tested against its old,
            # smaller sphere, so contacts near its corners can be silently skipped --
            # the robot clips through the very obstacles that make the terrain hard.
            # Recomputing it here keeps collisions honest.
            self._mj_model.geom_rbound[gid] = float(np.linalg.norm(half_extents))

    @staticmethod
    def _normalise01(field):
        """Rescale a field to span exactly [0, 1] (blurring shrinks its range)."""
        out = field - field.min()
        peak = out.max()
        return out / peak if peak > 1e-8 else out

    @staticmethod
    def _smooth(field):
        """Average each cell with its 4 neighbours (a cheap blur, edges clamped)."""
        out = field.copy()
        out[1:-1, 1:-1] = (
            field[1:-1, 1:-1]
            + field[:-2, 1:-1] + field[2:, 1:-1]
            + field[1:-1, :-2] + field[1:-1, 2:]
        ) / 5.0
        return out

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM)
        self._prev_action_for_obs = np.zeros(ACTION_DIM)
        self._last_command = self._neutral.copy()
        self._gait_phase = 0.0

        # Draw this episode's speed order. Varying it (rather than always asking
        # for the same speed) stops the policy memorising one fixed gait and
        # forces it to actually READ the command it is given.
        lo, hi = COMMAND["vx_range"]
        self._vx_command = float(self.np_random.uniform(lo, hi))

        # Generate fresh uneven terrain for this episode (if terrain is on).
        # Done before resetting the robot so it spawns onto the new ground.
        if self.terrain:
            self._generate_terrain()

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
        if terminated:
            # A one-time, deliberately large cost for actually falling -- on
            # top of losing all future alive_bonus/forward reward. Without
            # this, sprinting recklessly and falling near episode-end can
            # still out-score walking carefully and surviving, since falling
            # only forfeits FUTURE reward, never costs anything concrete.
            reward += REWARD["fall_penalty"]
        truncated = self._step_count >= EPISODE_LENGTH_STEPS

        forward_velocity = float(self._mj_data.qvel[FORWARD_AXIS])
        self._prev_action = action.copy()

        info = {
            "step": self._step_count,
            "forward_velocity": forward_velocity,
            "forward_position": float(self._mj_data.qpos[FORWARD_AXIS]),
            "height": float(self._mj_data.qpos[2]),
            # Velocity tracking is now the primary metric: every condition gets
            # the same command, so "how well did it obey?" is a cleaner
            # comparison than raw distance, which conflates wanting to go fast
            # with being able to.
            "vx_command": self._vx_command,
            "vx_error": abs(self._vx_command - forward_velocity),
            "terrain_difficulty": self._terrain_difficulty,
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
        obs[23] = self._vx_command               # the speed we are asking for
        return obs

    # ------------------------------------------------------------------ reward
    def _compute_reward(self, action):
        """
        Velocity-TRACKING reward: pay for matching the commanded speed.

        The old reward was `forward_velocity_coeff * speed + alive_bonus`, which
        had a fatal property: standing still still collected the alive bonus
        every step, so the best strategy was to survive without travelling.
        Measured from that config -- standing all 500 steps scored 500, walking
        and falling at step 50 scored 55. The policy was not broken; 500 > 55.

        The fix is to make the reward peak at a SPECIFIC speed instead of rising
        forever, so that being too SLOW is punished just like being too fast.
        With our tuned sigma (0.004), standing still while commanded 0.13 m/s
        scores about 0.03 out of a possible 1.0 -- there is no longer any
        meaningful reward for merely existing, so opting out stops paying.
        """
        forward_velocity = float(self._mj_data.qvel[FORWARD_AXIS])
        lateral_velocity = float(self._mj_data.qvel[0])   # X axis = sideways

        # --- The one positive term. Bounded to [0, 1], so nothing can drown it.
        # exp(-error^2 / sigma) is a smooth hill peaking at the commanded speed:
        # perfect tracking = 1.0, and it decays as the error grows. Unlike a
        # linear speed reward it can punish going too SLOW as well as too fast.
        vel_error = (self._vx_command - forward_velocity) ** 2
        reward = REWARD["tracking_lin_vel"] * float(
            np.exp(-vel_error / COMMAND["tracking_sigma"]))

        # Retained only for reproducing the old behaviour (coefficient is 0.0 now).
        reward += REWARD["forward_velocity_coeff"] * forward_velocity
        reward += REWARD["alive_bonus"]

        # Lateral drift: squared so small drifts are cheap, large crab-walks are costly.
        reward += REWARD["lateral_velocity_penalty"] * lateral_velocity ** 2

        # Tilt: continuous, from the horizontal part of projected gravity.
        # g[0]^2 + g[1]^2 = sin^2(tilt): 0 upright, 0.25 at 30deg, 0.5 at 45deg.
        # A smooth slope tells the policy WHICH WAY to correct; the old step
        # function only told it where the cliff was.
        gravity = self._projected_gravity()
        reward += REWARD["tilt_penalty_coeff"] * float(gravity[0] ** 2 + gravity[1] ** 2)

        # Energy proxy: penalize large offsets. Jitter proxy: penalize fast changes.
        reward += REWARD["action_size_penalty"] * np.sum(np.square(action))
        reward += REWARD["action_smoothness_penalty"] * np.sum(np.square(action - self._prev_action))

        # Clip the per-step total at zero BEFORE the fall penalty is added in
        # step(). If a living step could score negative, ending the episode
        # early becomes an improvement -- i.e. falling on purpose to stop the
        # bleeding. legged_gym does exactly this (`only_positive_rewards`).
        return float(max(reward, 0.0))

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
