"""
config.py — The single source of truth for every number in this project.

Every design decision, every research variable, every hyperparameter lives here.
If a judge asks "why did you choose that value?" — the answer comes from this file.
If you want to run a different experimental condition — you change this file.

Read this file carefully. You should be able to explain every line.
"""

# ---------------------------------------------------------------------------
# OBSERVATION SPACE
# What the robot "senses" at each control step.
# This is the input to the neural network policy.
# Total: 23 numbers packed into a single array.
# ---------------------------------------------------------------------------

# Layout of the 23-element observation vector.
# Indices tell you where each piece of information lives in the array.
OBS_LAYOUT = {
    "imu_orientation":    (0, 3),   # "projected gravity": the direction of DOWN as seen
                                    # from the robot's own body. [0,0,-1] when perfectly
                                    # upright; tilts away as the robot leans. This is what
                                    # an IMU effectively gives you, and it makes sim-to-real
                                    # robust (no absolute-heading drift like yaw has).
    "imu_angular_vel":    (3, 6),   # how fast the robot is rotating on each axis (rad/s)
    "last_joint_angles":  (6, 14),  # the 8 joint angles we commanded last step
    "previous_action":    (14, 22), # the 8 joint angle offsets from the step before that
    "gait_phase_timer":   (22, 23), # a clock signal (0→1→0→1...) that helps the robot
                                    # learn rhythmic gaits — without this, walking tends
                                    # to be jerky rather than cyclic
}

OBS_DIM = 23  # total size of the observation vector

# Why these observations and not others?
# - IMU is the ONLY real sensor on the Bittle. Everything else comes from the sim or memory.
# - "Last joint angles" tells the policy what it commanded, not what the servos actually did.
#   (The servos may lag behind — this mismatch is one of our research variables.)
# - "Previous action" helps the policy learn smooth motion (avoids jitter).
# - Gait phase timer: without it, the policy has no sense of rhythm.

# ---------------------------------------------------------------------------
# ACTION SPACE
# What the robot does at each control step.
# These are offsets (in radians) from the neutral standing pose.
# ---------------------------------------------------------------------------

ACTION_DIM = 8          # 8 leg joints (4 legs × 2 joints each: hip + knee)
ACTION_LIMIT = 0.5      # maximum offset from neutral pose, in radians (~28 degrees)
                        # Larger values risk hardware damage on real servos.

# Neutral standing pose for each joint (radians), in the SAME order the actuators
# appear in scene.xml:
#   [lb_shoulder, lb_knee, lf_shoulder, lf_knee, rb_shoulder, rb_knee, rf_shoulder, rf_knee]
#   (l/r = left/right, b/f = back/front)
# 0.56 rad is the model's "home" keyframe — the pose where it stands stably.
# The policy's actions are offsets ADDED to this pose.
NEUTRAL_POSE = [0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56]
# NOTE: refine these once you measure the real Bittle's resting joint angles.

# ---------------------------------------------------------------------------
# REWARD FUNCTION
# The score the policy receives at each step.
# More reward → the policy does more of that behavior.
# This is where your scientific intuition goes into code.
# ---------------------------------------------------------------------------

REWARD = {
    # Positive: reward forward movement (m/s in the robot's forward direction).
    # This MUST dominate, or the policy just stands still to farm the alive bonus.
    # We learned this the hard way: with coeff=1.0 and alive=0.5, a 60k-step run
    # scored 250 reward but walked 0.03 m — it stood still. Raising this makes
    # walking clearly worth more than standing.
    "forward_velocity_coeff": 5.0,

    # Positive: bonus for each step it stays upright (does not fall). With
    # forward_velocity_coeff=5.0 a moving robot always out-scores a standing one,
    # so this no longer causes standing-still — instead it makes SURVIVING pay,
    # which stops the "lunge forward then fall over" behaviour (a 300k run moved
    # 0.36 m but fell after 70 of 500 steps when this was only 0.1).
    "alive_bonus": 0.5,

    # Negative: penalty for drifting sideways (lateral = X axis, not forward Y).
    # Without this the policy sometimes crab-walks to one side rather than going
    # straight forward, which looks bad and performs worse on real flat ground.
    # Coefficient scales the SQUARED lateral speed, so small drifts barely matter
    # but large crab-walk is strongly penalised.
    "lateral_velocity_penalty": -0.5,

    # Negative: penalty if the robot tilts too much.
    # Applied when roll or pitch exceeds this threshold (radians).
    "tilt_threshold": 0.5,       # ~28 degrees
    "tilt_penalty": -1.0,        # subtracted from reward when threshold exceeded

    # Negative: penalty for commanding large joint angles.
    # Acts as a proxy for torque/energy usage.
    # Discourages the policy from thrashing the servos.
    "action_size_penalty": -0.01,

    # Negative: penalty for changing joint angles too fast between steps.
    # Discourages jitter, produces smoother motion.
    "action_smoothness_penalty": -0.005,

    # Episode ends (robot "falls") if tilt exceeds this angle (radians).
    # ~45 degrees — if the robot tips this far, it has fallen.
    "fall_angle": 0.8,

    # Negative: ONE-TIME penalty applied the instant the robot actually falls
    # (in addition to losing all remaining alive_bonus + forward reward for the
    # rest of the episode). This is REWARD HACKING insurance: without it, a
    # policy that sprints recklessly and falls near the end can still out-score
    # a policy that walks carefully and survives, because falling only costs
    # "future" reward, never actual banked reward. We saw exactly this: a run
    # was crowned "best" at 0.846m distance despite falling in most episodes.
    # This penalty makes falling COST something concrete, not just forfeit
    # potential future gains, so "walk carefully, survive" clearly beats
    # "sprint and risk it" in expectation.
    "fall_penalty": -30.0,
}

# ---------------------------------------------------------------------------
# DOMAIN RANDOMIZATION
# These are your RESEARCH VARIABLES.
# Each factor is randomized within a range at the start of each episode.
# The ablation study turns these on/off one at a time to measure each one's impact.
# ---------------------------------------------------------------------------

DOMAIN_RAND = {
    # Ground friction coefficient (dimensionless).
    # Real floor varies: carpet ~0.8, tile ~0.4, wood ~0.6.
    "friction": {
        "enabled": True,
        "range": (0.3, 1.2),    # (min, max)
    },

    # Body mass multiplier (scales total robot mass).
    # Accounts for battery weight variation, payload uncertainty.
    "body_mass": {
        "enabled": True,
        "range": (0.8, 1.2),    # ±20% of nominal mass
    },

    # Motor strength multiplier (scales actuator force).
    # Real hobby servos vary significantly in actual torque output.
    "motor_strength": {
        "enabled": True,
        "range": (0.7, 1.3),    # ±30% of nominal strength
    },

    # Control latency (seconds).
    # The real serial link adds ~10–30ms delay.
    # The policy must be robust to commands arriving late.
    "control_latency": {
        "enabled": True,
        "range": (0.0, 0.03),   # 0 to 30ms
    },

    # IMU sensor noise (standard deviation in radians).
    # The NyBoard IMU is a cheap MEMS sensor — it drifts and jitters.
    "imu_noise": {
        "enabled": True,
        "std": 0.02,            # Gaussian noise added to orientation readings
    },

    # Initial pose variation (radians).
    # Each episode starts with joints randomly offset from neutral.
    # Teaches the policy to recover from imperfect starting positions.
    "initial_pose": {
        "enabled": True,
        "std": 0.1,             # Gaussian offset on each joint at episode start
    },
}

# ---------------------------------------------------------------------------
# TERRAIN (the core research variable: flat vs. uneven ground)
# This is what justifies using RL at all — the stock gait handles flat ground,
# but fails on the uneven terrain the RL policy learns to cross.
# ---------------------------------------------------------------------------

TERRAIN = {
    # The heightfield grid in scene_rough.xml is filled with random bumps at
    # every reset, scaled by the current difficulty (0.0 = flat, 1.0 = max).
    # "difficulty" here is the DEFAULT for a fixed-difficulty run; the training
    # curriculum overrides it and raises it automatically as the robot improves.
    "difficulty": 0.5,

    # The tallest possible bump (metres) at difficulty 1.0. Must match z_top in
    # scene_rough.xml's <hfield size="... z_top ...">. The Bittle is ~12 cm tall,
    # so 0.06 m (6 cm) bumps are genuinely hard terrain for it.
    "max_bump_height": 0.06,

    # Smoothness of the bumps AT DIFFICULTY 0 (bittle_env.py scales this down to
    # ZERO smoothing at difficulty 1.0, so the terrain goes from gentle rolling
    # ground at the start of the curriculum to genuinely jagged rock at the end).
    "smoothing_passes": 3,

    # Discrete rubble/debris chunks scattered on top of the rolling ground —
    # this is what makes the terrain read as "destroyed house" rubble rather
    # than plain hills, and gives the robot actual obstacles to step over.
    # We deliberately do NOT make the heightfield itself spiky (sharp cliffs
    # make MuJoCo contacts unstable) — discrete boxes are the safer way to add
    # real difficulty. Count and size both scale with terrain_difficulty.
    "debris": {
        "enabled": True,
        "max_count": 14,            # number of chunks placed at difficulty 1.0
        "size_range": (0.02, 0.05), # half-extent per axis, metres (2-5 cm chunks)
    },

    # --- Curriculum (used by the training callback) ---
    # Start flat, raise difficulty when the robot walks well, cap at max.
    "curriculum": {
        "enabled": True,
        "start_difficulty": 0.0,       # begin on flat ground
        "max_difficulty": 1.0,         # hardest terrain to reach
        "step": 0.1,                   # how much to raise difficulty each time
        # Raise difficulty when the mean forward distance (metres) over the
        # last evaluation exceeds this. Tuned so the robot must actually walk
        # across the current terrain before it gets harder.
        #
        # RECALIBRATED (was 0.8): a run spent its entire 5M-step budget stuck
        # at difficulty 0.1, plateauing around 0.5m stable distance. 0.8m is
        # close to the BEST fixed-gait performance on perfectly flat ground
        # (~1.1m) -- an unreasonable bar to clear before the robot is even
        # allowed to see harder terrain. Lowered so the curriculum actually
        # progresses through difficulty levels, since exposure to a RANGE of
        # terrain is the point, not maximizing performance at each rung.
        "promote_distance": 0.4,
    },
}

# ---------------------------------------------------------------------------
# SIMULATION SETTINGS
# ---------------------------------------------------------------------------

SIM_TIMESTEP = 0.002        # MuJoCo physics step: 2ms (500 Hz physics)
CONTROL_TIMESTEP = 0.02     # how often the policy acts: 20ms = 50 Hz control loop
                            # Each policy step = 10 physics steps (500/50)
                            # 50 Hz matches our target for the real robot.

EPISODE_LENGTH_SECS = 10.0  # maximum episode length in seconds
EPISODE_LENGTH_STEPS = int(EPISODE_LENGTH_SECS / CONTROL_TIMESTEP)  # = 500 steps

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS (Stable-Baselines3 PPO)
# ---------------------------------------------------------------------------

PPO = {
    # How many steps to collect before updating the policy.
    # Larger = more data per update = more stable but slower.
    "n_steps": 2048,

    # How many samples to use per gradient update (subset of n_steps).
    "batch_size": 256,

    # How many parallel copies of the environment to collect experience from.
    # More envs = more diverse data per update = steadier learning.
    "n_envs": 4,

    # Entropy bonus: nudges the policy to keep exploring instead of committing
    # too early. 0.0 works well for locomotion with normalized observations.
    "ent_coef": 0.0,

    # Value-function loss weight and gradient clipping — standard PPO defaults.
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # Hidden layers of the policy/value networks (two layers of 256 neurons).
    "net_arch": [256, 256],

    # How many times to reuse the collected data for updates.
    # PPO's key innovation: safe to reuse data a few times without diverging.
    "n_epochs": 10,

    # Learning rate: how big each gradient step is.
    # 3e-4 = 0.0003 — a standard starting point for PPO on locomotion.
    "learning_rate": 3e-4,

    # Discount factor: how much future rewards matter vs. immediate ones.
    # 0.99 = the policy thinks far ahead (99% weight on future).
    "gamma": 0.99,

    # GAE lambda: controls bias vs. variance in advantage estimation.
    # 0.95 is the standard default — rarely needs tuning.
    "gae_lambda": 0.95,

    # Clipping range: how much the policy can change in one update.
    # This IS PPO's core idea — prevents destructively large updates.
    "clip_range": 0.2,

    # Safety valve on top of clip_range. clip_range limits how far a SINGLE
    # sample's update can move; it does NOT stop the overall policy from
    # drifting a huge distance across a whole batch of n_epochs=10 passes.
    # We measured this happening for real: approx_kl readings of 1.0-2.3 and
    # clip_fraction pinned at ~0.85 (healthy PPO training looks like
    # approx_kl ~0.01-0.05, clip_fraction well under 0.4) -- the policy was
    # thrashing, not converging. target_kl makes SB3 stop taking further
    # epoch passes on a batch the moment the policy has moved this far,
    # directly capping the runaway updates we observed.
    "target_kl": 0.03,

    # Total environment steps to train for.
    # 5 million is a starting point for locomotion — may need more.
    "total_timesteps": 5_000_000,
}

# ---------------------------------------------------------------------------
# HARDWARE SETTINGS (Phase 3 — real robot)
# ---------------------------------------------------------------------------

HARDWARE = {
    # USB serial port the NyBoard appears on.
    # Windows: check Device Manager → Ports (COM & LPT) after plugging in.
    # Typical values: "COM3", "COM5", "COM7"
    "serial_port": "COM3",
    "baud_rate": 115200,

    # Control loop rate for real hardware — must match CONTROL_TIMESTEP above.
    "control_hz": 50,

    # Tilt angle (radians) at which we cut power and stand still.
    # Same as fall_angle in sim so the policy's experience matches hardware.
    "fall_stop_angle": 0.8,

    # How long to wait for an IMU response before giving up (seconds).
    "imu_timeout_s": 0.005,
}

# ---------------------------------------------------------------------------
# SERVO CALIBRATION (Phase 3 — must be verified on real hardware)
# ---------------------------------------------------------------------------

CALIBRATION = {
    # NyBoard slot number for each of our 8 simulated joints.
    # Our sim order: [lb_shoulder, lb_knee, lf_shoulder, lf_knee,
    #                 rb_shoulder, rb_knee, rf_shoulder, rf_knee]
    # Standard Bittle wiring (verify against your physical robot):
    #   Slot 8 = LF shoulder,  9 = RF shoulder, 10 = LB shoulder, 11 = RB shoulder
    #   Slot 12 = LF knee,    13 = RF knee,    14 = LB knee,     15 = RB knee
    "servo_slots": [10, 14, 8, 12, 11, 15, 9, 13],

    # +1 if sim-positive rotation = servo-positive, -1 if reversed.
    # Verify by commanding +0.3 rad to each joint one at a time and watching
    # which direction the limb moves vs. what the sim shows.
    "direction_signs": [1, -1, 1, -1, -1, 1, -1, 1],

    # Per-servo trim in degrees, added after all other conversions.
    # Start at 0. If the robot stands but leans sideways, tweak individual
    # trims here rather than touching the main calibration.
    "trim_deg": [0, 0, 0, 0, 0, 0, 0, 0],

    # Hard limit on how far any joint can deviate from neutral (degrees).
    # Prevents hardware damage from bad policy outputs.
    "max_joint_offset_deg": 45,
}

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
BITTLE_MODEL_PATH = PROJECT_ROOT / "src" / "sim" / "bittle_model" / "scene.xml"
BITTLE_ROUGH_MODEL_PATH = PROJECT_ROOT / "src" / "sim" / "bittle_model" / "scene_rough.xml"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
