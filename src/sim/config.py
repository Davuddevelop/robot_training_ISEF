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

    # Positive: small flat bonus for staying alive (not falling). Kept SMALL on
    # purpose — just enough to discourage suicidal falling, not enough to make
    # standing still a winning strategy.
    "alive_bonus": 0.1,

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

    # Total environment steps to train for.
    # 5 million is a starting point for locomotion — may need more.
    "total_timesteps": 5_000_000,
}

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
BITTLE_MODEL_PATH = PROJECT_ROOT / "src" / "sim" / "bittle_model" / "scene.xml"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
