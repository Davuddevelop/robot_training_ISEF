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
    "velocity_command":   (23, 24), # the forward speed (m/s) the robot is being TOLD to
                                    # hit this episode. The policy must SEE the command to
                                    # be able to obey it — a reward for matching a target
                                    # the network cannot observe is unlearnable noise.
                                    # Fully sim-to-real valid: it is a number WE choose,
                                    # not a measurement, so the real robot has it too.
}

OBS_DIM = 24  # total size of the observation vector (23 -> 24 on 2026-08-14)

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

# ---------------------------------------------------------------------------
# VELOCITY COMMAND
# Each episode the robot is TOLD a target forward speed, that number is put in
# the observation, and the reward pays for MATCHING it.
#
# Why this replaced "go as fast as you can" (2026-08-14): with a per-step
# alive_bonus, standing still for a whole 500-step episode scored 500, while
# walking and falling at step 50 scored 55 -- standing was 9x better, so the
# policy correctly learned to barely move. Under a tracking reward, standing
# still when told to move earns almost nothing, so opting out stops paying.
# This is what every mainstream legged-RL stack does (legged_gym, Isaac Lab,
# MuJoCo Playground); none of them reward raw speed, and none use an alive bonus.
# ---------------------------------------------------------------------------
COMMAND = {
    # Forward speed range (m/s) sampled once per episode.
    #
    # These are calibrated to what this robot ACTUALLY achieves, measured from
    # the scripted-gait benchmark -- 0.128 m/s on flat, 0.068 m/s at difficulty
    # 0.5, 0.026 m/s at difficulty 1.0. Commanding a speed the robot cannot
    # reach is the number-one way this reward fails: the error is large no
    # matter what it does, exp(-large) ~ 0 everywhere, the gradient vanishes and
    # the policy gives up. So the range spans "slow but real" to "as good as the
    # best hand-tuned gait on flat ground" -- ambitious but not fantasy.
    "vx_range": (0.08, 0.15),

    # Width of the tracking reward's tolerance, in (m/s)^2:
    #     reward = exp(-(v_cmd - v_actual)^2 / sigma)
    #
    # SIGMA MUST BE SCALED TO THE ROBOT'S SPEED. legged_gym uses 0.25, but for
    # robots commanded up to 1.0 m/s. The error is SQUARED, so sigma scales with
    # velocity squared: our ~0.13x speeds imply 0.25 * 0.13^2 ~ 0.004.
    #
    # Why this matters more than it sounds -- what STANDING STILL scores:
    #     sigma=0.25   -> 0.88 to 0.98   (standing is still nearly optimal!)
    #     sigma=0.05   -> 0.52 to 0.88
    #     sigma=0.004  -> 0.004 to 0.20  <-- what we use
    # Copying 0.25 unchanged would have reproduced the exact bug we are fixing,
    # just with extra steps.
    "tracking_sigma": 0.004,
}

REWARD = {
    # Positive: reward forward movement (m/s in the robot's forward direction).
    # This MUST dominate, or the policy just stands still to farm the alive bonus.
    # We learned this the hard way: with coeff=1.0 and alive=0.5, a 60k-step run
    # scored 250 reward but walked 0.03 m — it stood still. Raising this makes
    # walking clearly worth more than standing.
    #
    # RETIRED 2026-08-14 -- kept at 0.0 so old configs still load, but the
    # reward no longer pays for raw speed. Rewarding "faster is always better"
    # cannot express "go at THIS speed", so it could never punish going too
    # slow; combined with alive_bonus it made standing still optimal. Replaced
    # by tracking_lin_vel below. Set >0 only to reproduce the old behaviour.
    "forward_velocity_coeff": 0.0,

    # Weight on the velocity-tracking reward: exp(-(v_cmd - v_x)^2 / sigma).
    # This is now the ONLY positive term. Bounded to [0, 1], so no other term
    # can drown it out, and it is worth ~0 when standing still under a nonzero
    # command -- which is exactly the property the old reward lacked.
    # legged_gym/Isaac Lab/MuJoCo Playground all use 1.0 here.
    "tracking_lin_vel": 1.0,

    # Positive: bonus for each step it stays upright (does not fall). With
    # forward_velocity_coeff=5.0 a moving robot always out-scores a standing one,
    # so this no longer causes standing-still — instead it makes SURVIVING pay,
    # which stops the "lunge forward then fall over" behaviour (a 300k run moved
    # 0.36 m but fell after 70 of 500 steps when this was only 0.1).
    #
    # RETIRED 2026-08-14 -- set to 0.0. THIS WAS THE BUG.
    #
    # A per-step bonus for merely existing means the best way to score is to
    # exist for as long as possible. Measured from this exact config: standing
    # still for all 500 steps = 500 reward; walking at 0.2 m/s and falling at
    # step 50 = 55. Standing was 9x better. The -30 fall penalty was never the
    # real deterrent -- the 450 points of FORFEITED alive bonus was.
    #
    # legged_gym (the ETH stack behind ANYmal) has no alive bonus at all.
    # Staying upright is rewarded implicitly: falling ends the episode, and the
    # only positive reward is one you can only earn by moving. Reda et al.
    # (MIG 2020) found no good value exists -- too small gives falling-forward,
    # too large gives standing still -- which is why the term is removed, not
    # retuned.
    "alive_bonus": 0.0,

    # Negative: penalty for drifting sideways (lateral = X axis, not forward Y).
    # Without this the policy sometimes crab-walks to one side rather than going
    # straight forward, which looks bad and performs worse on real flat ground.
    # Coefficient scales the SQUARED lateral speed, so small drifts barely matter
    # but large crab-walk is strongly penalised.
    "lateral_velocity_penalty": -0.5,

    # Negative: penalty for tilting away from upright, applied CONTINUOUSLY.
    #
    # Replaces a step function ("if tilt > 0.5 rad: -1.0") which had ZERO
    # gradient everywhere except at the cliff edge -- it told the policy that
    # leaning 27 degrees was perfectly fine and 29 degrees was a disaster, with
    # no signal in between about which direction to correct. A cliff teaches
    # avoidance, not control, which is part of why the policy became timid.
    #
    # The new term uses the horizontal components of projected gravity:
    #   g[0]^2 + g[1]^2 = sin^2(tilt)  ->  0.0 upright, 0.25 at 30deg, 0.5 at 45deg
    # Smooth, bounded, and computed only from what the real IMU already gives
    # us, so it stays sim-to-real valid. This is legged_gym's "orientation" term.
    "tilt_penalty_coeff": -1.0,

    # Kept only so `_is_fallen` and old configs still resolve; the continuous
    # term above is what actually shapes behaviour now.
    "tilt_threshold": 0.5,       # ~28 degrees
    "tilt_penalty": 0.0,         # RETIRED -- superseded by tilt_penalty_coeff

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
    # REDUCED -30.0 -> -1.0 on 2026-08-14. The reasoning above was sound while
    # the reward paid for raw speed, but -30 turned out to be ~30x larger than
    # any value used in published legged-RL work (legged_gym uses -0.0;
    # MuJoCo Playground -1.0; Isaac Lab has no such term). Combined with the
    # alive bonus it produced extreme risk aversion -- the policy stopped
    # attempting terrain at all. With a velocity-tracking reward, falling is
    # already punished implicitly (the episode ends, so all remaining tracking
    # reward is forfeited); the explicit penalty only needs to break ties.
    "fall_penalty": -1.0,
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

        # Fall-rate ceiling for an evaluation to count as "stable" at all.
        # This only became a real measurement when n_eval rose from 3 to 10:
        # with 3 episodes, fall_rate could only be 0, 0.33, 0.67 or 1.0, so a
        # 0.5 threshold was closer to a coin flip than a criterion.
        "promote_max_fall_rate": 0.3,

        # --- Demotion. Promotion used to be one-way, so a single lucky
        # evaluation could push the robot onto ground it could not actually
        # hold, with no way back -- runs then burned millions of steps
        # thrashing there. Dropping a rung lets the policy consolidate and
        # re-climb. Requires TWO consecutive bad evaluations so ordinary
        # evaluation noise cannot trigger it.
        "demote_distance": 0.2,
        "demote_fall_rate": 0.6,
        "demote_after_bad_evals": 2,

        # Minimum steps to spend at a difficulty before promotion is allowed.
        # Anti-thrash guard: without it, promote/demote can oscillate every
        # evaluation and the policy never settles anywhere long enough to learn.
        "min_steps_at_difficulty": 200_000,
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
    # too early.
    #
    # RAISED from 0.0 after diagnosing a training plateau. With ent_coef=0.0
    # nothing resists the policy's action stddev shrinking, and in a 16M-step
    # run it collapsed to std~0.137 (verified: SB3 reported entropy_loss +4.58,
    # and an 8-dim Gaussian at std=0.137 has entropy exactly -4.55). That
    # collapse is self-reinforcing and it traps training:
    #   - KL divergence scales as (delta/std)^2, so at std=0.137 the policy is
    #     ~53x more KL-sensitive than at std=1.0. Measured consequence: it takes
    #     only 0.44 deg of joint-angle change to hit KL=0.05, versus 3.2 deg at
    #     std=1.0. target_kl then aborted the update after 1 of 10 epochs on
    #     EVERY iteration, so the policy could barely move at all.
    #   - A near-deterministic policy cannot explore out of a bad gait. An
    #     experiment sweeping 80 open-loop gaits found only ~5% are viable at
    #     terrain difficulty 0.4 -- a narrow basin that needs real exploration.
    # Measured directly: resuming a collapsed policy with ent_coef=0.0 let std
    # drift further DOWN (-0.0029 over 8k steps), while ent_coef=0.01 pushed it
    # back UP (+0.0109). 0.01 is a standard locomotion value.
    "ent_coef": 0.01,

    # Recovery knob for an ALREADY-collapsed policy. ent_coef stops a healthy
    # policy from collapsing, but it only nudges a collapsed one back very
    # slowly. Set BITTLE_RESET_STD=0.5 (or any std) when resuming to reset the
    # policy's action stddev in place, which frees it to explore again
    # immediately. Leave at None for normal runs; this is a deliberate
    # intervention, not something that should happen silently.
    #
    # MUST be applied as an in-place .data edit -- verified that REPLACING the
    # log_std Parameter object leaves SB3's optimizer pointing at the old
    # tensor, so the reset silently fails to train.
    "reset_std_default": 0.5,

    # How the learning rate behaves over a run.
    #   "constant" -- the same rate from start to finish (what we have always used)
    #   "linear"   -- decays smoothly from `learning_rate` down to 0 at the end
    # Linear decay is standard for PPO locomotion: big steps early while the policy
    # is bad, small careful steps late while it is fine-tuning. Kept as a switch so
    # its effect can be A/B measured on its own rather than bundled with other edits.
    # NOTE: the schedule is driven by progress through `total_timesteps`, so changing
    # the step budget mid-run makes the rate jump.
    "lr_schedule": "constant",

    # Hard floor and ceiling on the policy's action stddev, enforced every
    # eval cycle during training. Nothing in ordinary PPO stops std from
    # drifting to an unhealthy extreme in EITHER direction over a long run:
    # we measured it collapse to ~0.137 in one run (too rigid to explore) and
    # explode to 4.57 in another (loss dominated by the entropy bonus, action
    # noise close to random) -- see EXPERIMENT_LOG.md, 2026-08-13. This clamp
    # cannot fix WHY std drifts, but it guarantees it can never leave a sane
    # operating range regardless of how the entropy math behaves at any given
    # reward scale.
    "std_clamp_min": 0.1,
    "std_clamp_max": 1.5,

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
