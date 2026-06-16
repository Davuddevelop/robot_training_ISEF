"""
05_improved_training.py — Better training for more natural, dog-like movement.

The first run (02_first_training.py) used default Ant settings — fine for
a demo, but not shaped for natural locomotion. This script tunes the reward
function and environment parameters to produce movement that looks much more
like a real walking dog: smooth, forward-moving, upright, energy-efficient.

These reward terms are EXACTLY what we will use for Bittle. So understanding
this script means understanding your entire research project.

Run with:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\05_improved_training.py

Training time: ~45-90 minutes for 1,000,000 steps on CPU.
Start it and let it run while you study or sleep.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import pathlib

# Fresh run folder. The previous run ("ant_improved") had a flat, negative
# learning curve because the healthy-height rule was too strict. We keep that
# run's data for comparison and write THIS corrected run to a new folder, so
# the two learning curves stay separate and clean.
SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / "ant_v2"
MONITOR_DIR = SAVE_DIR / "monitor_logs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# THE REWARD DESIGN — this is the most important part of this file.
#
# Ant-v5 lets us pass parameters that shape the reward function directly.
# Each one answers: "what do we want the robot to care about?"
# This is where your engineering judgment goes into code.
# ---------------------------------------------------------------------------

ENV_KWARGS = {
    # --- FORWARD MOVEMENT ---
    # How much the robot is rewarded for moving forward (in the x-direction).
    # Higher = the policy will push for more speed, but may sacrifice stability.
    # Default is 1.0. We keep it at 1.0 — speed is the primary goal.
    "forward_reward_weight": 1.0,

    # --- ENERGY EFFICIENCY (action cost) ---
    # Penalty for large joint torques (a proxy for energy use).
    # Default is 0.5 — very conservative, which makes the ant hesitant to move.
    # We reduce it to 0.05 so the robot isn't afraid to use its legs.
    # RESULT: more active, decisive leg movements. More dog-like.
    "ctrl_cost_weight": 0.05,

    # --- CONTACT FORCE PENALTY ---
    # Penalty for large forces at contact points (feet hitting ground too hard).
    # Back to the tested default — one less thing changed from known-good.
    "contact_cost_weight": 5e-4,

    # --- ALIVE BONUS ---
    # Flat reward per step just for staying upright and functional.
    # This is why a healthy walking episode scores strongly POSITIVE: it
    # collects this bonus on every one of up to 1000 steps. If episodes die
    # early, the agent barely collects it — which is what went wrong last time.
    "healthy_reward": 1.0,

    # --- BODY HEIGHT RANGE  (THE KEY FIX) ---
    # The episode ENDS ("unhealthy") if the torso leaves this height range.
    # Last run used a lower bound of 0.35m. Early in training the Ant cannot
    # hold itself that high, so episodes were killed within a few steps —
    # before the agent could learn anything. The reward stayed flat and
    # negative because almost no alive-bonus was ever collected.
    # FIX: back to the tested default (0.2, 1.0). Let it learn first; we can
    # tighten toward "more upright" LATER, once it can already walk.
    # LESSON (this is your science too): change ONE thing at a time, and start
    # from settings that are known to work.
    "healthy_z_range": (0.2, 1.0),

    # --- EPISODE LENGTH ---
    # How many steps each episode runs before resetting (if not fallen).
    # Longer episodes = the robot needs to sustain walking, not just get lucky.
    "max_episode_steps": 1000,
}

print("=" * 60)
print("  Training v2 — Corrected Reward Design")
print("=" * 60)
print()
print("What went wrong last run (ant_improved):")
print("  healthy_z_range lower bound was 0.35m -> episodes died instantly")
print("  -> flat, negative learning curve (agent never learned to walk)")
print()
print("This run (ant_v2):")
print("  healthy_z_range:   0.35 -> 0.2   (let it survive long enough to learn)")
print("  ctrl_cost_weight:  0.5  -> 0.05  (robot uses legs more freely)")
print("  network:           64x64 -> 256x256 (more capacity)")
print()
print("Expected now: a curve that CLIMBS into positive reward and flattens.")
print()

n_envs = 4

env = make_vec_env(
    "Ant-v5",
    n_envs=n_envs,
    monitor_dir=str(MONITOR_DIR),
    env_kwargs=ENV_KWARGS,
)

print(f"Environment: {n_envs} parallel Ant simulations")
print(f"Observation: {env.observation_space.shape[0]} numbers")
print(f"Actions:     {env.action_space.shape[0]} joint torques")
print()

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    # Network architecture: two hidden layers of 256 neurons each.
    # Bigger than default (64) because locomotion needs more capacity.
    policy_kwargs={"net_arch": [256, 256]},
    device="cpu",
    verbose=1,
)

print(f"Policy network: {sum(p.numel() for p in model.policy.parameters()):,} parameters")
print(f"  (bigger network than last time — more capacity for complex movement)")
print()

checkpoint = CheckpointCallback(
    save_freq=100_000 // n_envs,
    save_path=str(SAVE_DIR / "checkpoints"),
    name_prefix="ant_v2",
    verbose=1,
)

TOTAL_STEPS = 1_000_000
print(f"Training for {TOTAL_STEPS:,} steps (~45-90 min on CPU).")
print("Checkpoints saved every 100k steps — you can stop and resume anytime.")
print()

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=checkpoint,
)

model.save(str(SAVE_DIR / "ant_v2_1M"))
print()
print(f"Saved: {SAVE_DIR / 'ant_v2_1M.zip'}")
print()
print("Next:")
print("  1) python notebooks/04_plot_learning_curve.py   (curve should climb now)")
print("  2) python notebooks/03_watch_trained_agent.py    (watch it walk)")
print("Both scripts already point at the ant_v2 run.")
env.close()
