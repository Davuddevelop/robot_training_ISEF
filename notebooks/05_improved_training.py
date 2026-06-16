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

SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / "ant_improved"
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
    # Default is 5e-4. We reduce it slightly to allow more dynamic stepping.
    "contact_cost_weight": 2e-4,

    # --- ALIVE BONUS ---
    # Flat reward per step just for staying upright and functional.
    # Encourages the policy to stay healthy even when it's not moving fast.
    "healthy_reward": 1.0,

    # --- BODY HEIGHT RANGE ---
    # The robot is considered "healthy" (gets the alive bonus) only if its
    # torso stays within this height range (in meters).
    # Default: (0.2, 1.0) — allows belly-crawling or wild jumping.
    # We tighten the lower bound to 0.35 to discourage crawling.
    # RESULT: the robot stays upright like a dog, not like a snake.
    "healthy_z_range": (0.35, 1.0),

    # --- EPISODE LENGTH ---
    # How many steps each episode runs before resetting (if not fallen).
    # Longer episodes = the robot needs to sustain walking, not just get lucky.
    "max_episode_steps": 1000,
}

print("=" * 60)
print("  Improved Training — Dog-Like Locomotion")
print("=" * 60)
print()
print("Reward design vs. first run:")
print("  ctrl_cost_weight:  0.5  →  0.05  (robot uses legs more freely)")
print("  healthy_z_range:   0.2+ →  0.35+ (robot stays upright, not crawling)")
print("  episode length:    500  →  1000  (robot must sustain walking longer)")
print()
print("Expected: smoother, more upright, more forward-directed movement.")
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
    name_prefix="ant_improved",
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

model.save(str(SAVE_DIR / "ant_1M"))
print()
print(f"Saved: {SAVE_DIR / 'ant_1M.zip'}")
print()
print("Run 04_plot_learning_curve.py with MONITOR_DIR updated to see the curve.")
print("Run 03_watch_trained_agent.py with MODEL_PATH updated to see it walk.")
env.close()
