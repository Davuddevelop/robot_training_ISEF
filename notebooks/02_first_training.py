"""
02_first_training.py — Your first real training run.

This trains a 4-legged robot (Ant-v5) to walk using the exact same
algorithm (PPO) and library (Stable-Baselines3) we will use for Bittle.

Ant is a built-in MuJoCo environment — no custom code needed.
It has 4 legs, like Bittle. Watch it go from random flailing to walking.

Run with:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\02_first_training.py

What to watch for in the output:
    ep_rew_mean  — average reward per episode. Should increase over time.
    ep_len_mean  — average episode length. Should increase (survives longer).

Time: ~5-10 minutes for 300,000 steps on your CPU.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import pathlib

# Where to save the trained model
SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / "ant_first_run"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  First Training Run — Ant Quadruped")
print("=" * 60)
print()
print("The Ant environment:")
print("  - 4 legs, 8 joints (same as Bittle)")
print("  - Goal: walk forward as fast as possible")
print("  - Algorithm: PPO (same as we will use for Bittle)")
print()
print("Watch 'ep_rew_mean' — it should increase over time.")
print("If it stays near 0 or goes negative, the agent is struggling.")
print()

# --- CREATE THE ENVIRONMENT ---
# make_vec_env creates N parallel copies of the environment.
# More copies = more data per second = faster training.
# We use 4 here — your CPU can handle 4 Ant simulations at once.
n_envs = 4
env = make_vec_env("Ant-v5", n_envs=n_envs)

print(f"Environment created: {n_envs} parallel Ant simulations")
print(f"Observation size: {env.observation_space.shape[0]} numbers")
print(f"Action size:      {env.action_space.shape[0]} joint angles")
print()

# --- CREATE THE PPO AGENT ---
# "MlpPolicy" = a simple 2-layer neural network.
# The network reads observations and outputs joint angles.
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,     # how big each learning step is
    n_steps=2048,           # steps collected before each update
    batch_size=64,          # how many samples per gradient update
    n_epochs=10,            # how many times to reuse each batch
    gamma=0.99,             # how much to value future reward vs. now
    gae_lambda=0.95,        # smoothing parameter for advantage estimation
    clip_range=0.2,         # PPO's core: limits how much policy can change at once
    verbose=1,
    tensorboard_log=str(SAVE_DIR / "logs"),
)

print("PPO agent created.")
print(f"Policy network: {sum(p.numel() for p in model.policy.parameters())} parameters")
print()

# --- CHECKPOINT CALLBACK ---
# Saves the model every 50,000 steps so you don't lose progress.
checkpoint = CheckpointCallback(
    save_freq=50_000 // n_envs,  # every 50k total steps
    save_path=str(SAVE_DIR / "checkpoints"),
    name_prefix="ant_ppo",
)

# --- TRAIN ---
TOTAL_STEPS = 300_000   # short run — just to see it working
                        # For Bittle, we will use 5,000,000+ steps

print(f"Training for {TOTAL_STEPS:,} steps...")
print("(This takes ~5-10 minutes. Watch the numbers change.)")
print()

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=checkpoint,
    progress_bar=True,
)

# --- SAVE ---
model.save(str(SAVE_DIR / "ant_300k"))
print()
print(f"Model saved to: {SAVE_DIR / 'ant_300k.zip'}")
print()
print("=" * 60)
print("  Training complete.")
print()
print("  What just happened:")
print("  - The PPO agent collected 300,000 steps of experience")
print("  - After every 2,048 steps, it updated the network weights")
print("  - The network slowly learned which actions lead to more reward")
print()
print("  Next: run 03_watch_trained_agent.py to see it walk.")
print("=" * 60)

env.close()
