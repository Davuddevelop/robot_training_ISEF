"""
07_upright_training.py — Fix the crawling problem. Train the Ant to walk UPRIGHT.

WHY WE NEED THIS
The previous run (ant_v2) produced a crawler: the agent learned to slide flat
along the ground. That earns forward-distance reward while avoiding the risk of
tipping over. It is rational given the reward, but a real quadruped robot must
stand on its legs — a crawling policy is useless on Bittle.

ROOT CAUSE
The reward function in ant_v2 had no concept of "height" or "upright posture".
The agent optimised what we measured, and we measured the wrong thing.

THREE CHANGES (one at a time so you can understand each):

  1. UprightRewardWrapper
     A thin layer around the environment that reads the torso height from the
     info dict and adds a per-step bonus. Standing tall (z ≈ 0.75 m) earns
     ~+0.94 extra per step. Crawling flat (z ≈ 0.25 m) earns nothing. This
     makes upright posture *worth* something.

  2. contact_cost_weight = 0.005  (was 5e-4, now 10× higher)
     Ant-v5 tracks the contact forces on every body part. When the body drags
     on the ground, those forces are large → heavy penalty. When only the feet
     touch the ground (walking), forces are small → low penalty. This makes
     dragging *expensive*.

  3. healthy_z_range = (0.28, 1.0)  (minimum raised from 0.2 → 0.28)
     If the torso drops below 0.28 m the episode ends immediately. True
     pancake-flat crawling is now a losing strategy. NOTE: 0.28 is carefully
     chosen — we tried 0.35 before (ant_improved run) and episodes died before
     the agent could learn anything. 0.28 is strict enough to kill flat crawling
     but permissive enough that early stumbling episodes survive.

Run:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\07_upright_training.py

Quick test (50 k steps):
    (PowerShell)  $env:ANT_TIMESTEPS=50000
    D:\\robot_venv\\Scripts\\python.exe notebooks\\07_upright_training.py
"""

import os
import pathlib

import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_NAME = "ant_v3"
TOTAL_STEPS = int(os.environ.get("ANT_TIMESTEPS", 3_000_000))
N_ENVS = 4  # 4 parallel copies of the environment — 4× the experience per update

SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / RUN_NAME
MONITOR_DIR = SAVE_DIR / "monitor_logs"
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

# --- Environment settings ---
# Changes from ant_v2 are marked with  <-- CHANGED
ENV_KWARGS = {
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.05,
    "contact_cost_weight": 0.005,       # <-- CHANGED: was 5e-4, now 10x heavier
    "healthy_reward": 1.0,
    "healthy_z_range": (0.28, 1.0),     # <-- CHANGED: minimum raised from 0.2 to 0.28
    "max_episode_steps": 1000,
}

# --- PPO hyperparameters (same proven settings as ant_v2) ---
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,      # slightly higher than ant_v2's 0.0 → encourages exploring new gaits
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={"net_arch": [256, 256]},
    device="cpu",
    verbose=1,
)


# ---------------------------------------------------------------------------
# Fix 1: UprightRewardWrapper
# ---------------------------------------------------------------------------

class UprightRewardWrapper(Wrapper):
    """
    Adds a height bonus to the reward every step.

    The Ant-v5 info dict always contains 'z_position' — the height of the
    torso above the ground. We read that value and add:

        bonus = HEIGHT_WEIGHT × max(0, z − MIN_HEIGHT)

    So the agent earns nothing extra when flat, but earns a meaningful bonus
    every step it spends upright. Over a 1000-step episode that adds up to
    hundreds of reward points — a strong signal to stand tall.

    HEIGHT_WEIGHT = 2.0 means:
        z = 0.28 m  (just above terminate threshold) → bonus = 0
        z = 0.50 m  (halfway up)                     → bonus = +0.44
        z = 0.75 m  (healthy upright Ant)             → bonus = +0.94
    """

    HEIGHT_WEIGHT = 2.0
    MIN_HEIGHT = 0.28  # same as healthy_z_range minimum — no bonus for crawling

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        z = info.get("z_position", 0.0)
        height_bonus = self.HEIGHT_WEIGHT * max(0.0, z - self.MIN_HEIGHT)
        return obs, reward + height_bonus, terminated, truncated, info


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------

def make_env(rank: int):
    """Creates one Ant-v5 environment with all three upright fixes applied."""
    def _init():
        env = gym.make("Ant-v5", **ENV_KWARGS)
        env = UprightRewardWrapper(env)   # Fix 1: height bonus
        env = Monitor(env, str(MONITOR_DIR / f"env_{rank}"))
        return env
    return _init


def make_normalized_env():
    """
    Builds the full training environment stack:
      Ant-v5 → UprightRewardWrapper → Monitor → DummyVecEnv → VecNormalize

    VecNormalize keeps a running mean/std and rescales all observations and
    rewards to roughly [-1, 1]. This is essential for PPO — without it,
    gradients from large-scale observations dominate and learning breaks.
    """
    vec_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"  Run: {RUN_NAME}")
    print(f"  Steps: {TOTAL_STEPS:,}")
    print(f"  Envs:  {N_ENVS} parallel")
    print()
    print("  Upright fixes active:")
    print(f"    Height reward:    +{UprightRewardWrapper.HEIGHT_WEIGHT} × max(0, z − {UprightRewardWrapper.MIN_HEIGHT})")
    print(f"    Contact cost:     {ENV_KWARGS['contact_cost_weight']} (10× ant_v2)")
    print(f"    Min height rule:  terminate if z < {ENV_KWARGS['healthy_z_range'][0]} m")
    print("=" * 60)
    print()

    env = make_normalized_env()

    model = PPO("MlpPolicy", env, **PPO_KWARGS)

    print(f"Training for {TOTAL_STEPS:,} steps...\n")
    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    print("\nSaving model and normalisation stats...")
    model.save(str(MODEL_PATH))
    env.save(str(VECNORM_PATH))

    print(f"  Model  → {MODEL_PATH}.zip")
    print(f"  Stats  → {VECNORM_PATH}")
    print()
    print("Next steps:")
    print(f"  $env:ANT_RUN_NAME='{RUN_NAME}'")
    print("  D:\\robot_venv\\Scripts\\python.exe notebooks\\06_evaluate.py")
    print("  D:\\robot_venv\\Scripts\\python.exe notebooks\\03_watch_trained_agent.py")

    env.close()


if __name__ == "__main__":
    main()
