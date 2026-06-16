"""
08_quadruped_training.py — Train a proper dog-shaped quadruped.

WHY WE SWITCH AWAY FROM ANT
The MuJoCo Ant has four legs sticking out SIDEWAYS, like a spider or crab.
Bittle has four legs hanging DOWNWARD from under its body, like a dog.

These are completely different body plans. A policy trained on Ant will learn
gait patterns that physically cannot transfer to Bittle. We need a model with
the same body plan: body suspended above the ground, legs pointing down.

THE MODEL: dm_control quadruped (by Google DeepMind)
DeepMind published this as part of their locomotion research suite. It is
a proper 4-legged robot body with:
  • Body held HIGH off the ground (like Bittle)
  • 4 legs pointing DOWN (like Bittle)
  • 3 joints per leg: hip, thigh, shin  → 12 joints total (Bittle has 8+1)
  • Built-in IMU-like sensors: orientation, angular velocity
  • A "walk" task that rewards forward speed × upright posture together

We access it through "shimmy" — a small Python package that wraps dm_control
environments to work with the Gymnasium API that SB3 already understands.

INSTALL FIRST (run once in PowerShell):
    D:\\robot_venv\\Scripts\\pip.exe install dm_control shimmy[dm_control]

Run training:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\08_quadruped_training.py

Quick test (50 k steps):
    (PowerShell)  $env:ANT_TIMESTEPS=50000
    D:\\robot_venv\\Scripts\\python.exe notebooks\\08_quadruped_training.py

Watch the result:
    (PowerShell)  $env:ANT_RUN_NAME="quad_v1"
    D:\\robot_venv\\Scripts\\python.exe notebooks\\03_watch_trained_agent.py

Evaluate with numbers:
    (PowerShell)  $env:ANT_RUN_NAME="quad_v1"
    D:\\robot_venv\\Scripts\\python.exe notebooks\\06_evaluate.py
"""

import os
import pathlib

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_NAME = "quad_v1"
TOTAL_STEPS = int(os.environ.get("ANT_TIMESTEPS", 3_000_000))
N_ENVS = 4

SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / RUN_NAME
MONITOR_DIR = SAVE_DIR / "monitor_logs"
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

# The dm_control environment ID. shimmy translates it for us.
# "walk" = target speed ~1 m/s, calm trot. Alternatively: "quadruped-run-v0".
ENV_ID = "dm_control/quadruped-walk-v0"

# PPO hyperparameters — same proven settings as ant_v3.
# No changes needed: PPO works the same way regardless of environment shape.
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={"net_arch": [256, 256]},
    device="cpu",
    verbose=1,
)


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------

def make_env(rank: int):
    """
    Creates one dm_control quadruped environment.

    No ENV_KWARGS needed here — unlike Ant-v5, the dm_control task already
    sets up the reward correctly. We do NOT need our UprightRewardWrapper
    because the "walk" task reward is:

        reward = (how close to target speed?) × (how upright is the body?)

    Both components are multiplied together. If the robot falls over, the
    upright part goes to 0 → reward goes to 0 even if it is moving fast.
    If it stands still, the speed part goes to 0. To earn reward, it MUST
    walk AND stay upright simultaneously. This is exactly what we want.
    """
    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env, str(MONITOR_DIR / f"env_{rank}"))
        return env
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    # Check that shimmy is installed before trying anything else.
    try:
        import shimmy  # noqa: F401
        import dm_control  # noqa: F401
    except ImportError:
        print("ERROR: dm_control or shimmy is not installed.")
        print()
        print("Run this first:")
        print("  D:\\robot_venv\\Scripts\\pip.exe install dm_control shimmy[dm_control]")
        return

    print("=" * 62)
    print(f"  Model:   dm_control quadruped  (dog-shaped body plan)")
    print(f"  Run:     {RUN_NAME}")
    print(f"  Steps:   {TOTAL_STEPS:,}")
    print(f"  Envs:    {N_ENVS} parallel environments")
    print()
    print("  Reward structure (already built into dm_control task):")
    print("    reward = forward_speed_score × upright_score")
    print("    Both must be high to earn reward — no crawling shortcut.")
    print("=" * 62)
    print()

    # Build the vectorised + normalised environment stack.
    vec_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", env, **PPO_KWARGS)

    print(f"Training for {TOTAL_STEPS:,} steps...\n")
    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    print("\nSaving model and normalisation stats...")
    model.save(str(MODEL_PATH))
    env.save(str(VECNORM_PATH))
    env.close()

    print(f"  Model  → {MODEL_PATH}.zip")
    print(f"  Stats  → {VECNORM_PATH}")
    print()
    print("Next — evaluate then watch:")
    print(f"  $env:ANT_RUN_NAME='{RUN_NAME}'")
    print("  D:\\robot_venv\\Scripts\\python.exe notebooks\\06_evaluate.py")
    print("  D:\\robot_venv\\Scripts\\python.exe notebooks\\03_watch_trained_agent.py")


if __name__ == "__main__":
    main()
