"""
11_train_bittle.py — Teach the REAL Bittle to walk in simulation. (Phase 2)

This is the same proven recipe we validated on the Ant (PPO + VecNormalize),
now pointed at OUR Bittle environment (src/sim/bittle_env.py). Every number
comes from src/sim/config.py — the single source of truth.

This first run uses domain_rand=False. That is deliberate: it is BOTH the
easiest setting to learn a first gait AND experimental condition (1) of your
study, "no domain randomization". Later runs flip domain_rand=True for the
robustness conditions and the ablation.

Run:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\11_train_bittle.py

Quick test (50k steps, ~2 min, just to confirm it runs):
    (PowerShell)  $env:BITTLE_TIMESTEPS=50000
    D:\\robot_venv\\Scripts\\python.exe notebooks\\11_train_bittle.py
"""

import os
import pathlib
import sys

# Make "src" importable when running this file directly.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.sim.bittle_env import BittleEnv
from src.sim.config import PPO as PPO_CFG

RUN_NAME = "bittle_v1"
DOMAIN_RAND = False   # condition (1): no randomization — clean baseline / first gait
TOTAL_STEPS = int(os.environ.get("BITTLE_TIMESTEPS", PPO_CFG["total_timesteps"]))

ROOT = pathlib.Path(__file__).parent.parent
SAVE_DIR = ROOT / "models" / RUN_NAME
MONITOR_DIR = SAVE_DIR / "monitor_logs"
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_DIR.mkdir(parents=True, exist_ok=True)


def make_env(rank):
    """One Bittle environment, wrapped in Monitor so we get a learning curve."""
    def _init():
        env = BittleEnv(domain_rand=DOMAIN_RAND)
        env = Monitor(env, str(MONITOR_DIR / f"env_{rank}"))
        return env
    return _init


def main():
    print("=" * 60)
    print(f"  Run: {RUN_NAME}   domain_rand={DOMAIN_RAND}")
    print(f"  Steps: {TOTAL_STEPS:,}   parallel envs: {PPO_CFG['n_envs']}")
    print("=" * 60)

    # Same env stack as Ant: parallel envs -> VecNormalize (scales obs + reward).
    vec_env = DummyVecEnv([make_env(i) for i in range(PPO_CFG["n_envs"])])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=PPO_CFG["learning_rate"],
        n_steps=PPO_CFG["n_steps"],
        batch_size=PPO_CFG["batch_size"],
        n_epochs=PPO_CFG["n_epochs"],
        gamma=PPO_CFG["gamma"],
        gae_lambda=PPO_CFG["gae_lambda"],
        clip_range=PPO_CFG["clip_range"],
        ent_coef=PPO_CFG["ent_coef"],
        vf_coef=PPO_CFG["vf_coef"],
        max_grad_norm=PPO_CFG["max_grad_norm"],
        policy_kwargs={"net_arch": PPO_CFG["net_arch"]},
        device="cpu",
        verbose=1,
    )

    print(f"\nTraining for {TOTAL_STEPS:,} steps...\n")
    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True)

    print("\nSaving...")
    model.save(str(MODEL_PATH))
    env.save(str(VECNORM_PATH))
    env.close()
    print(f"  Model  -> {MODEL_PATH}.zip")
    print(f"  Stats  -> {VECNORM_PATH}")
    print("\nNext:")
    print("  D:\\robot_venv\\Scripts\\python.exe notebooks\\13_evaluate_bittle.py")
    print("  D:\\robot_venv\\Scripts\\python.exe notebooks\\12_watch_bittle.py")


if __name__ == "__main__":
    main()
