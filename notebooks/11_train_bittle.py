"""
11_train_bittle.py — Teach the Bittle to walk in simulation.

Supports TWO experimental conditions by environment variable:

  Condition 1 — no domain randomization (clean baseline, easiest to learn):
      D:\\robot_venv\\Scripts\\python.exe notebooks\\11_train_bittle.py
      Saves to: models/bittle_v1/

  Condition 2 — full domain randomization (robustness training):
      (PowerShell)  $env:BITTLE_DOMAIN_RAND="1"
      D:\\robot_venv\\Scripts\\python.exe notebooks\\11_train_bittle.py
      Saves to: models/bittle_v2/

Quick test (any condition, 50k steps ≈ 2 min):
      (PowerShell)  $env:BITTLE_TIMESTEPS=50000
      D:\\robot_venv\\Scripts\\python.exe notebooks\\11_train_bittle.py

Every setting comes from src/sim/config.py — that is the single source of truth.

WHAT THIS SAVES:
  bittle_vX_model.zip       the final policy weights
  vecnormalize.pkl          observation scaling statistics (must travel with model)
  best_model.zip            the best policy seen during training (by distance)
  best_vecnormalize.pkl     scaling stats that go with best_model
  checkpoints/              a checkpoint every 100k steps (crash recovery)
  monitor_logs/             per-episode reward log (for the learning curve)
"""

import os
import pathlib
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.sim.bittle_env import BittleEnv
from src.sim.config import PPO as PPO_CFG

# ---------------------------------------------------------------------------
# CONFIGURATION  (override with environment variables)
# ---------------------------------------------------------------------------

DOMAIN_RAND = os.environ.get("BITTLE_DOMAIN_RAND", "0") == "1"
TOTAL_STEPS = int(os.environ.get("BITTLE_TIMESTEPS", PPO_CFG["total_timesteps"]))

# Auto-name so both conditions live in separate folders.
# Override: $env:BITTLE_RUN_NAME="my_run"
_default_name = "bittle_v2" if DOMAIN_RAND else "bittle_v1"
RUN_NAME      = os.environ.get("BITTLE_RUN_NAME", _default_name)

ROOT       = pathlib.Path(__file__).parent.parent
SAVE_DIR   = ROOT / "models" / RUN_NAME
MONITOR_DIR = SAVE_DIR / "monitor_logs"
MODEL_PATH  = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# ENVIRONMENT FACTORY
# ---------------------------------------------------------------------------

def make_env(rank, domain_rand=DOMAIN_RAND):
    """One Bittle environment wrapped in Monitor for logging."""
    def _init():
        env = BittleEnv(domain_rand=domain_rand)
        env = Monitor(env, str(MONITOR_DIR / f"env_{rank}"))
        return env
    return _init


# ---------------------------------------------------------------------------
# BEST-MODEL CALLBACK
# ---------------------------------------------------------------------------

class BestDistanceCallback(BaseCallback):
    """
    Every eval_freq environment steps, runs a few episodes on a clean
    (no domain randomization) environment and measures how far the robot
    walked.  Saves the model + VecNormalize stats whenever a new best is found.

    Why distance and not reward?
    Reward can be inflated by the alive bonus (standing still scores well).
    Distance in metres is the honest metric — it proves the robot actually walks.
    """

    def __init__(self, save_dir, eval_freq=100_000, n_eval=3, verbose=1):
        super().__init__(verbose)
        self._save_dir  = pathlib.Path(save_dir)
        self._eval_freq = eval_freq
        self._n_eval    = n_eval
        self._best_dist = -np.inf
        self._last_eval = 0

        # Raw eval env (no VecNormalize — we borrow normalization from the
        # training env so the stats stay in sync).
        self._raw_env = DummyVecEnv([lambda: BittleEnv(domain_rand=False)])

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self._eval_freq:
            return True
        self._last_eval = self.num_timesteps

        distances = []
        for _ in range(self._n_eval):
            raw_obs   = self._raw_env.reset()
            start_y   = None
            done      = [False]
            while not done[0]:
                # Normalize using the TRAINING env's running statistics.
                norm_obs = self.training_env.normalize_obs(raw_obs)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                raw_obs, _, done, infos = self._raw_env.step(action)
                if start_y is None:
                    start_y = infos[0].get("forward_position", 0.0)
                last_y = infos[0].get("forward_position", 0.0)
            distances.append(last_y - (start_y or 0.0))

        mean_dist = float(np.mean(distances))
        marker = ""
        if mean_dist > self._best_dist:
            self._best_dist = mean_dist
            self.model.save(str(self._save_dir / "best_model"))
            self.training_env.save(str(self._save_dir / "best_vecnormalize.pkl"))
            marker = "  ← NEW BEST, saved"

        if self.verbose:
            print(f"\n  [Best-model eval @ {self.num_timesteps:,} steps]  "
                  f"distance: {mean_dist:.3f} m  "
                  f"(best: {self._best_dist:.3f} m){marker}\n")
        return True

    def _on_training_end(self):
        self._raw_env.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"  Run: {RUN_NAME}")
    print(f"  Domain randomization: {DOMAIN_RAND}")
    print(f"  Steps: {TOTAL_STEPS:,}   parallel envs: {PPO_CFG['n_envs']}")
    print(f"  Saving to: {SAVE_DIR}")
    print("=" * 60)

    vec_env = DummyVecEnv([make_env(i) for i in range(PPO_CFG["n_envs"])])
    env     = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        learning_rate   = PPO_CFG["learning_rate"],
        n_steps         = PPO_CFG["n_steps"],
        batch_size      = PPO_CFG["batch_size"],
        n_epochs        = PPO_CFG["n_epochs"],
        gamma           = PPO_CFG["gamma"],
        gae_lambda      = PPO_CFG["gae_lambda"],
        clip_range      = PPO_CFG["clip_range"],
        ent_coef        = PPO_CFG["ent_coef"],
        vf_coef         = PPO_CFG["vf_coef"],
        max_grad_norm   = PPO_CFG["max_grad_norm"],
        policy_kwargs   = {"net_arch": PPO_CFG["net_arch"]},
        device          = "cpu",
        verbose         = 1,
    )

    callbacks = CallbackList([
        # Crash recovery: saves every 100k steps (model + vecnormalize).
        CheckpointCallback(
            save_freq        = 100_000,
            save_path        = str(SAVE_DIR / "checkpoints"),
            name_prefix      = f"{RUN_NAME}_ckpt",
            save_vecnormalize = True,
        ),
        # Best model: evaluates on clean env, saves best by distance.
        BestDistanceCallback(
            save_dir  = SAVE_DIR,
            eval_freq = 100_000,
            n_eval    = 3,
            verbose   = 1,
        ),
    ])

    print(f"\nTraining for {TOTAL_STEPS:,} steps …\n")
    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=True,
                callback=callbacks)

    print("\nSaving final model …")
    model.save(str(MODEL_PATH))
    env.save(str(VECNORM_PATH))
    env.close()

    print(f"\n  Final model → {MODEL_PATH}.zip")
    print(f"  Best model  → {SAVE_DIR / 'best_model'}.zip")
    print(f"\nNext steps:")
    print(f"  Evaluate:  D:\\robot_venv\\Scripts\\python.exe notebooks\\13_evaluate_bittle.py")
    print(f"  Watch:     D:\\robot_venv\\Scripts\\python.exe notebooks\\12_watch_bittle.py")
    print(f"  Compare:   D:\\robot_venv\\Scripts\\python.exe notebooks\\14_compare_runs.py")


if __name__ == "__main__":
    main()
