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

AUTO-RESUME (IMPORTANT — read this if you've been closing terminals):
  Closing the terminal kills this script mid-run. This script DETECTS that on
  the next run and picks up exactly where it left off — same model weights,
  same step count — instead of starting over from a brand-new random policy.
  You lose at most the ~100k steps since the last checkpoint, never the whole
  run. Just re-run the exact same command again.

WHAT THIS SAVES:
  bittle_vX_model.zip       the final policy weights
  vecnormalize.pkl          observation scaling statistics (must travel with model)
  best_model.zip            the best policy seen during training (by distance)
  best_vecnormalize.pkl     scaling stats that go with best_model
  checkpoints/              a checkpoint every 100k steps (crash recovery + resume)
  monitor_logs/             per-episode reward log (for the learning curve)
"""

import os
import pathlib
import re
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
from src.sim.config import EPISODE_LENGTH_STEPS

# ---------------------------------------------------------------------------
# CONFIGURATION  (override with environment variables)
# ---------------------------------------------------------------------------

DOMAIN_RAND = os.environ.get("BITTLE_DOMAIN_RAND", "0") == "1"
TOTAL_STEPS = int(os.environ.get("BITTLE_TIMESTEPS", PPO_CFG["total_timesteps"]))
CHECKPOINT_FREQ = int(os.environ.get("BITTLE_CHECKPOINT_FREQ", 100_000))

# Auto-name so both conditions live in separate folders.
# Override: $env:BITTLE_RUN_NAME="my_run"
_default_name = "bittle_v2" if DOMAIN_RAND else "bittle_v1"
RUN_NAME      = os.environ.get("BITTLE_RUN_NAME", _default_name)

ROOT        = pathlib.Path(__file__).parent.parent
SAVE_DIR    = ROOT / "models" / RUN_NAME
MONITOR_DIR = SAVE_DIR / "monitor_logs"
CHECKPT_DIR = SAVE_DIR / "checkpoints"
MODEL_PATH   = SAVE_DIR / f"{RUN_NAME}_model"
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


def find_latest_checkpoint():
    """
    Look for the highest-step checkpoint from a previous (possibly
    interrupted) run. Returns (steps, model_path, vecnorm_path) or None.
    """
    if not CHECKPT_DIR.exists():
        return None
    pattern = re.compile(rf"^{re.escape(RUN_NAME)}_ckpt_(\d+)_steps\.zip$")
    best_steps, best_model = -1, None
    for f in CHECKPT_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps, best_model = steps, f
    if best_model is None:
        return None
    vecnorm = CHECKPT_DIR / f"{RUN_NAME}_ckpt_vecnormalize_{best_steps}_steps.pkl"
    if not vecnorm.exists():
        print(f"  Warning: found {best_model.name} but no matching vecnormalize "
              f"file — cannot safely resume from it. Starting fresh instead.")
        return None
    return best_steps, best_model, vecnorm


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

    def __init__(self, save_dir, initial_best_dist=-np.inf, eval_freq=100_000,
                 n_eval=3, verbose=1):
        super().__init__(verbose)
        self._save_dir  = pathlib.Path(save_dir)
        self._eval_freq = eval_freq
        self._n_eval    = n_eval
        self._best_dist = initial_best_dist
        self._last_eval = 0

        # Raw eval env (no VecNormalize — we borrow normalization from the
        # training env so the stats stay in sync).
        self._raw_env = DummyVecEnv([lambda: BittleEnv(domain_rand=False)])

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self._eval_freq:
            return True
        self._last_eval = self.num_timesteps

        distances, falls = [], 0
        for _ in range(self._n_eval):
            raw_obs   = self._raw_env.reset()
            start_y   = None
            done      = [False]
            steps     = 0
            while not done[0]:
                # Normalize using the TRAINING env's running statistics.
                norm_obs = self.training_env.normalize_obs(raw_obs)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                raw_obs, _, done, infos = self._raw_env.step(action)
                steps += 1
                if start_y is None:
                    start_y = infos[0].get("forward_position", 0.0)
                last_y = infos[0].get("forward_position", 0.0)
            distances.append(last_y - (start_y or 0.0))
            if steps < EPISODE_LENGTH_STEPS:
                falls += 1

        mean_dist = float(np.mean(distances))
        fall_rate = falls / self._n_eval
        # Require most eval episodes to survive the full episode before we
        # trust this distance -- otherwise a policy that lunges forward and
        # falls near the end can look like "best" by distance alone (this
        # happened during flat-ground tuning: 0.36m distance, fell at 70/500
        # steps). See RUN_GUIDE.md.
        stable = fall_rate <= 0.5

        marker = ""
        if stable and mean_dist > self._best_dist:
            self._best_dist = mean_dist
            self.model.save(str(self._save_dir / "best_model"))
            self.training_env.save(str(self._save_dir / "best_vecnormalize.pkl"))
            marker = "  ← NEW BEST, saved"

        if self.verbose:
            stability_note = "" if stable else "  [UNSTABLE -- not counted]"
            print(f"\n  [Best-model eval @ {self.num_timesteps:,} steps]  "
                  f"distance: {mean_dist:.3f} m  fall_rate: {fall_rate:.2f}  "
                  f"(best: {self._best_dist:.3f} m){marker}{stability_note}\n")
        return True

    def _on_training_end(self):
        self._raw_env.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    checkpoint = find_latest_checkpoint()

    print("=" * 60)
    print(f"  Run: {RUN_NAME}")
    print(f"  Domain randomization: {DOMAIN_RAND}")
    if checkpoint:
        done_steps, ckpt_model, ckpt_vecnorm = checkpoint
        print(f"  RESUMING from checkpoint: {done_steps:,} steps already done "
              f"({ckpt_model.name})")
    else:
        done_steps = 0
        print("  STARTING FRESH — no checkpoint found.")
    print(f"  Target total steps: {TOTAL_STEPS:,}   parallel envs: {PPO_CFG['n_envs']}")
    print(f"  Saving to: {SAVE_DIR}")
    print("=" * 60)

    remaining_steps = TOTAL_STEPS - done_steps
    if remaining_steps <= 0:
        print(f"\nAlready at/past target ({done_steps:,} >= {TOTAL_STEPS:,} steps). "
              f"Nothing to do — raise BITTLE_TIMESTEPS to train further.")
        return

    vec_env = DummyVecEnv([make_env(i) for i in range(PPO_CFG["n_envs"])])

    if checkpoint:
        env = VecNormalize.load(str(ckpt_vecnorm), vec_env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(str(ckpt_model), env=env, device="cpu")
        # PPO.load() restores hyperparameters AS THEY WERE WHEN SAVED -- config
        # changes made since then (like adding target_kl) do NOT apply unless
        # we explicitly override them here.
        model.target_kl = PPO_CFG["target_kl"]
        model.learning_rate = PPO_CFG["learning_rate"]
    else:
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
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
            target_kl       = PPO_CFG["target_kl"],
            policy_kwargs   = {"net_arch": PPO_CFG["net_arch"]},
            device          = "cpu",
            verbose         = 1,
        )

    callbacks = CallbackList([
        # Crash/close recovery: saves every CHECKPOINT_FREQ steps, and the
        # next run auto-resumes from the latest one instead of restarting.
        CheckpointCallback(
            save_freq        = CHECKPOINT_FREQ,
            save_path        = str(CHECKPT_DIR),
            name_prefix      = f"{RUN_NAME}_ckpt",
            save_vecnormalize = True,
        ),
        # Best model: evaluates on clean env, saves best by distance.
        BestDistanceCallback(
            save_dir  = SAVE_DIR,
            eval_freq = CHECKPOINT_FREQ,
            n_eval    = 3,
            verbose   = 1,
        ),
    ])

    print(f"\nTraining for {remaining_steps:,} more steps "
          f"(target {TOTAL_STEPS:,} total) …\n")
    model.learn(total_timesteps=remaining_steps,
                reset_num_timesteps=(checkpoint is None),
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
