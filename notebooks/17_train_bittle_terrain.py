"""
17_train_bittle_terrain.py — Teach the Bittle to walk over UNEVEN terrain.

This is the project's core original contribution: RL locomotion over rough
ground. The stock OpenCat gait handles flat ground, so training on flat ground
alone does not justify using RL. Uneven terrain does — a fixed gait cannot
adapt to bumps it has never seen, but a learned policy can.

HOW IT WORKS — automatic curriculum:
  The robot cannot learn on hard terrain from scratch (it just falls over and
  never discovers walking). So we start on FLAT ground (difficulty 0.0) and,
  every time the robot proves it can cross the current terrain, we raise the
  difficulty a little. By the end it walks over the full-height bumps.

  This is the same idea used to train real robots (ANYmal, Unitree) and is
  itself a technique you can explain to a judge.

Runs (blind locomotion — the robot feels terrain through its IMU + joints,
it does NOT get terrain heights; that matches the real Bittle's sensors):

  Terrain, no domain randomization (condition: terrain baseline):
      D:\\robot_venv\\Scripts\\python.exe notebooks\\17_train_bittle_terrain.py
      Saves to: models/bittle_terrain/

  Terrain + full domain randomization (the sim-to-real condition):
      (PowerShell)  $env:BITTLE_DOMAIN_RAND="1"
      D:\\robot_venv\\Scripts\\python.exe notebooks\\17_train_bittle_terrain.py
      Saves to: models/bittle_terrain_dr/

Quick test (50k steps):
      (PowerShell)  $env:BITTLE_TIMESTEPS=50000
"""

import os
import pathlib
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.sim.bittle_env import BittleEnv
from src.sim.config import PPO as PPO_CFG
from src.sim.config import TERRAIN

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DOMAIN_RAND = os.environ.get("BITTLE_DOMAIN_RAND", "0") == "1"
TOTAL_STEPS = int(os.environ.get("BITTLE_TIMESTEPS", PPO_CFG["total_timesteps"]))

_default_name = "bittle_terrain_dr" if DOMAIN_RAND else "bittle_terrain"
RUN_NAME = os.environ.get("BITTLE_RUN_NAME", _default_name)

ROOT        = pathlib.Path(__file__).parent.parent
SAVE_DIR    = ROOT / "models" / RUN_NAME
MONITOR_DIR = SAVE_DIR / "monitor_logs"
MODEL_PATH  = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

CUR = TERRAIN["curriculum"]


def make_env(rank):
    """One terrain Bittle environment, starting flat (curriculum raises it)."""
    def _init():
        env = BittleEnv(
            domain_rand=DOMAIN_RAND,
            terrain=True,
            terrain_difficulty=CUR["start_difficulty"],
        )
        env = Monitor(env, str(MONITOR_DIR / f"env_{rank}"),
                      info_keywords=("terrain_difficulty",))
        return env
    return _init


# ---------------------------------------------------------------------------
# CURRICULUM CALLBACK
# ---------------------------------------------------------------------------

class TerrainCurriculumCallback(BaseCallback):
    """
    Periodically checks how far the robot walks on the CURRENT difficulty.
    When it walks far enough (promote_distance), raise the difficulty on all
    training envs by `step`, up to max_difficulty. Also saves the best model.

    Why evaluate on the current difficulty (not flat)? Because we want proof the
    robot can handle the terrain it is actually training on before making it
    harder — otherwise we would promote it into terrain it cannot walk on yet.
    """

    def __init__(self, save_dir, eval_freq=100_000, n_eval=3, verbose=1):
        super().__init__(verbose)
        self._save_dir  = pathlib.Path(save_dir)
        self._eval_freq = eval_freq
        self._n_eval    = n_eval
        self._difficulty = CUR["start_difficulty"]
        self._best_dist  = -np.inf
        self._last_eval  = 0
        # Eval env mirrors the training terrain difficulty.
        self._eval_env = DummyVecEnv([lambda: BittleEnv(
            domain_rand=False, terrain=True,
            terrain_difficulty=CUR["start_difficulty"])])

    def _set_all_difficulty(self, difficulty):
        """Raise difficulty on every parallel training env and the eval env."""
        difficulty = float(min(difficulty, CUR["max_difficulty"]))
        self._difficulty = difficulty
        # env_method broadcasts a method call to every sub-env in the VecEnv.
        self.training_env.env_method("set_terrain_difficulty", difficulty)
        self._eval_env.env_method("set_terrain_difficulty", difficulty)
        return difficulty

    def _evaluate_distance(self):
        distances = []
        for _ in range(self._n_eval):
            raw_obs = self._eval_env.reset()
            start_y, last_y, done = None, 0.0, [False]
            while not done[0]:
                norm_obs = self.training_env.normalize_obs(raw_obs)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                raw_obs, _, done, infos = self._eval_env.step(action)
                if start_y is None:
                    start_y = infos[0].get("forward_position", 0.0)
                last_y = infos[0].get("forward_position", 0.0)
            distances.append(last_y - (start_y or 0.0))
        return float(np.mean(distances))

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self._eval_freq:
            return True
        self._last_eval = self.num_timesteps

        mean_dist = self._evaluate_distance()

        # Save best model (by distance on current terrain).
        best_marker = ""
        if mean_dist > self._best_dist:
            self._best_dist = mean_dist
            self.model.save(str(self._save_dir / "best_model"))
            self.training_env.save(str(self._save_dir / "best_vecnormalize.pkl"))
            best_marker = "  ← new best, saved"

        # Promote difficulty if the robot crossed the current terrain well.
        promoted = ""
        if (CUR["enabled"]
                and mean_dist >= CUR["promote_distance"]
                and self._difficulty < CUR["max_difficulty"]):
            new_d = self._set_all_difficulty(self._difficulty + CUR["step"])
            promoted = f"  ↑ difficulty raised to {new_d:.2f}"

        if self.verbose:
            print(f"\n  [Curriculum @ {self.num_timesteps:,} steps]  "
                  f"difficulty {self._difficulty:.2f}  |  "
                  f"distance {mean_dist:.3f} m  "
                  f"(best {self._best_dist:.3f}){best_marker}{promoted}\n")
        return True

    def _on_training_end(self):
        self._eval_env.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"  Run: {RUN_NAME}   (UNEVEN TERRAIN)")
    print(f"  Domain randomization: {DOMAIN_RAND}")
    print(f"  Curriculum: {CUR['start_difficulty']} → {CUR['max_difficulty']} "
          f"(step {CUR['step']}, promote at {CUR['promote_distance']} m)")
    print(f"  Steps: {TOTAL_STEPS:,}   parallel envs: {PPO_CFG['n_envs']}")
    print(f"  Saving to: {SAVE_DIR}")
    print("=" * 60)

    vec_env = DummyVecEnv([make_env(i) for i in range(PPO_CFG["n_envs"])])
    env     = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        learning_rate = PPO_CFG["learning_rate"],
        n_steps       = PPO_CFG["n_steps"],
        batch_size    = PPO_CFG["batch_size"],
        n_epochs      = PPO_CFG["n_epochs"],
        gamma         = PPO_CFG["gamma"],
        gae_lambda    = PPO_CFG["gae_lambda"],
        clip_range    = PPO_CFG["clip_range"],
        ent_coef      = PPO_CFG["ent_coef"],
        vf_coef       = PPO_CFG["vf_coef"],
        max_grad_norm = PPO_CFG["max_grad_norm"],
        policy_kwargs = {"net_arch": PPO_CFG["net_arch"]},
        device        = "cpu",
        verbose       = 1,
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq        = 100_000,
            save_path        = str(SAVE_DIR / "checkpoints"),
            name_prefix      = f"{RUN_NAME}_ckpt",
            save_vecnormalize = True,
        ),
        TerrainCurriculumCallback(
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


if __name__ == "__main__":
    main()
