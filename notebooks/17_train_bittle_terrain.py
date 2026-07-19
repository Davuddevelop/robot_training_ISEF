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

AUTO-RESUME (IMPORTANT — read this if you've been closing terminals):
  Closing the terminal kills this script mid-run. This script now DETECTS
  that on the next run and picks up exactly where it left off — same model
  weights, same step count, same curriculum difficulty — instead of starting
  over from a brand-new random policy. You lose at most the ~100k steps
  since the last checkpoint, never the whole run.

  Just re-run the exact same command again. It prints "RESUMING" if it found
  a checkpoint, or "STARTING FRESH" if this is a new run.

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

import json
import os
import pathlib
import re
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
from src.sim.config import TERRAIN, EPISODE_LENGTH_STEPS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DOMAIN_RAND = os.environ.get("BITTLE_DOMAIN_RAND", "0") == "1"
TOTAL_STEPS = int(os.environ.get("BITTLE_TIMESTEPS", PPO_CFG["total_timesteps"]))
CHECKPOINT_FREQ = int(os.environ.get("BITTLE_CHECKPOINT_FREQ", 100_000))

_default_name = "bittle_terrain_dr" if DOMAIN_RAND else "bittle_terrain"
RUN_NAME = os.environ.get("BITTLE_RUN_NAME", _default_name)

ROOT         = pathlib.Path(__file__).parent.parent
SAVE_DIR     = ROOT / "models" / RUN_NAME
MONITOR_DIR  = SAVE_DIR / "monitor_logs"
CHECKPT_DIR  = SAVE_DIR / "checkpoints"
MODEL_PATH   = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"
CURRICULUM_STATE_PATH = SAVE_DIR / "curriculum_state.json"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

CUR = TERRAIN["curriculum"]


def make_env(rank, start_difficulty):
    """One terrain Bittle environment, starting at whatever difficulty we resume at."""
    def _init():
        env = BittleEnv(
            domain_rand=DOMAIN_RAND,
            terrain=True,
            terrain_difficulty=start_difficulty,
        )
        env = Monitor(env, str(MONITOR_DIR / f"env_{rank}"),
                      info_keywords=("terrain_difficulty",))
        return env
    return _init


def find_latest_checkpoint():
    """
    Look for the highest-step checkpoint saved by a previous (possibly
    interrupted) run of this script. Returns (steps, model_path, vecnorm_path)
    or None if no checkpoint exists yet.
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


def load_curriculum_state():
    """Read back the curriculum's difficulty/best-distance from the last run."""
    if CURRICULUM_STATE_PATH.exists():
        with open(CURRICULUM_STATE_PATH) as f:
            state = json.load(f)
        return state["difficulty"], state["best_dist"]
    return CUR["start_difficulty"], -np.inf


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

    Persists (difficulty, best_dist) to disk every eval so a resumed run picks
    up the curriculum where it left off, instead of restarting on flat ground.
    """

    def __init__(self, save_dir, state_path, initial_difficulty, initial_best_dist,
                 eval_freq=100_000, n_eval=3, verbose=1):
        super().__init__(verbose)
        self._save_dir   = pathlib.Path(save_dir)
        self._state_path = pathlib.Path(state_path)
        self._eval_freq  = eval_freq
        self._n_eval     = n_eval
        self._difficulty = initial_difficulty
        self._best_dist  = initial_best_dist
        self._last_eval  = 0
        self._eval_env = DummyVecEnv([lambda: BittleEnv(
            domain_rand=False, terrain=True, terrain_difficulty=initial_difficulty)])

    def _on_training_start(self):
        # Apply the (possibly resumed) difficulty to every training env and
        # the eval env — they all default to CUR["start_difficulty"] otherwise.
        self._set_all_difficulty(self._difficulty)

    def _set_all_difficulty(self, difficulty):
        difficulty = float(min(difficulty, CUR["max_difficulty"]))
        self._difficulty = difficulty
        self.training_env.env_method("set_terrain_difficulty", difficulty)
        self._eval_env.env_method("set_terrain_difficulty", difficulty)
        return difficulty

    def _save_state(self):
        with open(self._state_path, "w") as f:
            json.dump({"difficulty": self._difficulty, "best_dist": self._best_dist}, f)

    def _evaluate_distance(self):
        """
        Returns (mean_distance, fall_rate).

        A policy can score high mean distance by lunging forward and toppling
        near the end of the episode -- distance alone can't tell that apart
        from genuine stable walking (we saw exactly this on flat ground months
        ago: a run scored well on distance but fell at 70/500 steps). So we
        also track fall_rate: an episode "falls" if it ends before reaching
        EPISODE_LENGTH_STEPS (the only way an episode ends early is _is_fallen
        triggering termination -- reaching the step limit is truncation, not
        a fall).
        """
        distances, falls = [], 0
        for _ in range(self._n_eval):
            raw_obs = self._eval_env.reset()
            start_y, last_y, done, steps = None, 0.0, [False], 0
            while not done[0]:
                norm_obs = self.training_env.normalize_obs(raw_obs)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                raw_obs, _, done, infos = self._eval_env.step(action)
                steps += 1
                if start_y is None:
                    start_y = infos[0].get("forward_position", 0.0)
                last_y = infos[0].get("forward_position", 0.0)
            distances.append(last_y - (start_y or 0.0))
            if steps < EPISODE_LENGTH_STEPS:
                falls += 1
        return float(np.mean(distances)), falls / self._n_eval

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self._eval_freq:
            return True
        self._last_eval = self.num_timesteps

        mean_dist, fall_rate = self._evaluate_distance()
        # Require the MAJORITY of eval episodes to survive the full episode
        # before we trust this distance number at all. Otherwise a policy
        # that lunges forward and falls near the end can look like "best" by
        # distance alone, even though it never walks stably.
        stable = fall_rate <= 0.5

        best_marker = ""
        if stable and mean_dist > self._best_dist:
            self._best_dist = mean_dist
            self.model.save(str(self._save_dir / "best_model"))
            self.training_env.save(str(self._save_dir / "best_vecnormalize.pkl"))
            best_marker = "  ← new best, saved"

        promoted = ""
        if (CUR["enabled"] and stable
                and mean_dist >= CUR["promote_distance"]
                and self._difficulty < CUR["max_difficulty"]):
            new_d = self._set_all_difficulty(self._difficulty + CUR["step"])
            promoted = f"  ↑ difficulty raised to {new_d:.2f}"

        self._save_state()   # persist difficulty + best_dist so a resume picks these up

        if self.verbose:
            stability_note = "" if stable else "  [UNSTABLE -- not counted]"
            print(f"\n  [Curriculum @ {self.num_timesteps:,} steps]  "
                  f"difficulty {self._difficulty:.2f}  |  "
                  f"distance {mean_dist:.3f} m  fall_rate {fall_rate:.2f}  "
                  f"(best {self._best_dist:.3f}){best_marker}{promoted}{stability_note}\n")
        return True

    def _on_training_end(self):
        self._eval_env.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    checkpoint = find_latest_checkpoint()
    difficulty, best_dist = load_curriculum_state()

    print("=" * 60)
    print(f"  Run: {RUN_NAME}   (UNEVEN TERRAIN)")
    print(f"  Domain randomization: {DOMAIN_RAND}")
    if checkpoint:
        done_steps, ckpt_model, ckpt_vecnorm = checkpoint
        print(f"  RESUMING from checkpoint: {done_steps:,} steps already done "
              f"({ckpt_model.name})")
        print(f"  Curriculum resumes at difficulty {difficulty:.2f} "
              f"(best distance so far: {best_dist:.3f} m)")
    else:
        done_steps = 0
        print("  STARTING FRESH — no checkpoint found.")
    print(f"  Curriculum: {CUR['start_difficulty']} → {CUR['max_difficulty']} "
          f"(step {CUR['step']}, promote at {CUR['promote_distance']} m)")
    print(f"  Target total steps: {TOTAL_STEPS:,}   parallel envs: {PPO_CFG['n_envs']}")
    print(f"  Saving to: {SAVE_DIR}")
    print("=" * 60)

    remaining_steps = TOTAL_STEPS - done_steps
    if remaining_steps <= 0:
        print(f"\nAlready at/past target ({done_steps:,} >= {TOTAL_STEPS:,} steps). "
              f"Nothing to do — raise BITTLE_TIMESTEPS to train further.")
        return

    vec_env = DummyVecEnv([make_env(i, difficulty) for i in range(PPO_CFG["n_envs"])])

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
            target_kl     = PPO_CFG["target_kl"],
            policy_kwargs = {"net_arch": PPO_CFG["net_arch"]},
            device        = "cpu",
            verbose       = 1,
        )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq         = CHECKPOINT_FREQ,
            save_path         = str(CHECKPT_DIR),
            name_prefix       = f"{RUN_NAME}_ckpt",
            save_vecnormalize = True,
        ),
        TerrainCurriculumCallback(
            save_dir            = SAVE_DIR,
            state_path          = CURRICULUM_STATE_PATH,
            initial_difficulty  = difficulty,
            initial_best_dist   = best_dist,
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


if __name__ == "__main__":
    main()
