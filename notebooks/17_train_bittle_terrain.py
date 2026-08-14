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
import torch
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

# Escape hatch: resume training from best_model.zip instead of the latest
# checkpoint. Use this if "latest" has drifted into a worse policy than the
# best one ever recorded (e.g. after a long unstable stretch before target_kl
# was added) -- best_model is the last snapshot that passed the stability gate.
RESUME_FROM_BEST = os.environ.get("BITTLE_RESUME_FROM_BEST", "0") == "1"

# Recovery knob for an already-collapsed policy: BITTLE_RESET_STD=0.5 resets the
# resumed policy's action stddev so it can explore again. See config.py's
# "reset_std_default" comment for the full diagnosis of why this is needed.
_reset_std_raw = os.environ.get("BITTLE_RESET_STD", "")
RESET_STD = float(_reset_std_raw) if _reset_std_raw else None

_default_name = "bittle_terrain_dr" if DOMAIN_RAND else "bittle_terrain"
RUN_NAME = os.environ.get("BITTLE_RUN_NAME", _default_name)

# Random seed. Until this existed, NOTHING in this script was seeded -- not PPO's
# weight init, not the terrain generator -- so re-running the same config gave a
# different answer every time. That makes any A/B comparison ("did change X help?")
# uninterpretable, because you cannot tell a real effect from run-to-run luck.
# Set BITTLE_SEED to a different integer per arm to measure seed variance.
SEED = int(os.environ.get("BITTLE_SEED", "0"))

# Pin the terrain difficulty and disable curriculum promotion. Used for controlled
# A/B experiments: it holds the operating point fixed so a measured difference comes
# from the change under test, not from one arm happening to promote earlier.
_fixed_diff_raw = os.environ.get("BITTLE_FIXED_DIFFICULTY", "")
FIXED_DIFFICULTY = float(_fixed_diff_raw) if _fixed_diff_raw else None

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


def _fresh_curriculum_state():
    return {
        "version": 2,
        "difficulty": CUR["start_difficulty"],
        # Best is ranked LEXICOGRAPHICALLY on (difficulty, score) -- see
        # TerrainCurriculumCallback._is_better for why a single scalar was wrong.
        "best_difficulty": -1.0,
        "best_score": -np.inf,
        "best_by_difficulty": {},   # "0.90" -> best score seen at that rung
        "bad_evals": 0,
        "steps_at_difficulty": 0,
    }


def load_curriculum_state():
    """
    Read back the curriculum state from the last run.

    v1 files stored a single global `best_dist` compared across ALL difficulties.
    That was a bug: distance naturally FALLS as terrain gets harder, so a best set
    on easy ground could never be beaten on hard ground, which froze best_model.zip
    at an early, undertrained snapshot for entire 5M-step runs. We deliberately
    DISCARD that number on upgrade rather than migrating it -- it is precisely the
    corrupted quantity -- and keep only the difficulty we had reached.
    """
    if not CURRICULUM_STATE_PATH.exists():
        return _fresh_curriculum_state()

    with open(CURRICULUM_STATE_PATH) as f:
        state = json.load(f)

    if state.get("version", 1) < 2:
        print("  NOTE: found a v1 curriculum_state.json. Its 'best_dist' was a "
              "cross-difficulty global scalar (known bug) -- DISCARDING it and "
              "keeping the difficulty only. best_model will be re-established.")
        fresh = _fresh_curriculum_state()
        fresh["difficulty"] = state.get("difficulty", CUR["start_difficulty"])
        return fresh

    return state


def make_learning_rate():
    """
    Build the learning rate PPO should use -- either a constant, or a schedule.

    SB3 accepts either a float OR a callable taking `progress_remaining`, which
    runs from 1.0 at the start of training down to 0.0 at the end. A linearly
    decaying rate takes big steps early (when the policy is bad and moving fast
    is cheap) and small steps late (when it is fine-tuning and a big step would
    wreck what it has). Controlled by PPO["lr_schedule"] in config.py so the
    change can be measured on its own instead of being tangled with other edits.
    """
    base = PPO_CFG["learning_rate"]
    if PPO_CFG.get("lr_schedule", "constant") != "linear":
        return base

    def linear(progress_remaining):
        return progress_remaining * base

    return linear


def apply_resume_overrides(model, reset_std=None):
    """
    Re-apply current config to a model loaded from a checkpoint, and optionally
    rescue a collapsed action stddev.

    PPO.load() restores hyperparameters AS THEY WERE WHEN SAVED, so edits made
    to config.py since that checkpoint do NOT apply unless set here explicitly.

    reset_std: if given, force the policy's action stddev to this value. This is
    the recovery path for an already-collapsed policy (see config.py). It MUST
    be an in-place .data edit: replacing the log_std Parameter object leaves
    SB3's optimizer holding the OLD tensor, so the reset would silently never
    train. Verified experimentally.
    """
    model.target_kl = PPO_CFG["target_kl"]
    model.ent_coef = PPO_CFG["ent_coef"]

    # Learning rate needs BOTH lines. SB3 does not read `model.learning_rate`
    # during training -- it reads `model.lr_schedule`, which PPO.load() already
    # built from the value baked into the checkpoint. So assigning the attribute
    # alone silently does nothing, and every resumed run kept training at the OLD
    # learning rate. _setup_lr_schedule() rebuilds the schedule from the new value.
    # (ent_coef and target_kl above ARE read directly each update, so they work.)
    model.learning_rate = make_learning_rate()
    model._setup_lr_schedule()

    std_before = float(np.exp(model.policy.log_std.detach().cpu().numpy()).mean())
    if reset_std is not None:
        with torch.no_grad():
            model.policy.log_std.data.fill_(float(np.log(reset_std)))
        std_after = float(np.exp(model.policy.log_std.detach().cpu().numpy()).mean())
        print(f"  RESET policy action stddev: {std_before:.4f} -> {std_after:.4f} "
              f"(exploration restored)")
    else:
        print(f"  Policy action stddev on resume: {std_before:.4f}"
              + ("   <-- COLLAPSED; consider BITTLE_RESET_STD=0.5"
                 if std_before < 0.25 else ""))
    print(f"  ent_coef={model.ent_coef}  target_kl={model.target_kl}")
    return model


# ---------------------------------------------------------------------------
# STD CLAMP CALLBACK
# ---------------------------------------------------------------------------

class StdClampCallback(BaseCallback):
    """
    Keeps the policy's action stddev inside [std_clamp_min, std_clamp_max].

    PPO's entropy bonus has no ceiling or floor of its own -- over a long
    enough run std can drift to either extreme (collapse: too rigid to
    explore; explosion: actions close to random). Checked every rollout
    (each call to _on_step is one env step across all parallel envs, so this
    runs far more often than eval -- cheap, and catches drift early).
    """

    def __init__(self, std_min, std_max, verbose=0):
        super().__init__(verbose)
        self._log_min = float(np.log(std_min))
        self._log_max = float(np.log(std_max))

    def _on_step(self):
        with torch.no_grad():
            self.model.policy.log_std.data.clamp_(self._log_min, self._log_max)
        return True


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

    Persists its state to disk every eval so a resumed run picks up the
    curriculum where it left off, instead of restarting on flat ground.

    Evaluation always runs on FIXED seeds (EVAL_SEEDS) so every evaluation faces
    the SAME set of terrains. Without that, consecutive evaluations differ because
    the map changed, not because the policy did -- you would be measuring the dice.
    The eval env also keeps domain_rand=False even in DR runs: it is the measuring
    instrument, so it stays fixed even when training conditions vary.
    """

    def __init__(self, save_dir, state_path, state, allow_promotion=True,
                 eval_freq=100_000, n_eval=10, verbose=1):
        super().__init__(verbose)
        self._save_dir   = pathlib.Path(save_dir)
        self._state_path = pathlib.Path(state_path)
        self._eval_freq  = eval_freq
        self._n_eval     = n_eval
        self._allow_promotion = allow_promotion

        self._difficulty      = state["difficulty"]
        self._best_difficulty = state["best_difficulty"]
        self._best_score      = state["best_score"]
        self._best_by_diff    = dict(state["best_by_difficulty"])
        self._bad_evals       = state["bad_evals"]
        self._steps_at_diff   = state["steps_at_difficulty"]

        self._last_eval  = 0
        self._last_eval_steps = 0
        self._eval_seeds = [9000 + i for i in range(n_eval)]
        self._eval_env = DummyVecEnv([lambda: BittleEnv(
            domain_rand=False, terrain=True, terrain_difficulty=state["difficulty"])])

    def _is_better(self, difficulty, score):
        """
        Rank checkpoints lexicographically: harder terrain wins outright, and
        only ties on difficulty are broken by distance.

        The old code compared one global distance across ALL difficulties, which
        is comparing apples to oranges -- distance naturally drops as the ground
        gets harder, so an easy-terrain best could never be beaten and best_model
        stayed frozen on a barely-trained snapshot for an entire run. Walking
        0.4 m over rock genuinely beats walking 0.9 m over flat ground.
        """
        if difficulty > self._best_difficulty + 1e-9:
            return True
        if abs(difficulty - self._best_difficulty) < 1e-9:
            return score > self._best_score
        return False

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
            json.dump({
                "version": 2,
                "difficulty": self._difficulty,
                "best_difficulty": self._best_difficulty,
                "best_score": self._best_score,
                "best_by_difficulty": self._best_by_diff,
                "bad_evals": self._bad_evals,
                "steps_at_difficulty": self._steps_at_diff,
            }, f, indent=2)

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
        for seed in self._eval_seeds:
            # Same terrains every evaluation, so a change in the number means the
            # POLICY changed, not the map. VecEnv.seed() applies at the next reset.
            self._eval_env.seed(seed)
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

        self._steps_at_diff += self.num_timesteps - self._last_eval_steps
        self._last_eval_steps = self.num_timesteps

        mean_dist, fall_rate = self._evaluate_distance()
        stable = fall_rate <= CUR["promote_max_fall_rate"]

        # --- Best model: lexicographic on (difficulty, distance). See _is_better.
        best_marker = ""
        if stable and self._is_better(self._difficulty, mean_dist):
            self._best_difficulty = self._difficulty
            self._best_score = mean_dist
            self.model.save(str(self._save_dir / "best_model"))
            self.training_env.save(str(self._save_dir / "best_vecnormalize.pkl"))
            # Metadata so a benchmark can NEVER again silently measure an early
            # snapshot without us noticing -- 16_terrain_benchmark.py prints this.
            with open(self._save_dir / "best_model_meta.json", "w") as f:
                json.dump({
                    "difficulty": self._difficulty,
                    "score_distance_m": round(mean_dist, 4),
                    "fall_rate": fall_rate,
                    "timesteps": int(self.num_timesteps),
                    "n_eval_episodes": self._n_eval,
                }, f, indent=2)
            best_marker = "  ← new best, saved"

        # Track the best at EVERY rung, so a later demotion can never erase the
        # evidence that we once walked well on the hardest terrain.
        rung = f"{self._difficulty:.2f}"
        if stable and mean_dist > self._best_by_diff.get(rung, -np.inf):
            self._best_by_diff[rung] = round(mean_dist, 4)

        # --- Promotion / demotion
        transition = ""
        promote = (CUR["enabled"] and self._allow_promotion and stable
                   and mean_dist >= CUR["promote_distance"]
                   and self._steps_at_diff >= CUR["min_steps_at_difficulty"]
                   and self._difficulty < CUR["max_difficulty"])

        bad = (fall_rate >= CUR["demote_fall_rate"]
               or mean_dist < CUR["demote_distance"])
        self._bad_evals = self._bad_evals + 1 if bad else 0
        demote = (CUR["enabled"] and self._allow_promotion and not promote
                  and self._bad_evals >= CUR["demote_after_bad_evals"]
                  and self._difficulty > CUR["start_difficulty"])

        if promote:
            new_d = self._set_all_difficulty(self._difficulty + CUR["step"])
            transition = f"  ↑ difficulty raised to {new_d:.2f}"
            self._bad_evals, self._steps_at_diff = 0, 0
        elif demote:
            # Not a failure -- it means we promoted onto ground the policy could
            # not actually hold. Dropping back lets it consolidate and re-climb.
            new_d = self._set_all_difficulty(self._difficulty - CUR["step"])
            transition = f"  ↓ difficulty LOWERED to {new_d:.2f} (2 bad evals)"
            self._bad_evals, self._steps_at_diff = 0, 0

        self._save_state()

        if self.verbose:
            note = "" if stable else "  [UNSTABLE -- not counted]"
            best_str = ("none yet" if self._best_difficulty < 0
                        else f"{self._best_score:.3f} m @ d{self._best_difficulty:.2f}")
            print(f"\n  [Curriculum @ {self.num_timesteps:,} steps]  "
                  f"difficulty {self._difficulty:.2f}  |  "
                  f"distance {mean_dist:.3f} m  fall_rate {fall_rate:.2f}  "
                  f"(best {best_str}){best_marker}{transition}{note}\n")
        return True

    def _on_training_end(self):
        self._eval_env.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    state = load_curriculum_state()
    difficulty = state["difficulty"]

    if FIXED_DIFFICULTY is not None:
        difficulty = FIXED_DIFFICULTY
        state["difficulty"] = FIXED_DIFFICULTY

    vec_env = DummyVecEnv([make_env(i, difficulty) for i in range(PPO_CFG["n_envs"])])

    print("=" * 60)
    print(f"  Run: {RUN_NAME}   (UNEVEN TERRAIN)")
    print(f"  Domain randomization: {DOMAIN_RAND}")
    print(f"  Seed: {SEED}")
    if FIXED_DIFFICULTY is not None:
        print(f"  FIXED difficulty {FIXED_DIFFICULTY:.2f} — curriculum promotion DISABLED "
              f"(controlled A/B mode)")

    if RESUME_FROM_BEST:
        # Deliberate escape hatch: the "latest" checkpoint can end up worse
        # than best_model if training spent a long stretch unstable (this is
        # why we added target_kl -- but that doesn't undo damage already
        # done to a policy that kept training through the instability).
        # best_model.zip is the last snapshot that passed the stability gate,
        # so we resume training from THERE instead, now protected by target_kl.
        best_model_path   = SAVE_DIR / "best_model"
        best_vecnorm_path = SAVE_DIR / "best_vecnormalize.pkl"
        if not (best_model_path.with_suffix(".zip").exists() and best_vecnorm_path.exists()):
            print("  ERROR: BITTLE_RESUME_FROM_BEST=1 set but no best_model found "
                  f"in {SAVE_DIR}.")
            return
        env = VecNormalize.load(str(best_vecnorm_path), vec_env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(str(best_model_path), env=env, device="cpu")
        done_steps = model.num_timesteps
        resumed = True
        print(f"  RESUMING FROM BEST_MODEL (skipping the 'latest' checkpoint on purpose): "
              f"{done_steps:,} steps")
        apply_resume_overrides(model, reset_std=RESET_STD)
    else:
        checkpoint = find_latest_checkpoint()
        if checkpoint:
            done_steps, ckpt_model, ckpt_vecnorm = checkpoint
            env = VecNormalize.load(str(ckpt_vecnorm), vec_env)
            env.training = True
            env.norm_reward = True
            model = PPO.load(str(ckpt_model), env=env, device="cpu")
            resumed = True
            print(f"  RESUMING from checkpoint: {done_steps:,} steps already done "
                  f"({ckpt_model.name})")
            apply_resume_overrides(model, reset_std=RESET_STD)
        else:
            done_steps = 0
            resumed = False
            print("  STARTING FRESH — no checkpoint found.")
            env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            model = PPO(
                "MlpPolicy", env,
                learning_rate = make_learning_rate(),
                seed          = SEED,
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

    _bd = state["best_difficulty"]
    _best_str = ("none yet" if _bd < 0
                 else f"{state['best_score']:.3f} m @ difficulty {_bd:.2f}")
    print(f"  Curriculum resumes at difficulty {difficulty:.2f}  (best so far: {_best_str})")
    print(f"  Curriculum: {CUR['start_difficulty']} → {CUR['max_difficulty']} "
          f"(step {CUR['step']}, promote at {CUR['promote_distance']} m "
          f"and fall_rate ≤ {CUR['promote_max_fall_rate']})")
    print(f"  Target total steps: {TOTAL_STEPS:,}   parallel envs: {PPO_CFG['n_envs']}")
    print(f"  Saving to: {SAVE_DIR}")
    print("=" * 60)

    remaining_steps = TOTAL_STEPS - done_steps
    if remaining_steps <= 0:
        print(f"\nAlready at/past target ({done_steps:,} >= {TOTAL_STEPS:,} steps). "
              f"Nothing to do — raise BITTLE_TIMESTEPS to train further.")
        return

    callbacks = CallbackList([
        CheckpointCallback(
            # CheckpointCallback counts VECTORISED steps, not environment steps:
            # its counter ticks once per rollout step across all envs at once. So
            # passing 100_000 with 4 envs actually saved every 400k real steps --
            # 4x less often than intended, and it would get worse as n_envs grows.
            # Dividing here makes the interval mean what the name says.
            save_freq         = max(1, CHECKPOINT_FREQ // PPO_CFG["n_envs"]),
            save_path         = str(CHECKPT_DIR),
            name_prefix       = f"{RUN_NAME}_ckpt",
            save_vecnormalize = True,
        ),
        StdClampCallback(
            std_min = PPO_CFG["std_clamp_min"],
            std_max = PPO_CFG["std_clamp_max"],
        ),
        TerrainCurriculumCallback(
            save_dir        = SAVE_DIR,
            state_path      = CURRICULUM_STATE_PATH,
            state           = state,
            allow_promotion = (FIXED_DIFFICULTY is None),
            eval_freq       = CHECKPOINT_FREQ,
            # 10, not 3: with 3 episodes fall_rate could only be 0/0.33/0.67/1.0,
            # so every promotion decision was made on what was effectively a coin
            # flip. 10 fixed-seed episodes cost ~8 s and make the number mean something.
            n_eval          = 10,
            verbose         = 1,
        ),
    ])

    print(f"\nTraining for {remaining_steps:,} more steps "
          f"(target {TOTAL_STEPS:,} total) …\n")
    model.learn(total_timesteps=remaining_steps,
                reset_num_timesteps=(not resumed),
                callback=callbacks)

    print("\nSaving final model …")
    model.save(str(MODEL_PATH))
    env.save(str(VECNORM_PATH))
    env.close()

    print(f"\n  Final model → {MODEL_PATH}.zip")
    print(f"  Best model  → {SAVE_DIR / 'best_model'}.zip")


if __name__ == "__main__":
    main()
