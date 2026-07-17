"""
16_terrain_benchmark.py — The headline experiment: RL policy vs. fixed gait
                          over identical terrain at rising difficulty.

This is the comparison that shows WHY reinforcement learning is needed. Both
controllers walk the SAME robot over the SAME randomly-generated terrain (we
seed each episode so both face an identical map), at difficulty levels from
flat (0.0) to rough (1.0). We measure, for each:

    distance   — how far forward it walked (metres)   [the main result]
    fall_rate  — fraction of episodes it fell         [robustness]
    survival   — mean control steps before falling / timeout

THE PREDICTION:
    On flat ground both do fine. As difficulty rises, the fixed gait's numbers
    collapse (it can't react to bumps) while the RL policy degrades gracefully.
    That growing gap is the evidence that learning matters on rough terrain.

Run AFTER training a terrain policy (17_train_bittle_terrain.py):
    D:\\robot_venv\\Scripts\\python.exe notebooks\\16_terrain_benchmark.py

If no trained model exists yet, it still runs the fixed-gait half so you can
see the baseline; the RL columns will say "no model".
"""

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.sim.bittle_env import BittleEnv
from src.sim.scripted_gait import ScriptedTrotGait
from src.sim.config import NEUTRAL_POSE, ACTION_LIMIT, CONTROL_TIMESTEP

# Difficulty levels to test (flat → rough).
DIFFICULTIES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_EPISODES   = 10          # episodes per controller per difficulty (rule: >=10 trials)

RUN_NAME     = "bittle_terrain"
ROOT         = pathlib.Path(__file__).parent.parent
SAVE_DIR     = ROOT / "models" / RUN_NAME
# Prefer best_model (highest distance seen during training) over the final
# model — PPO can degrade late in training, so "final" is not always "best".
# Same preference 18_watch_bittle_terrain.py already uses.
MODEL_PATH   = SAVE_DIR / "best_model"
VECNORM_PATH = SAVE_DIR / "best_vecnormalize.pkl"
_FINAL_MODEL_PATH   = SAVE_DIR / f"{RUN_NAME}_model"
_FINAL_VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"
if not (MODEL_PATH.with_suffix(".zip").exists() and VECNORM_PATH.exists()):
    MODEL_PATH, VECNORM_PATH = _FINAL_MODEL_PATH, _FINAL_VECNORM_PATH
DATA_DIR     = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NEUTRAL = np.array(NEUTRAL_POSE, dtype=np.float64)


def scripted_action_for(angles):
    """
    Convert absolute target joint angles into the env's action format.

    The env expects an action in [-1, 1] that it scales to an offset from
    neutral (offset = action * ACTION_LIMIT, target = neutral + offset). So to
    command absolute angles `a`, we invert that: action = (a - neutral)/LIMIT.
    """
    return np.clip((angles - NEUTRAL) / ACTION_LIMIT, -1.0, 1.0)


def run_fixed_gait(difficulty, n_episodes):
    """Run the fixed open-loop trot; return (mean_distance, fall_rate, mean_survival)."""
    env = BittleEnv(terrain=True, terrain_difficulty=difficulty)
    gait = ScriptedTrotGait()
    distances, falls, survivals = [], 0, []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)   # same seeds used for RL → identical terrain
        gait.reset()
        start_y = env._mj_data.qpos[1]
        t, steps, fell = 0.0, 0, False
        while True:
            angles = gait.step(t)
            action = scripted_action_for(angles)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            t += CONTROL_TIMESTEP
            if terminated or truncated:
                fell = terminated   # terminated = fell; truncated = survived to timeout
                break
        distances.append(float(env._mj_data.qpos[1] - start_y))
        survivals.append(steps)
        falls += int(fell)

    env.close()
    return np.mean(distances), falls / n_episodes, np.mean(survivals)


def run_rl_policy(difficulty, n_episodes, model, norm):
    """Run the trained RL policy; return (mean_distance, fall_rate, mean_survival)."""
    env = BittleEnv(terrain=True, terrain_difficulty=difficulty)
    distances, falls, survivals = [], 0, []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)   # SAME seeds as fixed gait → identical terrain
        start_y = env._mj_data.qpos[1]
        steps, fell = 0, False
        while True:
            norm_obs = norm.normalize_obs(obs)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                fell = terminated
                break
        distances.append(float(env._mj_data.qpos[1] - start_y))
        survivals.append(steps)
        falls += int(fell)

    env.close()
    return np.mean(distances), falls / n_episodes, np.mean(survivals)


def main():
    # Try to load the RL policy; if missing, run the fixed-gait half only.
    model = norm = None
    if MODEL_PATH.with_suffix(".zip").exists() and VECNORM_PATH.exists():
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        which = "best_model" if MODEL_PATH.name == "best_model" else "final model"
        print(f"Loading RL policy: {RUN_NAME} ({which})")
        model = PPO.load(str(MODEL_PATH))
        norm  = VecNormalize.load(str(VECNORM_PATH),
                                  DummyVecEnv([lambda: BittleEnv(terrain=True)]))
        norm.training = False
    else:
        print(f"No trained model at {MODEL_PATH}.zip — running fixed gait only.")
        print("Train one with 17_train_bittle_terrain.py to fill the RL columns.\n")

    rows = []
    print("\n" + "=" * 78)
    print(f"{'difficulty':>10} | {'controller':>10} | {'distance(m)':>11} | "
          f"{'fall_rate':>9} | {'survival':>8}")
    print("-" * 78)

    for d in DIFFICULTIES:
        fx_dist, fx_fall, fx_surv = run_fixed_gait(d, N_EPISODES)
        print(f"{d:>10.2f} | {'fixed':>10} | {fx_dist:>11.3f} | "
              f"{fx_fall:>9.2f} | {fx_surv:>8.0f}")
        rows.append([d, "fixed", fx_dist, fx_fall, fx_surv])

        if model is not None:
            rl_dist, rl_fall, rl_surv = run_rl_policy(d, N_EPISODES, model, norm)
            print(f"{d:>10.2f} | {'RL':>10} | {rl_dist:>11.3f} | "
                  f"{rl_fall:>9.2f} | {rl_surv:>8.0f}")
            rows.append([d, "RL", rl_dist, rl_fall, rl_surv])
        print("-" * 78)

    # Save the raw numbers for the report.
    csv_path = DATA_DIR / "terrain_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["difficulty", "controller", "distance_m", "fall_rate", "survival_steps"])
        w.writerows(rows)
    print(f"\nSaved results → {csv_path}")

    # Plot distance vs difficulty if we have both controllers.
    if model is not None:
        _plot(rows)


def _plot(rows):
    import matplotlib.pyplot as plt
    fixed = [(r[0], r[2]) for r in rows if r[1] == "fixed"]
    rl    = [(r[0], r[2]) for r in rows if r[1] == "RL"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([d for d, _ in fixed], [v for _, v in fixed],
            "o-", color="C3", linewidth=2.5, label="Fixed gait (no learning)")
    ax.plot([d for d, _ in rl], [v for _, v in rl],
            "o-", color="C0", linewidth=2.5, label="RL policy (learned)")
    ax.set_xlabel("Terrain difficulty (0 = flat, 1 = roughest)", fontsize=12)
    ax.set_ylabel("Forward distance walked (m)", fontsize=12)
    ax.set_title("Why RL is needed: distance vs. terrain difficulty", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = ROOT / "data" / "terrain_benchmark.png"
    plt.savefig(out, dpi=150)
    print(f"Saved graph → {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
