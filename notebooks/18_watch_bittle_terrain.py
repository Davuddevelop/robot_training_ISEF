"""
18_watch_bittle_terrain.py — Watch the trained terrain policy walk over
                              uneven ground in a 3D window.

Run AFTER 17_train_bittle_terrain.py.

Same idea as 12_watch_bittle.py, but loads the ROUGH scene and lets you pick
which difficulty to watch.

Run (defaults to difficulty 1.0 — the hardest terrain):
    D:\\robot_venv\\Scripts\\python.exe notebooks\\18_watch_bittle_terrain.py

Watch an easier level instead:
    (PowerShell)  $env:WATCH_DIFFICULTY="0.3"
    D:\\robot_venv\\Scripts\\python.exe notebooks\\18_watch_bittle_terrain.py
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.sim.bittle_env import BittleEnv

RUN_NAME = os.environ.get("BITTLE_RUN_NAME", "bittle_terrain")
DIFFICULTY = float(os.environ.get("WATCH_DIFFICULTY", "1.0"))

ROOT = pathlib.Path(__file__).parent.parent
SAVE_DIR = ROOT / "models" / RUN_NAME
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"

# Prefer the best-by-distance checkpoint if it exists; falls back to final model.
BEST_MODEL_PATH = SAVE_DIR / "best_model"
BEST_VECNORM_PATH = SAVE_DIR / "best_vecnormalize.pkl"


def main():
    model_path, vecnorm_path = MODEL_PATH, VECNORM_PATH
    if BEST_MODEL_PATH.with_suffix(".zip").exists():
        model_path, vecnorm_path = BEST_MODEL_PATH, BEST_VECNORM_PATH
        print("Using best_model (highest distance seen during training).")

    if not model_path.with_suffix(".zip").exists():
        print(f"No model at {model_path}.zip — run 17_train_bittle_terrain.py first.")
        return

    model = PPO.load(str(model_path))

    view_env = BittleEnv(render_mode="human", terrain=True, terrain_difficulty=DIFFICULTY)
    norm = VecNormalize.load(str(vecnorm_path),
                              DummyVecEnv([lambda: BittleEnv(terrain=True,
                                                              terrain_difficulty=DIFFICULTY)]))
    norm.training = False

    print(f"\nWatching '{RUN_NAME}' at terrain difficulty {DIFFICULTY}.")
    print("Opening viewer. Close the window to stop.\n")

    obs, _ = view_env.reset()
    episode, total_r = 0, 0.0

    for _ in range(5000):
        norm_obs = norm.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = view_env.step(action)
        total_r += reward
        if terminated or truncated:
            episode += 1
            print(f"Episode {episode}: reward {total_r:.1f}, "
                  f"distance {info['forward_position']:.3f} m, "
                  f"{'FELL' if terminated else 'survived full episode'}")
            obs, _ = view_env.reset()
            total_r = 0.0

    view_env.close()


if __name__ == "__main__":
    main()
