"""
12_watch_bittle.py — Watch the trained Bittle walk in a 3D window. (Phase 2)

Run AFTER 11_train_bittle.py.

Like the Ant viewer, this loads the saved VecNormalize statistics and scales
each observation the SAME way it was scaled during training — otherwise a good
policy looks broken.

Run:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\12_watch_bittle.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.sim.bittle_env import BittleEnv

RUN_NAME = "bittle_v1"
ROOT = pathlib.Path(__file__).parent.parent
SAVE_DIR = ROOT / "models" / RUN_NAME
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"


def main():
    if not MODEL_PATH.with_suffix(".zip").exists():
        print(f"No model at {MODEL_PATH}.zip — run 11_train_bittle.py first.")
        return

    model = PPO.load(str(MODEL_PATH))

    # The viewer env renders; the normalizer only scales observations for us.
    view_env = BittleEnv(render_mode="human", domain_rand=False)
    norm = VecNormalize.load(str(VECNORM_PATH), DummyVecEnv([lambda: BittleEnv(domain_rand=False)]))
    norm.training = False

    print("Opening viewer. Close the window to stop.\n")
    obs, _ = view_env.reset()
    episode, total_r = 0, 0.0

    for _ in range(5000):
        norm_obs = norm.normalize_obs(obs)         # scale exactly as in training
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = view_env.step(action)
        total_r += reward
        if terminated or truncated:
            episode += 1
            print(f"Episode {episode}: reward {total_r:.1f}, distance {info['forward_position']:.3f} m")
            obs, _ = view_env.reset()
            total_r = 0.0

    view_env.close()


if __name__ == "__main__":
    main()
