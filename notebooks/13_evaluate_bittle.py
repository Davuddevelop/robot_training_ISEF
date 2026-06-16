"""
13_evaluate_bittle.py — Measure the trained Bittle with numbers. (Phase 2)

Reports, per episode and averaged:
    reward    — total score
    length    — control steps survived (max = EPISODE_LENGTH_STEPS)
    distance  — how far FORWARD it travelled, in metres  <-- the real metric

Distance is what proves walking: a high reward can come from just standing and
collecting the alive bonus. Your science-fair metric is distance in fixed time.

Run AFTER 11_train_bittle.py:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\13_evaluate_bittle.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.sim.bittle_env import BittleEnv

RUN_NAME = "bittle_v1"
ROOT = pathlib.Path(__file__).parent.parent
SAVE_DIR = ROOT / "models" / RUN_NAME
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"

N_EVAL_EPISODES = 10


def main():
    if not MODEL_PATH.with_suffix(".zip").exists():
        print(f"No model at {MODEL_PATH}.zip — run 11_train_bittle.py first.")
        return

    model = PPO.load(str(MODEL_PATH))

    # Rebuild the same env stack and load the saved normalization stats.
    venv = DummyVecEnv([lambda: BittleEnv(domain_rand=False)])
    norm = VecNormalize.load(str(VECNORM_PATH), venv)
    norm.training = False
    norm.norm_reward = False   # we want RAW reward/metrics when evaluating

    rewards, lengths, distances = [], [], []
    print(f"Evaluating {N_EVAL_EPISODES} episodes...\n")

    for ep in range(N_EVAL_EPISODES):
        obs = norm.reset()
        start_y, last_y = None, None
        total_r, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = norm.step(action)
            info = infos[0]
            if start_y is None:
                start_y = info["forward_position"]
            last_y = info["forward_position"]
            total_r += float(norm.get_original_reward()[0])
            steps += 1
            if done[0]:
                break

        distance = last_y - start_y
        rewards.append(total_r); lengths.append(steps); distances.append(distance)
        print(f"  Episode {ep+1:2d}:  reward {total_r:7.1f}   "
              f"length {steps:4d}   distance {distance:6.3f} m")

    print("\n" + "=" * 54)
    print(f"  Mean reward:    {np.mean(rewards):8.1f}")
    print(f"  Mean length:    {np.mean(lengths):8.1f}  steps")
    print(f"  Mean distance:  {np.mean(distances):8.3f}  m  (std {np.std(distances):.3f})")
    print("=" * 54)
    print("\n  distance near 0 -> standing still;  large + full length -> walking.")


if __name__ == "__main__":
    main()
