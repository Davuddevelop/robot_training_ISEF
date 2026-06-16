"""
06_evaluate.py — Measure how good the trained policy is, with numbers.

Watching is nice; science needs numbers. This runs the trained policy for
several episodes and reports, per episode and averaged:
    reward    — total score (higher = better)
    length    — steps survived (longer = more stable; max is the episode cap)
    distance  — how far forward it actually travelled, in metres

WHY distance matters: a high reward can be earned just by standing still and
collecting the per-step "alive" bonus. Distance tells you whether it is really
WALKING. Your real project measures distance in a fixed time — this is the
template for that metric.

It loads the saved VecNormalize statistics so observations are scaled exactly
as in training (feeding raw observations to a model trained on normalized ones
produces broken behaviour).

Run AFTER 05_improved_training.py:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\06_evaluate.py
"""

import pathlib

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

RUN_NAME = "ant_v2"
SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / RUN_NAME
MODEL_PATH = SAVE_DIR / "ant_v2_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"

N_EVAL_EPISODES = 10

ENV_KWARGS = {
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.05,
    "contact_cost_weight": 5e-4,
    "healthy_reward": 1.0,
    "healthy_z_range": (0.2, 1.0),
    "max_episode_steps": 1000,
}


def main():
    if not MODEL_PATH.with_suffix(".zip").exists():
        print(f"No model found at {MODEL_PATH}.zip — run 05_improved_training.py first.")
        return

    model = PPO.load(str(MODEL_PATH))

    # Load normalization stats (for scaling observations the trained way).
    normalizer = None
    if VECNORM_PATH.exists():
        normalizer = VecNormalize.load(
            str(VECNORM_PATH),
            make_vec_env("Ant-v5", n_envs=1, env_kwargs=ENV_KWARGS),
        )
        normalizer.training = False
        print("Loaded normalization statistics (evaluation mode).")
    else:
        print("No VecNormalize stats found — evaluating without normalization.")

    # Single env so we can read the info dict (which holds the x position).
    env = gym.make("Ant-v5", **ENV_KWARGS)

    rewards, lengths, distances = [], [], []
    print(f"Evaluating over {N_EVAL_EPISODES} episodes...\n")

    for ep in range(N_EVAL_EPISODES):
        obs, info = env.reset()
        start_x = info.get("x_position", 0.0)
        total_reward, steps, last_x = 0.0, 0, start_x

        while True:
            model_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            last_x = info.get("x_position", last_x)
            if terminated or truncated:
                break

        distance = last_x - start_x
        rewards.append(total_reward)
        lengths.append(steps)
        distances.append(distance)
        print(f"  Episode {ep + 1:2d}:  reward {total_reward:7.1f}   "
              f"length {steps:4d}   distance {distance:6.2f} m")

    env.close()

    print("\n" + "=" * 54)
    print(f"  Mean reward:    {np.mean(rewards):8.1f}  (std {np.std(rewards):.1f})")
    print(f"  Mean length:    {np.mean(lengths):8.1f}  steps")
    print(f"  Mean distance:  {np.mean(distances):8.2f}  m  (std {np.std(distances):.2f})")
    print("=" * 54)
    print()
    print("  Reading it:")
    print("   - distance near 0  -> it is standing still, not walking")
    print("   - distance large + length near max -> it is walking and stable")


if __name__ == "__main__":
    main()
