"""
06_evaluate.py — Measure how good the trained policy is, with numbers.

Works with Ant runs (ant_v2, ant_v3) and dm_control quadruped (quad_v1).

Switch run with the ANT_RUN_NAME env var:
    (PowerShell)  $env:ANT_RUN_NAME="quad_v1"
    D:\\robot_venv\\Scripts\\python.exe notebooks\\06_evaluate.py

For Ant:      reports reward, episode length, and DISTANCE in metres.
For Quadruped: reports reward, episode length, and REWARD QUALITY (0–1).

WHY DIFFERENT METRICS?
Ant-v5 provides x_position in its info dict, so we can measure exact metres
travelled. dm_control quadruped does not expose position directly, but its
reward is already on a [0, 1] scale where 1.0 means "walking at target speed
while perfectly upright". So reward quality IS the performance metric.
"""

import os
import pathlib

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

RUN_NAME = os.environ.get("ANT_RUN_NAME", "ant_v3")
SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / RUN_NAME
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"

N_EVAL_EPISODES = 10

IS_QUADRUPED = RUN_NAME.startswith("quad_")

if IS_QUADRUPED:
    ENV_ID = "dm_control/quadruped-walk-v0"
    ENV_KWARGS = {}
else:
    ENV_ID = "Ant-v5"
    _ENV_CONFIGS = {
        "ant_v2": {"contact_cost_weight": 5e-4,  "healthy_z_range": (0.2,  1.0)},
        "ant_v3": {"contact_cost_weight": 0.005, "healthy_z_range": (0.28, 1.0)},
    }
    _cfg = _ENV_CONFIGS.get(RUN_NAME, _ENV_CONFIGS["ant_v3"])
    ENV_KWARGS = {
        "forward_reward_weight": 1.0,
        "ctrl_cost_weight": 0.05,
        "healthy_reward": 1.0,
        "max_episode_steps": 1000,
        **_cfg,
    }


def main():
    if not MODEL_PATH.with_suffix(".zip").exists():
        print(f"No model found at {MODEL_PATH}.zip — run the training script first.")
        return

    print(f"Evaluating run: {RUN_NAME}  ({ENV_ID})")
    model = PPO.load(str(MODEL_PATH))

    normalizer = None
    if VECNORM_PATH.exists():
        dummy = make_vec_env(ENV_ID, n_envs=1,
                             env_kwargs=ENV_KWARGS if ENV_KWARGS else {})
        normalizer = VecNormalize.load(str(VECNORM_PATH), dummy)
        normalizer.training = False
        print("Loaded normalisation statistics (evaluation mode).")
    else:
        print("No VecNormalize stats found — evaluating without normalisation.")

    env = gym.make(ENV_ID, **ENV_KWARGS)

    rewards, lengths, third_metric = [], [], []
    print(f"\nRunning {N_EVAL_EPISODES} episodes...\n")

    for ep in range(N_EVAL_EPISODES):
        obs, info = env.reset()
        start_x = info.get("x_position", None)
        total_reward, steps, last_x = 0.0, 0, start_x
        step_rewards = []

        while True:
            model_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_rewards.append(reward)
            steps += 1
            if not IS_QUADRUPED:
                last_x = info.get("x_position", last_x)
            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(steps)

        if IS_QUADRUPED:
            # dm_control reward per step is in [0, 1].
            # Mean per-step reward = average walking quality over the episode.
            quality = float(np.mean(step_rewards))
            third_metric.append(quality)
            print(f"  Episode {ep + 1:2d}:  reward {total_reward:7.2f}   "
                  f"length {steps:4d}   quality {quality:.3f}")
        else:
            distance = (last_x or 0.0) - (start_x or 0.0)
            third_metric.append(distance)
            print(f"  Episode {ep + 1:2d}:  reward {total_reward:7.1f}   "
                  f"length {steps:4d}   distance {distance:6.2f} m")

    env.close()

    print("\n" + "=" * 58)
    print(f"  Mean reward:   {np.mean(rewards):9.2f}  (std {np.std(rewards):.2f})")
    print(f"  Mean length:   {np.mean(lengths):9.1f}  steps")
    if IS_QUADRUPED:
        q = np.mean(third_metric)
        print(f"  Mean quality:  {q:9.3f}  (0=fallen/still, 1=walking perfectly)")
        print("=" * 58)
        print()
        print("  Reading it:")
        print("   quality < 0.3  → barely moving or unstable")
        print("   quality 0.3–0.7 → learning to walk, not yet fluent")
        print("   quality > 0.7  → walking well at target speed")
    else:
        d = np.mean(third_metric)
        print(f"  Mean distance: {d:9.2f}  m  (std {np.std(third_metric):.2f})")
        print("=" * 58)
        print()
        print("  Reading it:")
        print("   distance near 0  → standing still, not walking")
        print("   distance large + length near max → walking and stable")


if __name__ == "__main__":
    main()
