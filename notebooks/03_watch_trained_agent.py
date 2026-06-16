"""
03_watch_trained_agent.py — Watch the trained agent in a 3D viewer window.

Works with both Ant runs (ant_v2, ant_v3) and dm_control quadruped (quad_v1).
Switch between them with the ANT_RUN_NAME environment variable:

    (PowerShell)  $env:ANT_RUN_NAME="ant_v3"
    D:\\robot_venv\\Scripts\\python.exe notebooks\\03_watch_trained_agent.py

    (PowerShell)  $env:ANT_RUN_NAME="quad_v1"
    D:\\robot_venv\\Scripts\\python.exe notebooks\\03_watch_trained_agent.py

IMPORTANT: the model was trained on NORMALIZED observations. Before passing
an observation to the policy, we MUST scale it using the same statistics that
were saved during training (vecnormalize.pkl). Skipping this makes a trained
policy look completely broken.
"""

import os
import pathlib

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

RUN_NAME = os.environ.get("ANT_RUN_NAME", "ant_v3")
SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / RUN_NAME
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"

# Detect which environment family this run belongs to.
IS_QUADRUPED = RUN_NAME.startswith("quad_")

if IS_QUADRUPED:
    # dm_control quadruped — no extra kwargs needed, task handles everything.
    ENV_ID = "dm_control/quadruped-walk-v0"
    ENV_KWARGS = {}
else:
    # MuJoCo Ant — kwargs must exactly match what the model was trained with.
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
        print(f"Model not found at {MODEL_PATH}.zip")
        print("Run the matching training script first.")
        return

    print(f"Loading model: {RUN_NAME}")
    model = PPO.load(str(MODEL_PATH))

    # Load the saved normalisation statistics.
    # We spin up a tiny throw-away vec-env just to host the stats object —
    # we only use it to call normalizer.normalize_obs() on each step.
    normalizer = None
    if VECNORM_PATH.exists():
        dummy = make_vec_env(ENV_ID, n_envs=1,
                             env_kwargs=ENV_KWARGS if ENV_KWARGS else {})
        normalizer = VecNormalize.load(str(VECNORM_PATH), dummy)
        normalizer.training = False
        print("Loaded normalisation statistics.")
    else:
        print("WARNING: no vecnormalize.pkl found — movement may look broken.")

    print(f"Opening viewer for {ENV_ID}. Close the window to stop.\n")

    render_kwargs = {"render_mode": "human"}
    if ENV_KWARGS:
        render_kwargs.update(ENV_KWARGS)

    env = gym.make(ENV_ID, **render_kwargs)
    obs, _ = env.reset()

    episode = 0
    total_reward = 0.0

    for _ in range(5000):
        model_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
        action, _ = model.predict(model_obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            episode += 1
            print(f"Episode {episode} ended — total reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
