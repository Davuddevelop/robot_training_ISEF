"""
03_watch_trained_agent.py — Watch the trained Ant walk in a 3D window.

Run AFTER 05_improved_training.py has finished.

IMPORTANT: the model was trained on NORMALIZED observations. So before asking
the policy for an action, we must scale each raw observation using the SAME
statistics saved during training (vecnormalize.pkl). Skipping this makes a
good policy look broken. We load those stats and apply them every step.

Run with:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\03_watch_trained_agent.py
"""

import pathlib

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

RUN_NAME = "ant_v2"
SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / RUN_NAME
MODEL_PATH = SAVE_DIR / "ant_v2_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"

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
        print(f"Model not found at {MODEL_PATH}.zip — run 05_improved_training.py first.")
        return

    model = PPO.load(str(MODEL_PATH))

    # Load the saved normalization so we can scale observations the same way.
    # We host the stats on a tiny throwaway vec-env just to reuse normalize_obs().
    normalizer = None
    if VECNORM_PATH.exists():
        normalizer = VecNormalize.load(
            str(VECNORM_PATH),
            make_vec_env("Ant-v5", n_envs=1, env_kwargs=ENV_KWARGS),
        )
        normalizer.training = False
        print("Loaded normalization statistics.")
    else:
        print("WARNING: no vecnormalize.pkl found — movement may look broken.")

    print("Opening viewer. Close the window to stop.\n")
    env = gym.make("Ant-v5", render_mode="human", **ENV_KWARGS)
    obs, _ = env.reset()

    episode = 0
    total_reward = 0.0
    for _ in range(3000):
        # Scale the observation exactly as during training, then choose an action.
        model_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
        action, _ = model.predict(model_obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            episode += 1
            print(f"Episode {episode} ended — total reward: {total_reward:.1f}")
            obs, _ = env.reset()
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
