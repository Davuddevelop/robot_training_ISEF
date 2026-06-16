"""
03_watch_trained_agent.py — Watch the trained Ant walk.

Run AFTER 02_first_training.py has finished.
Opens a 3D viewer window showing the agent using its trained policy.

Run with:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\03_watch_trained_agent.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
import pathlib
import time

# Switch between models by uncommenting the one you want to watch.
# MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "ant_first_run" / "ant_300k"
# MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "ant_improved" / "ant_1M"
MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "ant_v2" / "ant_v2_1M"

if not MODEL_PATH.with_suffix(".zip").exists():
    print("Model not found. Run 02_first_training.py first.")
    raise SystemExit(1)

print("Loading trained model...")
model = PPO.load(str(MODEL_PATH))

print("Opening viewer. Close the window to stop.\n")

env = gym.make("Ant-v5", render_mode="human")
obs, _ = env.reset()

episode = 0
total_reward = 0.0

for step in range(2000):
    # The policy decides what to do based on the observation.
    # deterministic=True means it picks the best action, not a random one.
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        episode += 1
        print(f"Episode {episode} ended — total reward: {total_reward:.1f}")
        obs, _ = env.reset()
        total_reward = 0.0

env.close()
