"""
19_preview_terrain.py — Look at the rubble/terrain field WITHOUT needing a
                        trained policy. Just holds the robot's neutral pose
                        so you can inspect the ground.

Opening scene_rough.xml directly in the MuJoCo standalone viewer shows the
raw file — the debris chunks are parked far underground until our Python
code (bittle_env.py) places them. This script runs that placement so you
can actually see the rubble field.

Run (defaults to difficulty 1.0 — full rubble):
    D:\\robot_venv\\Scripts\\python.exe notebooks\\19_preview_terrain.py

Try a different difficulty:
    (PowerShell)  $env:PREVIEW_DIFFICULTY="0.5"
    D:\\robot_venv\\Scripts\\python.exe notebooks\\19_preview_terrain.py
"""

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.sim.bittle_env import BittleEnv

DIFFICULTY = float(os.environ.get("PREVIEW_DIFFICULTY", "1.0"))
SEED = int(os.environ.get("PREVIEW_SEED", "0"))


def main():
    print(f"Previewing terrain at difficulty {DIFFICULTY} (seed {SEED}).")
    print("Robot holds its neutral standing pose. Close the window to stop.\n")

    env = BittleEnv(render_mode="human", terrain=True, terrain_difficulty=DIFFICULTY)
    env.reset(seed=SEED)

    # Hold neutral pose (action = 0 offset) so the terrain is easy to inspect —
    # no walking, just standing still on the rubble so you can look around.
    neutral_action = [0.0] * 8
    for _ in range(100000):
        obs, reward, terminated, truncated, info = env.step(neutral_action)
        if terminated or truncated:
            env.reset(seed=SEED)
        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()
