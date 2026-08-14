"""
21_terrain_achievability.py — Is the hardest terrain physically walkable AT ALL?

WHY THIS EXISTS
    Before spending hours of training trying to master terrain difficulty 1.0, we
    should know whether ANY controller can cross it. If a competent hand-tuned gait
    cannot, then the terrain is the blocker and no amount of training will fix it --
    we would be optimising toward an impossible target.

    A fixed open-loop gait is a good probe precisely because it does NOT adapt: it
    is a known, repeatable walking pattern. If the ground defeats it everywhere,
    that tells us about the ground.

THE NUMBER THAT MATTERS
    Each episode starts on a deliberately flattened spawn patch (~0.63 m across) so
    the robot doesn't topple before it can take a step. That means it has roughly
    0.31 m of easy ground ahead of it before meeting the first bump.

    A result at or below ~0.31 m therefore does NOT mean "walked badly on rough
    terrain" -- it means "never reached the rough terrain at all". That distinction
    is invisible in a mean-distance number, which is why this script reports how
    many episodes actually cleared it.

RUN:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\21_terrain_achievability.py

NOTE ON COMPARING NUMBERS ACROSS MACHINES
    MuJoCo contact simulation is chaotic: the same seed on a different MuJoCo
    version or OS diverges into a visibly different trajectory. Numbers from this
    script are only comparable to other numbers produced on the SAME machine with
    the SAME MuJoCo version. Both are printed below so results stay traceable.
"""

import pathlib
import platform
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.sim.bittle_env import BittleEnv
from src.sim.scripted_gait import ScriptedTrotGait
from src.sim.config import (
    NEUTRAL_POSE, ACTION_LIMIT, CONTROL_TIMESTEP, EPISODE_LENGTH_STEPS,
)

NEUTRAL = np.array(NEUTRAL_POSE, dtype=np.float64)

DIFFICULTIES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
N_EPISODES = 10
FLAT_PATCH_M = 0.31

# Two gaits: the library defaults, and the parameters an earlier sweep picked as
# best. They are NOT the same controller and they do not agree -- keeping both
# honest here is the point.
GAITS = {
    "default": {},
    "sweep": dict(freq_hz=3.5, shoulder_amp=0.30, knee_amp=0.10),
}


def run_gait(gait_kwargs, difficulty, n_episodes=N_EPISODES):
    """Run one open-loop gait over one difficulty. Returns a stats dict."""
    env = BittleEnv(domain_rand=False, terrain=True, terrain_difficulty=difficulty)
    gait = ScriptedTrotGait(**gait_kwargs)

    distances, falls = [], 0
    for ep in range(n_episodes):
        # Same seeds 16_terrain_benchmark.py uses, so the two are comparable.
        env.reset(seed=1000 + ep)
        gait.reset()
        start_y = env._mj_data.qpos[1]
        t, steps = 0.0, 0
        while True:
            angles = gait.step(t)
            action = np.clip((angles - NEUTRAL) / ACTION_LIMIT, -1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            t += CONTROL_TIMESTEP
            if terminated or truncated:
                break
        distances.append(float(env._mj_data.qpos[1] - start_y))
        if steps < EPISODE_LENGTH_STEPS:
            falls += 1

    env.close()
    return {
        "mean": float(np.mean(distances)),
        "fall_rate": falls / n_episodes,
        "cleared": sum(1 for d in distances if d > FLAT_PATCH_M),
        "best": float(np.max(distances)),
    }


def main():
    print("=" * 72)
    print("  TERRAIN ACHIEVABILITY PROBE — can a fixed gait cross this ground?")
    print("=" * 72)
    print(f"  MuJoCo {mujoco.__version__} on {platform.system()} "
          f"({platform.machine()})   <- numbers only comparable within this setup")
    print(f"  {N_EPISODES} episodes per cell, seeds 1000..{1000 + N_EPISODES - 1}")
    print(f"  Flat spawn patch reaches ~{FLAT_PATCH_M} m — at or below that, the")
    print("  robot never actually met a bump.\n")

    print(f"{'diff':>5} | {'gait':>8} | {'dist(m)':>8} | {'fall':>5} | "
          f"{'cleared':>9} | {'best(m)':>8}")
    print("-" * 62)

    for d in DIFFICULTIES:
        for name, kwargs in GAITS.items():
            r = run_gait(kwargs, d)
            flag = "  <-- never left flat ground" if r["cleared"] == 0 else ""
            print(f"{d:>5.2f} | {name:>8} | {r['mean']:>8.3f} | "
                  f"{r['fall_rate']:>5.2f} | {r['cleared']:>4}/{N_EPISODES:<4} | "
                  f"{r['best']:>8.3f}{flag}")
        print("-" * 62)

    print("\nHOW TO READ THIS:")
    print("  * 'cleared 0/10' at a difficulty means no episode reached rough ground.")
    print("    That difficulty is beyond this controller, full stop.")
    print("  * A high fall rate WITH a good 'best' means the terrain is survivable")
    print("    but unreliable — worth training against.")
    print("  * A high fall rate AND a low 'best' means the ground itself is the")
    print("    blocker, and the difficulty scale should be recalibrated rather than")
    print("    trained harder against.")


if __name__ == "__main__":
    main()
