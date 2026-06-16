"""
10_view_bittle.py — Look at the real Bittle model in 3D, before any training.

This is the FIRST Bittle step, and it follows your "measure/verify before build"
rule: we open the actual robot model in a 3D window and make it hold its neutral
standing pose. No reinforcement learning yet — we are just confirming the body
is correct and behaves sensibly.

What you should see:
  - A small four-legged robot (NOT a spider) standing on a checkered floor.
  - Legs pointing DOWN underneath the body — the Bittle shape.
  - It holds a stable crouch/stand and does not fall through the floor.

You can drag with the mouse to orbit the camera and inspect it.

Run:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\10_view_bittle.py
"""

import pathlib
import time

import mujoco
import mujoco.viewer

# Path to OUR scene file (community body + our floor/actuators/sensors).
SCENE = pathlib.Path(__file__).parent.parent / "src" / "sim" / "bittle_model" / "scene.xml"

# The neutral standing angle every leg joint is held at (radians).
HOME_ANGLE = 0.56


def main():
    if not SCENE.exists():
        print(f"Scene not found at {SCENE}")
        print("Did you run `git pull`?")
        return

    # Load the model and create the data (the live simulation state).
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    print(f"Loaded Bittle: {model.nu} actuators (servos), {model.nq} position coords.")
    print("Opening 3D viewer. Drag to orbit; close the window to stop.\n")

    # Start from the "home" keyframe (index 0) — the neutral standing pose.
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Command every servo to hold the neutral angle, so it stands still.
    data.ctrl[:] = HOME_ANGLE

    # launch_passive opens a window and lets US control the loop, so we can
    # keep feeding the servo targets each step.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            # Sleep so the simulation runs at roughly real time (not warp speed).
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()
