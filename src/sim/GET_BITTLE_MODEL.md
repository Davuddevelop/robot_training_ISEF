# Getting the Bittle MuJoCo Model

The Bittle simulation uses a community-made MuJoCo model of the robot.
We do not include it in this repo because it is someone else's work.

## The two repos you need

**Option A — URDF only (needs conversion):**
https://github.com/AIWintermuteAI/Bittle_URDF
Contains URDF files and mesh (.obj) files for Bittle.
URDF is a different robot description format — needs converting to MuJoCo XML.

**Option B — MuJoCo XML (ready to use):**
https://github.com/gravesreid/mujoco_mpc_bittle
Contains a MuJoCo XML model of Bittle (converted from URDF).
This is the one to use — it's already in the right format.

## Exact steps (do this with your mentor)

1. Go to https://github.com/gravesreid/mujoco_mpc_bittle
2. Look inside the repo for a folder named `bittle/` or `mjpc/tasks/bittle/`
   It should contain a `.xml` file and a folder of mesh files.
3. Download those files.
4. Place them here in YOUR project:
   ```
   src/sim/bittle_model/
       bittle.xml        ← the main MuJoCo XML model file
       assets/           ← mesh (.stl or .obj) files for the 3D body parts
   ```
5. Test it loads:
   ```
   D:\robot_venv\Scripts\python.exe src/sim/test_render.py
   ```
   A 3D window should open showing the Bittle robot standing.

## What is a MuJoCo XML model?

It is a text file describing the robot in full detail:
- Every body part (torso, 4 thighs, 4 shins) with its mass and size
- Every joint connecting them, with angle limits
- Every actuator (motor) and its force limits
- The ground plane and gravity

When you open it in a text editor, it looks like HTML with robot-specific tags.
You do not need to write this file from scratch — you refine it for system
identification in Phase 4 (adjusting mass, motor gains to match the real robot).

## After you have the model

Run `test_render.py` to confirm it works, then come back here and we will
wire it into `bittle_env.py` to replace the stubs.
