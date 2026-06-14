# Getting the Bittle MuJoCo Model

The Bittle simulation uses a community-made MuJoCo XML model of the robot.
We do not include it in this repo because it is someone else's work.

## Steps

1. Search GitHub for "Petoi Bittle MuJoCo" or "Bittle MJCF"
   — the Petoi community and academic groups have published these.

2. Download the model folder (usually contains a `.xml` file and mesh files
   in a subfolder like `assets/` or `meshes/`).

3. Place it here:
   ```
   src/sim/bittle_model/
       bittle.xml        ← the main model file
       assets/           ← mesh and texture files
   ```

4. Test it loads by running (Phase 2 — after foundations are done):
   ```
   python src/sim/test_render.py
   ```

## What the model file is

A MuJoCo XML file (`.xml`) is a text description of the robot:
- Its body parts (links), their sizes and masses
- The joints between them and their limits
- The actuators (motors) driving each joint

You do not need to write this file — we use an existing one and refine it
for system identification later (Phase 4).
