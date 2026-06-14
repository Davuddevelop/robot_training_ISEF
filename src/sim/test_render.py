"""
MuJoCo smoke test — loads the Bittle model and renders one frame.
Run this after following GET_BITTLE_MODEL.md to confirm the model works.

Run with:  python src/sim/test_render.py
"""

import mujoco
import mujoco.viewer
import pathlib

MODEL_PATH = pathlib.Path(__file__).parent / "bittle_model" / "bittle.xml"

if not MODEL_PATH.exists():
    print(f"\nModel file not found: {MODEL_PATH}")
    print("Follow src/sim/GET_BITTLE_MODEL.md first.\n")
    raise SystemExit(1)

print(f"Loading model from: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data  = mujoco.MjData(model)

print(f"  Bodies    : {model.nbody}")
print(f"  Joints    : {model.njnt}")
print(f"  Actuators : {model.nu}")
print(f"\nOpening viewer — close the window to exit.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(500):
        mujoco.mj_step(model, data)
        viewer.sync()

print("Done.")
