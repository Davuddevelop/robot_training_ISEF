"""
20_deploy_to_robot.py — Run the trained Bittle policy on the REAL hardware.

READ THIS BEFORE RUNNING:
  1. Train first: run 11_train_bittle.py until it finishes.
  2. Connect the NyBoard to your laptop via the USB cable.
  3. Open Device Manager → Ports (COM & LPT) and note the COM port number.
     It will look like "Silicon Labs CP210x" or "USB Serial Port (COM3)".
  4. Set PORT below to match (e.g. "COM3", "COM7").
  5. Place the Bittle on a flat floor with at least 1 metre of clear space
     ahead of it. Make sure nothing will fall on it.
  6. Run this script. The robot will stand still for 2 seconds (connecting),
     then walk for N_STEPS × 0.02 s seconds.
  7. Press Ctrl+C at any time to stop immediately. The robot returns to the
     standing pose before the program exits.

What to expect on first run:
  - The robot will probably not walk well. This is normal.
  - Watch the tilt: if it falls, the safety monitor stops it automatically.
  - Log what you see. This data informs Phase 4 (system identification).

Run:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\20_deploy_to_robot.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.robot.control_loop import RealRobotController

# ── CONFIGURE THESE BEFORE RUNNING ───────────────────────────────────────────
PORT     = "COM3"      # ← change to your actual COM port
RUN_NAME = "bittle_v1" # which trained model to deploy
N_STEPS  = 500         # 500 × 0.02 s = 10 seconds of walking
# ─────────────────────────────────────────────────────────────────────────────

ROOT       = pathlib.Path(__file__).parent.parent
SAVE_DIR   = ROOT / "models" / RUN_NAME
MODEL_PATH = SAVE_DIR / f"{RUN_NAME}_model"
VN_PATH    = SAVE_DIR / "vecnormalize.pkl"


def main():
    if not MODEL_PATH.with_suffix(".zip").exists():
        print(f"ERROR: No model found at {MODEL_PATH}.zip")
        print("       Run 11_train_bittle.py first, then try again.")
        return

    if not VN_PATH.exists():
        print(f"ERROR: No normalization stats at {VN_PATH}")
        print("       The model file exists but vecnormalize.pkl is missing.")
        print("       Re-run 11_train_bittle.py — both files must exist together.")
        return

    print("=" * 60)
    print("  BITTLE REAL-ROBOT DEPLOYMENT  (Phase 3)")
    print(f"  Model : {RUN_NAME}")
    print(f"  Port  : {PORT}")
    print(f"  Steps : {N_STEPS}  ({N_STEPS * 0.02:.0f} s)")
    print("=" * 60)

    controller = RealRobotController(
        model_path=MODEL_PATH,
        vecnorm_path=VN_PATH,
        port=PORT,
    )

    try:
        controller.run(n_steps=N_STEPS)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
