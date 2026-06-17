"""
15_export_onnx.py — Export the trained policy to ONNX format.

ONNX is a universal format that can run on the Raspberry Pi Zero 2 W
without needing PyTorch or Stable-Baselines3 installed.  It is much
lighter and runs faster on low-power hardware.

The exported file runs with onnxruntime, which is already in requirements.txt.

Run AFTER training:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\15_export_onnx.py

What it exports:
    models/bittle_v1/bittle_v1_policy.onnx   (the neural network)
    models/bittle_v1/vecnorm_stats.npz        (mean + std for observation scaling)

The vecnorm_stats.npz must travel with the .onnx file — it contains the
normalization values the policy was trained with.  Without them the inputs
are on the wrong scale and the policy behaves randomly.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.sim.bittle_env import BittleEnv
from src.sim.config import OBS_DIM

RUN_NAME = "bittle_v1"   # change to "best_model" to export the best checkpoint

ROOT        = pathlib.Path(__file__).parent.parent
SAVE_DIR    = ROOT / "models" / RUN_NAME
MODEL_PATH  = SAVE_DIR / f"{RUN_NAME}_model"
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"
ONNX_PATH   = SAVE_DIR / f"{RUN_NAME}_policy.onnx"
STATS_PATH  = SAVE_DIR / "vecnorm_stats.npz"


def main():
    if not MODEL_PATH.with_suffix(".zip").exists():
        print(f"No model at {MODEL_PATH}.zip — run 11_train_bittle.py first.")
        return

    print(f"Loading model from {MODEL_PATH}.zip …")
    model = PPO.load(str(MODEL_PATH))

    # Load VecNormalize to extract the obs mean and std.
    dummy = DummyVecEnv([lambda: BittleEnv(domain_rand=False)])
    norm  = VecNormalize.load(str(VECNORM_PATH), dummy)
    norm.training = False

    # Export the policy network to ONNX.
    # SB3's extract_policy() works for MlpPolicy; the policy takes a single
    # observation vector and outputs (mean_action, log_std, value).
    # We export only the actor (mean_action) for deployment.
    print("Exporting to ONNX …")
    model.policy.to("cpu")

    import torch
    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    torch.onnx.export(
        model.policy,
        dummy_obs,
        str(ONNX_PATH),
        input_names=["observation"],
        output_names=["action", "log_std", "value"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    print(f"  Policy → {ONNX_PATH}")

    # Save normalization stats separately (can't embed in ONNX).
    obs_mean = norm.obs_rms.mean.astype(np.float32)
    obs_std  = np.sqrt(norm.obs_rms.var + norm.epsilon).astype(np.float32)
    np.savez(str(STATS_PATH), obs_mean=obs_mean, obs_std=obs_std)
    print(f"  Norm stats → {STATS_PATH}")

    # Quick sanity check: run the ONNX model and compare to PyTorch output.
    print("\nSanity check …")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ONNX_PATH))

        test_raw_obs = np.zeros(OBS_DIM, dtype=np.float32)
        test_raw_obs[2] = -1.0  # upright gravity vector

        # Normalize using saved stats
        test_norm_obs = (test_raw_obs - obs_mean) / obs_std
        test_norm_obs = np.clip(test_norm_obs, -10.0, 10.0)

        # ONNX inference
        onnx_out = sess.run(None, {"observation": test_norm_obs.reshape(1, -1)})
        onnx_action = onnx_out[0][0]

        # PyTorch inference
        import torch
        with torch.no_grad():
            pt_out = model.policy(
                torch.tensor(test_norm_obs).unsqueeze(0)
            )
        pt_action = pt_out[0].numpy()[0]

        max_diff = np.max(np.abs(onnx_action - pt_action))
        print(f"  Max difference between ONNX and PyTorch outputs: {max_diff:.6f}")
        if max_diff < 1e-4:
            print("  ✓ ONNX export verified — outputs match.")
        else:
            print("  ✗ Warning: outputs differ more than expected. Check export.")

    except ImportError:
        print("  (onnxruntime not installed — skipping sanity check)")
        print("  Install with: D:\\robot_venv\\Scripts\\pip.exe install onnxruntime")

    norm.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
