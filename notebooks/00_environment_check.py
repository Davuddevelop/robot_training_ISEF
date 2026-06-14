"""
Environment check — run this on your laptop to verify every dependency
is installed and your GPU is visible to PyTorch.

Run with:  python notebooks/00_environment_check.py
"""

import sys


def check(label, fn):
    try:
        result = fn()
        print(f"  OK  {label}: {result}")
        return True
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        return False


print("\n=== Robot Training Environment Check ===\n")

# 1. Python version
check("Python", lambda: sys.version.split()[0])

# 2. NumPy
check("NumPy", lambda: __import__("numpy").__version__)

# 3. PyTorch version
check("PyTorch", lambda: __import__("torch").__version__)

# 4. CUDA (GPU) available — critical for fast training
def cuda_check():
    import torch
    if torch.cuda.is_available():
        return f"CUDA {torch.version.cuda} — GPU: {torch.cuda.get_device_name(0)}"
    return "NOT AVAILABLE (training will be slow — install CUDA build of PyTorch)"

check("CUDA", cuda_check)

# 5. MuJoCo
check("MuJoCo", lambda: __import__("mujoco").__version__)

# 6. Gymnasium
check("Gymnasium", lambda: __import__("gymnasium").__version__)

# 7. Stable-Baselines3
check("Stable-Baselines3", lambda: __import__("stable_baselines3").__version__)

# 8. ONNX Runtime
check("ONNX Runtime", lambda: __import__("onnxruntime").__version__)

print("\nDone. Every line above should say OK.")
print("If any say FAIL, install the missing package with:  pip install <package-name>\n")
