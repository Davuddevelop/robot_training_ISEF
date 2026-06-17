"""
14_compare_runs.py — Plot all training conditions on one graph.

This is the graph that goes on your ISEF poster.
It shows how each experimental condition performed during training,
so a judge can see at a glance which condition produced the best walker.

Conditions plotted (whichever folders exist under models/):
    bittle_v1  — no domain randomization
    bittle_v2  — full domain randomization
    bittle_v3  — domain randomization + system identification  (Phase 4)

Run any time after training at least one condition:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\14_compare_runs.py
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Each entry: (folder_name, label_for_legend, colour)
CONDITIONS = [
    ("bittle_v1", "No domain randomization",                "C0"),
    ("bittle_v2", "Full domain randomization",              "C1"),
    ("bittle_v3", "Domain rand + system identification",    "C2"),
]

SMOOTH_WINDOW = 50   # episodes to average for the smoothed line


def moving_average(values, window):
    if len(values) < window:
        return values, np.arange(len(values))
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode="valid")
    return smoothed


def main():
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_any = False

    for run_name, label, colour in CONDITIONS:
        monitor_dir = ROOT / "models" / run_name / "monitor_logs"
        if not monitor_dir.exists():
            print(f"  Skipping {run_name} — no monitor logs found.")
            continue

        x, y = ts2xy(load_results(str(monitor_dir)), "timesteps")
        if len(x) == 0:
            print(f"  Skipping {run_name} — logs empty.")
            continue

        print(f"  {run_name}: {len(x)} episodes, "
              f"final smoothed reward ≈ {np.mean(y[-SMOOTH_WINDOW:]):.0f}")

        # Raw episodes (faint)
        ax.plot(x, y, color=colour, alpha=0.15, linewidth=0.6)

        # Smoothed trend (bold)
        if len(y) >= SMOOTH_WINDOW:
            y_smooth = moving_average(y, SMOOTH_WINDOW)
            x_smooth = x[SMOOTH_WINDOW - 1:]
            ax.plot(x_smooth, y_smooth, color=colour, linewidth=2.5, label=label)
        else:
            ax.plot(x, y, color=colour, linewidth=2.5, label=label)

        plotted_any = True

    if not plotted_any:
        print("\nNo training data found. Run 11_train_bittle.py first.")
        return

    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel("Episode reward", fontsize=12)
    ax.set_title("Bittle Sim-to-Real: Learning Curves by Condition", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = DATA_DIR / "condition_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved comparison graph → {out}")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
