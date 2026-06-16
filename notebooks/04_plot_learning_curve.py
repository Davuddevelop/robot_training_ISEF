"""
04_plot_learning_curve.py — Turn training data into a graph you can read.

A scrolling reward number tells you almost nothing. A *learning curve* tells
you the whole story: did the agent improve, when did it plateau, and how
noisy was the learning. This is the single most important plot in your
project — later you will put several of these on the same axes to compare
your experimental conditions (no randomization vs. full randomization vs.
randomization + system ID). That comparison IS your science fair result.

This script reads the per-episode data that 02_first_training.py logged into
models/ant_first_run/monitor_logs/ and draws:
    X axis = training steps (how much experience the agent has had)
    Y axis = episode reward  (how good the agent is)

A working run looks like a noisy line that trends UPWARD and then flattens.

Run AFTER 02_first_training.py has finished:
    D:\\robot_venv\\Scripts\\python.exe notebooks\\04_plot_learning_curve.py
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

# SB3 helpers that read the monitor CSV files for us.
# load_results: reads every *.monitor.csv in a folder into one table.
# ts2xy: converts that table into x = cumulative timesteps, y = episode reward.
import os

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

# Switch run with env var, e.g.:  $env:ANT_RUN_NAME="ant_v3"
# Runs: ant_first_run, ant_improved, ant_v2, ant_v3
RUN_NAME = os.environ.get("ANT_RUN_NAME", "ant_v2")

RUN_DIR = pathlib.Path(__file__).parent.parent / "models" / RUN_NAME
MONITOR_DIR = RUN_DIR / "monitor_logs"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = DATA_DIR / f"{RUN_NAME}_learning_curve.png"


def moving_average(values, window):
    """
    Smooths a noisy line by averaging each point with its neighbours.

    Episode rewards bounce around a lot from one episode to the next, so the
    raw line is hard to read. A moving average reveals the underlying trend
    without hiding it. window = how many episodes to average together.
    """
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def main():
    if not MONITOR_DIR.exists():
        print(f"No training logs found at: {MONITOR_DIR}")
        print("Run 02_first_training.py first.")
        return

    # Read the raw per-episode data and convert it to (steps, reward) points.
    x, y = ts2xy(load_results(str(MONITOR_DIR)), "timesteps")

    if len(x) == 0:
        print("Logs found but no completed episodes recorded yet.")
        return

    print(f"Loaded {len(x)} episodes of training data.")
    print(f"First episode reward:  {y[0]:.1f}")
    print(f"Last episode reward:   {y[-1]:.1f}")
    print(f"Best episode reward:   {max(y):.1f}")

    # --- DRAW THE GRAPH ---
    plt.figure(figsize=(10, 6))

    # Faint raw data in the background, so we don't hide the real noise.
    plt.plot(x, y, color="lightgray", linewidth=0.8, label="raw (every episode)")

    # Bold smoothed trend line on top — the part you actually read.
    window = min(50, len(y))  # average over 50 episodes (or fewer if we have fewer)
    if len(y) >= window:
        y_smooth = moving_average(y, window)
        x_smooth = x[window - 1:]  # line up the x values with the smoothed y values
        plt.plot(x_smooth, y_smooth, color="C0", linewidth=2.5,
                 label=f"smoothed (avg of {window} episodes)")

    plt.xlabel("Training steps (experience collected)")
    plt.ylabel("Episode reward (how good the agent is)")
    plt.title("Ant — Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_PNG, dpi=120)
    print(f"\nSaved graph to: {OUTPUT_PNG}")
    print("Open it. An upward trend that flattens out means training worked.")

    # Also pop it up on screen. Wrapped in try/except so it still works on a
    # machine without a display (the PNG is saved regardless).
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
