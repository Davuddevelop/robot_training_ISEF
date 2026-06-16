# Training Pipeline — Run Guide & Design Notes

This folder contains the **simulation learning pipeline**, built on the
built-in MuJoCo `Ant-v5` quadruped. Ant is a stand-in we use to build and
validate the whole workflow *before* the real Bittle model is wired in. The
algorithm (PPO), the library (Stable-Baselines3), and every design idea here
carry over directly to Bittle.

## The scripts, in order

| Script | What it does |
|---|---|
| `00_environment_check.py` | Confirms every package + the GPU are installed. |
| `02_first_training.py` | The very first demo run (default settings). Kept for history. |
| `05_improved_training.py` | Tuned + normalized run (ant_v2). Ant learned to crawl. |
| `07_upright_training.py` | **Current best script.** Adds upright reward fixes → ant_v3. |
| `04_plot_learning_curve.py` | Draws the learning curve (reward vs. steps) → saves a PNG. |
| `06_evaluate.py` | Reports numbers: mean reward, episode length, **forward distance**. |
| `03_watch_trained_agent.py` | Opens a 3D window to watch the trained policy. |

## How to run the upright training (ant_v3)

```
D:\robot_venv\Scripts\python.exe notebooks\07_upright_training.py
```

Then evaluate and watch:
```
(PowerShell)
$env:ANT_RUN_NAME="ant_v3"
D:\robot_venv\Scripts\python.exe notebooks\06_evaluate.py
D:\robot_venv\Scripts\python.exe notebooks\04_plot_learning_curve.py
D:\robot_venv\Scripts\python.exe notebooks\03_watch_trained_agent.py
```

Quick test before committing hours to training? Shorten it:
```
(PowerShell)  $env:ANT_TIMESTEPS=50000
D:\robot_venv\Scripts\python.exe notebooks\07_upright_training.py
```

## Key engineering decisions (be ready to explain these)

**1. VecNormalize (observation + reward scaling).**
Neural networks learn poorly when inputs live on wildly different scales (a
joint angle of 0.02 next to a velocity of 15). `VecNormalize` keeps a running
mean/standard deviation and rescales everything to roughly [-1, 1]. This is the
single biggest reason PPO locomotion works, and it is what the tuned reference
settings use. The statistics are saved to `vecnormalize.pkl` — you MUST load
them again to use the model, or it sees mis-scaled inputs and behaves randomly.

**2. Reward shaping.**
- `forward_reward_weight=1.0` — reward for moving forward (the goal).
- `ctrl_cost_weight=0.05` — small penalty on torque so the robot moves freely
  but not wastefully. (Default 0.5 made it too timid.)
- `healthy_reward=1.0` — per-step bonus for staying upright.
- `healthy_z_range=(0.2, 1.0)` — episode ends if the torso falls/jumps out of
  this height band.

**3. Measure distance, not just reward.**
A high reward can be earned by *standing still* and collecting the per-step
alive bonus. So `06_evaluate.py` also reports **forward distance** — that is
what proves the robot is actually *walking*. Your real Bittle project measures
distance in a fixed time; this is the same idea.

## Lessons already learned (these are results, not mistakes)

- **Too-strict termination kills learning.** An early run required the torso to
  stay above 0.35 m. The robot could not do that yet, so episodes ended in a
  few steps and it never learned — a flat, negative learning curve. Fix: use a
  permissive height range while learning, tighten later.
- **Reward can lie; distance tells the truth.** A short run scored ~995 reward
  but moved 0.08 m — it had learned to stand still, not walk.
- **Agents optimise what you measure — measure carefully.** ant_v2 learned to
  crawl because forward distance was rewarded but upright posture was not.
  Adding a height reward (ant_v3) gives the agent a reason to stand tall.
- **Change one thing at a time, starting from known-good defaults.** This is
  also exactly the method of your ablation study.

## Run folders under `models/`

- `ant_first_run/` — the 300k demo (no VecNormalize, CPU only).
- `ant_improved/` — the failed too-strict-height run (kept for comparison).
- `ant_v2/` — normalized + tuned 2M step run. Ant walked but crawled.
- `ant_v3/` — upright training: height reward + heavier contact cost. Target: upright gait.

Each keeps its own `monitor_logs/` so learning curves never mix.
