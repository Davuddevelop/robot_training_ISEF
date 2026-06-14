# Sim-to-Real Quadruped Locomotion — ISEF Research Project

**Research question:**
> Which simulation-to-reality mismatch factors most affect locomotion performance on a low-cost hobby-servo quadruped, and how much does basic system identification improve transfer compared to domain randomization alone?

## What this project does

A reinforcement learning policy is trained entirely in simulation (MuJoCo) to make a Petoi Bittle quadruped walk. The policy is then transferred to the real robot, and we measure how well different techniques (domain randomization, system identification) bridge the sim-to-real gap.

This is a **controlled scientific study** — the goal is rigorous, graphable data comparing conditions, not a deployed product.

## Hardware

- **Robot:** Petoi Bittle (alloy servos, 8 leg joints as action space)
- **Control board:** Petoi NyBoard with built-in IMU
- **Training machine:** ASUS laptop with NVIDIA RTX 30-series GPU (CUDA)

## Software stack

| Purpose | Library |
|---|---|
| Deep learning | PyTorch (CUDA) |
| Physics simulation | MuJoCo |
| RL environment API | Gymnasium |
| RL algorithm | PPO via Stable-Baselines3 |
| Policy export | ONNX |

## Project structure

```
src/sim/          MuJoCo simulation environment
src/train/        PPO training scripts
src/robot/        Serial communication with the real Bittle
src/experiments/  Ablation study and data collection
notebooks/        Learning notebooks (Phase 1 foundations)
models/           Saved policy checkpoints
data/             Experiment results and logs
```

## Competition

"Sabahın Alimləri" (Scientists of Tomorrow) — Azerbaijan national science fair, Robotics & AI category. Finalist pathway to ISEF.
