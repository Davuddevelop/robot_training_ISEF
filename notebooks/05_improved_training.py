"""
05_improved_training.py — The best-quality training run (Ant, dog-like gait).

This is the upgraded pipeline. Compared with the earlier runs it adds the one
ingredient that most reliably makes PPO locomotion actually work:

    VecNormalize — it keeps a running mean and standard deviation of the
    observations (and rewards) and rescales them so every number the network
    sees is roughly in the range [-1, 1]. Neural networks learn badly when
    some inputs are tiny (a joint angle of 0.02) and others are huge (a
    velocity of 15). Normalizing puts them on the same scale. This is standard
    practice in every serious RL codebase and is exactly what the tuned
    reference settings (Stable-Baselines3 RL Zoo) use for Ant.

Everything is commented so you can read and defend it. The hyperparameters are
close to the community-standard PPO-for-MuJoCo recipe, not random guesses.

Run with (after `git pull`):
    D:\\robot_venv\\Scripts\\python.exe notebooks\\05_improved_training.py

Want a shorter test first? Override the step count:
    (PowerShell)  $env:ANT_TIMESTEPS=50000; D:\\robot_venv\\Scripts\\python.exe notebooks\\05_improved_training.py
"""

import os
import pathlib

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# --- WHERE EVERYTHING IS SAVED ---
# A fresh run folder, kept separate from the earlier (failed) runs so its
# learning curve and model stay clean and comparable.
RUN_NAME = "ant_v2"
SAVE_DIR = pathlib.Path(__file__).parent.parent / "models" / RUN_NAME
MONITOR_DIR = SAVE_DIR / "monitor_logs"
MODEL_PATH = SAVE_DIR / "ant_v2_model"          # the trained network (.zip added automatically)
VECNORM_PATH = SAVE_DIR / "vecnormalize.pkl"    # the saved obs/reward scaling — needed to use the model later

# ---------------------------------------------------------------------------
# REWARD DESIGN (Ant-v5 lets us shape the reward through these parameters).
# These are the SAME ideas we will use for Bittle, so understanding this block
# means understanding your project's reward design.
# ---------------------------------------------------------------------------
ENV_KWARGS = {
    # Reward for moving forward (x-direction). The main objective.
    "forward_reward_weight": 1.0,

    # Penalty for large joint torques (energy proxy). Default 0.5 is timid;
    # 0.05 lets the robot use its legs decisively -> more natural movement.
    "ctrl_cost_weight": 0.05,

    # Penalty for hard foot impacts. Tested default — left alone.
    "contact_cost_weight": 5e-4,

    # Reward per step for staying "healthy" (upright and within height range).
    # Over up to 1000 steps this is what makes a good episode score high.
    "healthy_reward": 1.0,

    # Episode ENDS if the torso leaves this height band (metres).
    # The tested default (0.2, 1.0). An earlier run used 0.35 here and it
    # killed episodes before the agent could learn — we do NOT repeat that.
    "healthy_z_range": (0.2, 1.0),

    # Max steps per episode before reset (if it hasn't fallen).
    "max_episode_steps": 1000,
}

# ---------------------------------------------------------------------------
# TRAINING SETTINGS (PPO). Close to the standard PPO-for-MuJoCo recipe.
# ---------------------------------------------------------------------------
N_ENVS = 4                       # parallel simulations -> bigger, more stable batches
TOTAL_STEPS = int(os.environ.get("ANT_TIMESTEPS", 2_000_000))  # override via env var for quick tests

PPO_KWARGS = dict(
    learning_rate=3e-4,          # step size for weight updates
    n_steps=2048,                # steps PER ENV before each update (x4 envs = 8192 per update)
    batch_size=256,              # minibatch size for each gradient step (8192 / 256 = 32 minibatches)
    n_epochs=10,                 # times each batch of data is reused
    gamma=0.99,                  # how far ahead the agent plans
    gae_lambda=0.95,             # bias/variance trade-off in advantage estimation
    clip_range=0.2,              # PPO's core: cap how far the policy moves per update
    ent_coef=0.0,                # exploration bonus (0.0 is standard for continuous control)
    vf_coef=0.5,                 # weight of the value-function loss
    max_grad_norm=0.5,           # clip gradients to avoid destabilising spikes
    policy_kwargs={"net_arch": [256, 256]},  # two hidden layers of 256 neurons
    device="cpu",                # small MLP trains faster on CPU than GPU
    verbose=1,
)


def make_normalized_env():
    """
    Build the vectorised, monitored, NORMALIZED training environment.

    Order matters:
      1. make_vec_env wraps each Ant in a Monitor that records the TRUE
         (un-normalized) episode reward -> the learning curve stays meaningful.
      2. VecNormalize then rescales what the *agent* sees, without touching
         what Monitor logged.
    """
    venv = make_vec_env(
        "Ant-v5",
        n_envs=N_ENVS,
        monitor_dir=str(MONITOR_DIR),
        env_kwargs=ENV_KWARGS,
    )
    venv = VecNormalize(
        venv,
        norm_obs=True,     # rescale observations to ~zero mean / unit variance
        norm_reward=True,  # rescale rewards too (helps PPO stability)
        clip_obs=10.0,     # clip extreme normalized values
    )
    return venv


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Training — Ant, normalized + tuned (best version)")
    print("=" * 64)
    print(f"  Run folder:     {SAVE_DIR}")
    print(f"  Total steps:    {TOTAL_STEPS:,}  (override with ANT_TIMESTEPS)")
    print(f"  Parallel envs:  {N_ENVS}")
    print(f"  Key upgrade:    VecNormalize (obs + reward scaling)")
    print("=" * 64)
    print()

    env = make_normalized_env()
    print(f"Observation size: {env.observation_space.shape[0]} numbers")
    print(f"Action size:      {env.action_space.shape[0]} joint torques")

    model = PPO("MlpPolicy", env, **PPO_KWARGS)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy network:   {n_params:,} parameters (two 256-neuron layers)")
    print()

    # Save the model periodically so a crash or Ctrl+C never loses everything.
    checkpoint = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),
        save_path=str(SAVE_DIR / "checkpoints"),
        name_prefix="ant_v2",
        verbose=1,
    )

    print("Training... watch 'rollout/ep_rew_mean' — it should climb.")
    print("(Press Ctrl+C to stop early; the latest checkpoint is kept.)\n")
    model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint)

    # Save BOTH the model AND the normalization statistics. You need both to
    # use the policy later — feeding raw, un-normalized observations to a model
    # trained on normalized ones produces garbage. This is the #1 mistake
    # people make with VecNormalize.
    model.save(str(MODEL_PATH))
    env.save(str(VECNORM_PATH))

    print()
    print("=" * 64)
    print("  Done.")
    print(f"  Model saved:         {MODEL_PATH}.zip")
    print(f"  Normalization stats: {VECNORM_PATH}")
    print()
    print("  Next:")
    print("    1) python notebooks/04_plot_learning_curve.py   (curve should climb)")
    print("    2) python notebooks/06_evaluate.py              (numbers: mean reward)")
    print("    3) python notebooks/03_watch_trained_agent.py   (watch it walk)")
    print("=" * 64)
    env.close()


if __name__ == "__main__":
    # The __main__ guard is standard practice: it lets other files import this
    # one without accidentally launching a training run, and it is required if
    # you ever switch to subprocess-based parallel environments on Windows.
    main()
