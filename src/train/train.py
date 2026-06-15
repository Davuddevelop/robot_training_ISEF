"""
train.py — Trains the walking policy using PPO.

This script:
  1. Creates the simulation environment.
  2. Creates a PPO agent (the thing that learns).
  3. Runs training for several million steps.
  4. Saves the trained policy to disk.

Run with:
    python src/train/train.py

You will fully understand this file after completing Phase 3 of your learning
roadmap (RL concepts + Stable-Baselines3). Every line below is labeled with
which concept it uses so you know what to study.
"""

import datetime
import pathlib

# Concept: your custom Gymnasium environment (Phase 3)
from src.sim.bittle_env import BittleEnv

# Concept: PPO algorithm from Stable-Baselines3 (Phase 3)
from stable_baselines3 import PPO

# Concept: logging training progress (Phase 3)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# All hyperparameters come from config — never hardcode numbers in scripts
from src.sim.config import PPO as PPO_CFG, MODELS_DIR


def train(
    total_timesteps=PPO_CFG["total_timesteps"],
    domain_rand=True,
    run_name=None,
):
    """
    Main training function.

    Args:
        total_timesteps: how many environment steps to train for
        domain_rand:     whether to use domain randomization
        run_name:        label for this run (auto-generated if None)
    """
    # --- RUN LABEL ---
    # Every run gets a timestamped folder so results don't overwrite each other.
    # Condition names match the experimental conditions from your research plan:
    #   "no_dr"       = Condition 1: no domain randomization
    #   "full_dr"     = Condition 2: full domain randomization
    #   "dr_sysid"    = Condition 3: domain randomization + system ID (Phase 4)
    if run_name is None:
        condition = "full_dr" if domain_rand else "no_dr"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{condition}_{timestamp}"

    save_dir = MODELS_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining run: {run_name}")
    print(f"Saving to: {save_dir}\n")

    # --- ENVIRONMENT ---
    # Concept (Phase 3): the environment is where the agent lives and learns.
    # We pass domain_rand=True to enable randomization during training.
    env = BittleEnv(render_mode=None, domain_rand=domain_rand)

    # --- PPO AGENT ---
    # Concept (Phase 3): "MlpPolicy" = a simple fully-connected neural network policy.
    # The network reads the 23-number observation and outputs 8 joint angle offsets.
    # "verbose=1" prints training progress to the terminal.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_CFG["learning_rate"],
        n_steps=PPO_CFG["n_steps"],
        batch_size=PPO_CFG["batch_size"],
        n_epochs=PPO_CFG["n_epochs"],
        gamma=PPO_CFG["gamma"],
        gae_lambda=PPO_CFG["gae_lambda"],
        clip_range=PPO_CFG["clip_range"],
        verbose=1,
        tensorboard_log=str(save_dir / "logs"),
    )

    # --- CALLBACKS ---
    # Concept (Phase 3): callbacks are functions SB3 calls during training.
    # CheckpointCallback: saves the model every N steps (so you don't lose progress).
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="bittle_ppo",
    )

    # --- TRAINING ---
    # This is the line that actually runs training.
    # It will take hours. The terminal will print progress every few thousand steps.
    # You can interrupt with Ctrl+C and resume from the last checkpoint.
    print("Starting training. Press Ctrl+C to stop early.")
    print("Watch the 'ep_rew_mean' value — it should increase over time.\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # --- SAVE FINAL MODEL ---
    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"\nTraining complete. Model saved to: {final_path}.zip")

    env.close()
    return model, str(final_path)


if __name__ == "__main__":
    # Running this file directly starts a training run with domain randomization on.
    # To run WITHOUT domain randomization (Condition 1):
    #   train(domain_rand=False, run_name="no_dr")
    train(domain_rand=True)
