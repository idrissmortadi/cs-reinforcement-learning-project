import os
import sys
from typing import Dict

import gymnasium as gym
import numpy as np

# from gymnasium.wrappers import RecordVideo # Keep if you want videos later
from stable_baselines3 import PPO

# from stable_baselines3.common.callbacks import BaseCallback # Keep if you add callbacks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401

sys.path.append("..")
# Assuming your config is accessible like this:
from configs.task_3_config import config_dict  # Import directly


TRAIN = True
LOG_DIR = "racetrack_ppo_v2"  # Use a new log dir for the new run
MODEL_PATH = os.path.join(LOG_DIR, "model")
# VIDEO_DIR = os.path.join(LOG_DIR, "videos") # Keep if recording videos

if __name__ == "__main__":
    # Create necessary folders
    os.makedirs(LOG_DIR, exist_ok=True)
    # os.makedirs(VIDEO_DIR, exist_ok=True)

    # Training parameters
    n_cpu = 4
    batch_size = 64  # Mini-batch size for PPO updates, 64 is often fine
    # --- Hyperparameter Adjustments ---
    rollout_steps_per_env = 512  # Increased n_steps per environment
    learning_rate = 3e-4  # Reduced learning rate
    gamma = 0.99  # Increased discount factor
    total_timesteps = int(2e6)  # Increased total training time (adjust as needed)

    # Create the vectorized environment
    env = make_vec_env(
        "racetrack-v0",
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": config_dict},  # Pass the modified config
    )

    # Initialize the model
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=rollout_steps_per_env,  # Steps per env before update
        batch_size=batch_size,  # Mini-batch size during update epochs
        n_epochs=10,  # PPO internal update epochs
        learning_rate=learning_rate,  # Adjusted learning rate
        gamma=gamma,  # Adjusted discount factor
        verbose=1,
        tensorboard_log=LOG_DIR,
        # ent_coef=0.0, # Default is usually 0.0, can adjust if needed
        # vf_coef=0.5,  # Default is usually 0.5
        # max_grad_norm=0.5, # Default is usually 0.5
    )

    # Training
    if TRAIN:
        print(f"Starting training for {total_timesteps} timesteps...")
        print(f"Using config: {config_dict}")
        print(
            f"PPO Hyperparameters: n_steps={rollout_steps_per_env}, batch_size={batch_size}, lr={learning_rate}, gamma={gamma}"
        )
        model.learn(
            total_timesteps=total_timesteps, progress_bar=True, tb_log_name="PPO"
        )
        model.save(MODEL_PATH)
        print(f"Training finished and model saved to {MODEL_PATH}!")

    env.close()
