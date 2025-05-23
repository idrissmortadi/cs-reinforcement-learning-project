import os
import sys
import glob
from typing import Dict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import (
    event_accumulator,
)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env # noqa: F401

sys.path.append("..")
from configs.task_3_config import config_dict 

TRAIN = False 
LOG_DIR = "racetrack_ppo"
MODEL_PATH = os.path.join(LOG_DIR, "model")
TB_LOG_NAME = "PPO_6" 


def plot_tensorboard_logs(log_dir, tb_log_name, output_filename="training_metrics.png"):
    try:
        run_dirs = sorted(glob.glob(os.path.join(log_dir, f"{tb_log_name}_*")))
        if not run_dirs:
            run_dirs = glob.glob(os.path.join(log_dir, tb_log_name))
            if not run_dirs:
                print(
                    f"Error: No TensorBoard log directory starting with '{tb_log_name}' found in '{log_dir}'."
                )
                return

        latest_run_dir = run_dirs[-1]
        print(f"Reading logs from: {latest_run_dir}")

        event_files = glob.glob(os.path.join(latest_run_dir, "events.out.tfevents.*"))
        if not event_files:
            print(
                f"Error: No event file found in {latest_run_dir}."
            )
            return
        event_file = event_files[0]

        ea = event_accumulator.EventAccumulator(
            event_file, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        available_tags = ea.Tags()["scalars"]

        metrics_to_plot = [
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/loss",
            "train/policy_loss",
            "train/value_loss",
            "train/entropy_loss",
        ]

        plot_tags = [tag for tag in metrics_to_plot if tag in available_tags]
        if not plot_tags:
            print(
                "Error: None of the selected metrics found in TensorBoard logs."
            )
            print(f"Available metrics: {available_tags}")
            return

        num_metrics = len(plot_tags)
        ncols = min(3, num_metrics)
        nrows = (num_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 6, nrows * 4), squeeze=False
        )
        axes = axes.flatten()

        print(f"Plotting {num_metrics} metrics...")
        for i, tag in enumerate(plot_tags):
            try:
                scalar_events = ea.Scalars(tag)
                steps = np.array([event.step for event in scalar_events])
                values = np.array([event.value for event in scalar_events])

                ax = axes[i]
                ax.plot(steps, values, label=tag.split("/")[-1])
                ax.set_xlabel("Timesteps")
                ax.set_ylabel("Value")
                ax.set_title(tag)
                ax.grid(True)

            except KeyError:
                print(
                    f"Warning: Metric '{tag}' data not found. Skipping."
                )
            except Exception as e:
                print(f"Warning: Could not plot metric '{tag}'. Error: {e}. Skipping.")

        for j in range(num_metrics, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout(pad=2.0)
        save_path = os.path.join(log_dir, output_filename)
        plt.savefig(save_path)
        print(f"Saved metrics plot to: {save_path}")

    except FileNotFoundError:
        print(f"Error: Log directory '{log_dir}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)

    n_cpu = 6
    batch_size = 64
    rollout_steps_per_env = 512
    learning_rate = 3e-4
    gamma = 0.99
    total_timesteps = int(4e5)

    if TRAIN:
        print("Creating environment...")
        env = make_vec_env(
            "racetrack-v0",
            n_envs=n_cpu,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"config": config_dict},
        )

        print("Initializing PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=rollout_steps_per_env,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=1,
            tensorboard_log=LOG_DIR,
        )

        print("-" * 50)
        print(f"Starting training for {total_timesteps} timesteps...")
        print(f"Logging to: {LOG_DIR}")
        print(f"Using Environment Config from configs.task_3_config")
        print(f"PPO Hyperparameters:")
        print(f"  n_steps (rollout per env): {rollout_steps_per_env}")
        print(f"  batch_size (PPO minibatch): {batch_size}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  gamma: {gamma}")
        print("-" * 50)

        model.learn(
            total_timesteps=total_timesteps, progress_bar=True, tb_log_name=TB_LOG_NAME
        )
        model.save(MODEL_PATH)
        print(f"Training finished and model saved to {MODEL_PATH}.zip!")

        print("\nGenerating training metrics plot...")
        plot_tensorboard_logs(LOG_DIR, TB_LOG_NAME, "ppo_racetrack_metrics.png")

        print("Closing environment...")
        env.close()

    else:
        print("TRAIN=False. Skipping training.")
        print("Attempting to generate plots from existing logs (if found)...")
        if os.path.exists(LOG_DIR):
            plot_tensorboard_logs(LOG_DIR, TB_LOG_NAME, "ppo_racetrack_metrics.png")
        else:
            print(f"Log directory '{LOG_DIR}' not found. Cannot plot.")

    print("Script finished.")