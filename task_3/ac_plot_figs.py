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

from stable_baselines3 import A2C  
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401

sys.path.append("..")
from configs.task_3_config import config_dict

TRAIN = True
LOG_DIR = "racetrack_a2c"  
MODEL_PATH = os.path.join(LOG_DIR, "model")
TB_LOG_NAME = "A2C_1"  


def plot_tensorboard_logs(log_dir, tb_log_name, output_filename="a2c_training_metrics.png"): # Changed default filename
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
                f"Error: No event file found in {latest_run_dir}. Training might have failed or logs not written yet."
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
                "Error: None of the selected important metrics were found in the TensorBoard logs."
            )
            print(f"Available metrics were: {available_tags}")
            return

        plt.style.use("seaborn-v0_8-darkgrid")

        num_metrics = len(plot_tags)
        ncols = min(3, num_metrics)
        nrows = (num_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 6, nrows * 4),
            squeeze=False,
            dpi=120,
        )
        axes = axes.flatten()

        colors = plt.cm.viridis(np.linspace(0, 0.9, len(plot_tags)))

        print(f"Plotting {num_metrics} metrics...")
        for i, (tag, color) in enumerate(zip(plot_tags, colors)):
            try:
                scalar_events = ea.Scalars(tag)
                steps = np.array([event.step for event in scalar_events])
                values = np.array([event.value for event in scalar_events])

                ax = axes[i]

                ax.plot(
                    steps, values, linewidth=2, color=color, label=tag.split("/")[-1]
                )

                if len(steps) > 10:
                    window_size = min(20, len(steps) // 10)
                    rolling_avg = np.convolve(
                        values, np.ones(window_size) / window_size, mode="valid"
                    )
                    rolling_steps = steps[window_size - 1 :]
                    ax.plot(
                        rolling_steps,
                        rolling_avg,
                        linewidth=2.5,
                        color=color,
                        alpha=0.7,
                        linestyle="--",
                        label=f"Moving Avg (n={window_size})",
                    )

                ax.set_xlabel("Timesteps", fontweight="bold")
                ax.set_ylabel("Value", fontweight="bold")
                clean_title = tag.replace("_", " ").replace("/", " - ").title()
                ax.set_title(clean_title, fontsize=12, pad=10, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend(frameon=True, fancybox=True, framealpha=0.9)

                ax.get_xaxis().set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{int(x):,}")
                )

                ax.set_facecolor("#f9f9f9")

            except KeyError:
                print(
                    f"Warning: Metric '{tag}' listed but data not found by EventAccumulator. Skipping."
                )
            except Exception as e:
                print(f"Warning: Could not plot metric '{tag}'. Error: {e}. Skipping.")

        for j in range(num_metrics, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            f"Training Metrics for A2C on Racetrack Environment", # Changed title
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])

        save_path = os.path.join(log_dir, output_filename)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved metrics plot to: {save_path}")

    except FileNotFoundError:
        print(f"Error: Log directory '{log_dir}' not found. Cannot generate plots.")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)

    n_cpu = 6
    n_steps = 5 
    learning_rate = 7e-4 
    gamma = 0.99
    total_timesteps = int(4e5)

    print("Creating environment...")
    env = make_vec_env(
        "racetrack-v0",
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": config_dict},
    )

    print("Initializing A2C model...")
    model = A2C(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=1,
        tensorboard_log=LOG_DIR,
        # A2C specific parameters can be added here if needed, e.g.:
        # vf_coef=0.5,
        # ent_coef=0.0,
    )

    if TRAIN:
        print("-" * 50)
        print(f"Starting training for {total_timesteps} timesteps...")
        print(f"Logging to: {LOG_DIR}")
        print(f"Using Environment Config from configs.task_3_config")
        print(f"A2C Hyperparameters:") # Changed printout
        print(f"  n_steps (rollout per env): {n_steps}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  gamma: {gamma}")
        print("-" * 50)

        model.learn(
            total_timesteps=total_timesteps, progress_bar=True, tb_log_name=TB_LOG_NAME
        )
        model.save(MODEL_PATH)
        print(f"Training finished and model saved to {MODEL_PATH}.zip!")

        print("\nGenerating training metrics plot...")
        plot_tensorboard_logs(LOG_DIR, TB_LOG_NAME, "a2c_racetrack_metrics.png") # Changed filename

    else:
        print("TRAIN=False. Skipping training.")
        print("Attempting to generate plots from existing logs (if found)...")
        if os.path.exists(LOG_DIR):
            plot_tensorboard_logs(LOG_DIR, TB_LOG_NAME, "a2c_racetrack_metrics.png") # Changed filename
        else:
            print(f"Log directory '{LOG_DIR}' not found. Cannot plot.")

    print("Closing environment...")
    env.close()
    print("Script finished.")