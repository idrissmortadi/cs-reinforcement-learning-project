import os
import sys
from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback # Keep if used later
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env # noqa: F401

sys.path.append("..")
from configs.task_3_config import config_dict # Ensure this path is correct

TRAIN = False # Set to True if you want to retrain, False to evaluate
LOG_DIR = "racetrack_ppo_v2" # Directory from training script
MODEL_PATH = os.path.join(LOG_DIR, "model")
VIDEO_DIR = os.path.join(LOG_DIR, "videos_evaluation") # Separate evaluation videos

os.makedirs(VIDEO_DIR, exist_ok=True) 

eval_env = gym.make("racetrack-v0", render_mode="rgb_array")
eval_env.unwrapped.configure(config_dict)


def evaluate_agent(model, eval_env, n_eval_episodes=5) -> Dict[str, float]:
    episode_rewards = []
    episode_lengths = []

    for i in range(n_eval_episodes):
        print(f"Starting evaluation episode {i + 1}/{n_eval_episodes}")
        obs, _ = eval_env.reset()
        done = truncated = False
        cumulative_reward = 0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            cumulative_reward += reward
            steps += 1
            # Optional: Render during evaluation if needed, though RecordVideo handles it
            # eval_env.render()

        episode_rewards.append(cumulative_reward)
        episode_lengths.append(steps)
        print(f"Episode {i + 1} finished: Reward={cumulative_reward}, Length={steps}")

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }

if not os.path.exists(f"{MODEL_PATH}.zip"):
    print(f"Error: Model file not found at {MODEL_PATH}.zip")
    print("Please ensure the model was trained and saved correctly, or update MODEL_PATH.")
    sys.exit(1)

print(f"Loading model from {MODEL_PATH}.zip")
model = PPO.load(MODEL_PATH, env=None) # No need to pass env if just evaluating/predicting

print(f"Setting up evaluation environment with video recording to {VIDEO_DIR}")
eval_env_video = RecordVideo(eval_env, video_folder=VIDEO_DIR, episode_trigger=lambda e: e % 2 == 0, name_prefix="racetrack-eval") # Record every 2nd episode

print("Starting evaluation...")
eval_metrics = evaluate_agent(model, eval_env_video, n_eval_episodes=10) # Run 10 episodes for evaluation

print("\nEvaluation Metrics:")
print(
    f"Mean Reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}"
)
print(
    f"Mean Length: {eval_metrics['mean_length']:.2f} +/- {eval_metrics['std_length']:.2f}"
)
print(f"Videos saved in: {VIDEO_DIR}")

eval_env_video.close() # Close the wrapped env
eval_env.close() # Close the base env
print("Evaluation finished.")