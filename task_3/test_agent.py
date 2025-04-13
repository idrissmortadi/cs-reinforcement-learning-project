import os
import sys
from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env

sys.path.append("..")
from configs.task_3_config import config_dict

TRAIN = True
LOG_DIR = "racetrack_ppo_v2"
MODEL_PATH = os.path.join(LOG_DIR, "model")
VIDEO_DIR = os.path.join(LOG_DIR, "videos")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Paramètres d'entraînement
n_cpu = 1
batch_size = 64
total_timesteps = int(1e5)

# Créer l'environnement vectorisé
# Create the base environment
eval_env = gym.make("racetrack-v0", render_mode="rgb_array")
eval_env.unwrapped.configure(config_dict)


def evaluate_agent(model, eval_env, n_eval_episodes=10) -> Dict[str, float]:
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = truncated = False
        cumulative_reward = 0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(action)
            cumulative_reward += reward
            steps += 1

        episode_rewards.append(cumulative_reward)
        episode_lengths.append(steps)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }


model = PPO.load(MODEL_PATH, env=eval_env)

# Environnement d'évaluation avec enregistrement vidéo

eval_env = RecordVideo(eval_env, video_folder=VIDEO_DIR, episode_trigger=lambda e: True)

# Évaluation finale
eval_metrics = evaluate_agent(model, eval_env)
print("\nMétriques d'évaluation:")
print(
    f"Récompense moyenne: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}"
)
print(
    f"Longueur moyenne: {eval_metrics['mean_length']:.2f} ± {eval_metrics['std_length']:.2f}"
)

# Enregistrement de quelques épisodes
for video in range(10):
    done = truncated = False
    obs, info = eval_env.reset()
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        eval_env.render()

eval_env.close()
