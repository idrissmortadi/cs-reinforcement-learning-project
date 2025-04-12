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
LOG_DIR = "racetrack_ppo"
MODEL_PATH = os.path.join(LOG_DIR, "model")
VIDEO_DIR = os.path.join(LOG_DIR, "videos")

if __name__ == "__main__":
    # Créer les dossiers nécessaires
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # Paramètres d'entraînement
    n_cpu = 4
    batch_size = 64
    total_timesteps = int(1e5)

    # Créer l'environnement vectorisé
    env = make_vec_env(
        "racetrack-v0",
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": config_dict},
    )

    # Initialiser le modèle
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # Entraînement
    if TRAIN:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(MODEL_PATH)
        print("Entraînement terminé et modèle sauvegardé!")
