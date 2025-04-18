import argparse
import logging
import os
import sys
import time  # Import the time module

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
from hyperparameters import device
from q_network import QNetwork
from utils import preprocess_state

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.task_1_config import ENVIRONEMNT, config_dict


def play_or_record(model_path, video_folder=None, live=False, num_episodes=5):
    """
    Load a trained model and either play the environment live or record its performance.

    Args:
        model_path (str): Path to the saved model file.
        video_folder (str): Directory to save videos if recording. Ignored if `live` is True.
        live (bool): If True, plays the environment live. If False, records the performance.
        num_episodes (int): Number of episodes to play or record.
    """
    logging.info(f"Loading model from: {model_path}")

    # Initialize the environment
    env = gym.make(ENVIRONEMNT, render_mode="human" if live else "rgb_array")
    env.unwrapped.configure(config_dict)

    state, _ = env.reset()
    # Preprocess the initial state to get the correct shape
    processed_state = preprocess_state(state)

    # The input dimension for the Q-network is the size of the flattened state vector
    input_dim = processed_state.size
    logging.info(f"State preprocessed. Input dimension for Q-network: {input_dim}")

    # Get the number of discrete actions from the environment's action space
    n_actions = env.action_space.n
    logging.info(f"Number of discrete actions: {n_actions}")

    # Load the trained model
    policy_net = QNetwork(input_dim=input_dim, output_dim=n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    if not live and video_folder:
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env, video_folder=video_folder, name_prefix="play_or_record", fps=16
        )

    episode = 0
    while episode < num_episodes:
        logging.info(f"Starting episode {episode + 1}/{num_episodes}")
        state, _ = env.reset()
        state = preprocess_state(state).flatten()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action = policy_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = preprocess_state(next_state).flatten()
            done = terminated or truncated
            episode_reward += reward

            # Add a small delay for live mode to slow down rendering
            if live:
                time.sleep(0.1)  # Adjust sleep duration as needed

        logging.info(f"Episode {episode + 1} finished with reward: {episode_reward}")
        if not live:
            episode += 1

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play or record a trained agent.")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the saved model file."
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help="Directory to save videos (if recording).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Play the environment live instead of recording.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to play or record.",
    )

    args = parser.parse_args()

    play_or_record(
        model_path=args.model,
        video_folder=args.video_folder,
        live=args.live,
        num_episodes=args.num_episodes,
    )
