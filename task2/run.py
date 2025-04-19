"""
Policy Gradient (REINFORCE) implementation for a custom Gymnasium environment.

This script trains a Policy Gradient agent to interact with an environment
and learn an optimal policy. The environment and its configuration are defined
in the `configs.task_1_config` module.

Key Features:
- Preprocessing of state observations.
- Policy network for action selection.
- REINFORCE algorithm for training.
- Periodic saving of rewards and the trained model.

Usage:
    python task1.py
"""

import datetime
import logging
import os
import signal  # For handling Ctrl+C
import sys

# For accessing the config module;
sys.path.append("..")


import gymnasium as gym
import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from torch.utils.tensorboard import SummaryWriter

from configs.task_2_config import ENVIRONMENT, config_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Create environment using the config
ENV_NAME = ENVIRONMENT
env = gym.make(ENV_NAME, render_mode="rgb_array")
env.unwrapped.configure(config_dict)


# If observations are dicts (for OccupancyGrid), flatten them
def preprocess_state(state):
    """
    Preprocess the state observation by flattening it if it is a dictionary.

    Args:
        state (Union[dict, np.ndarray]): The raw state observation.

    Returns:
        np.ndarray: The preprocessed state as a flattened NumPy array.
    """
    if isinstance(state, dict):
        # Flatten state with sorted keys to ensure consistent ordering
        state = np.concatenate(
            [np.array(state[k]).flatten() for k in sorted(state.keys())]
        )
    return np.array(state, dtype=np.float32).flatten()  # Ensure the state is flattened


# DQN Hyperparameters
BATCH_SIZE = 64  # Batch size for training
GAMMA = 0.99  # Discount factor for future rewards
LR = 1e-4  # Learning rate for the optimizer
TARGET_UPDATE = 1_000  # Number of episodes between updates of the target network
MEMORY_SIZE = 10_000  # Maximum capacity of the replay buffer
NUM_EPISODES = 5_000  # Total number of episodes for training

# Determine input dimension from a sample observation
state, _ = env.reset()
state = preprocess_state(state)
input_dim = state.size  # Use the size of the flattened state
logging.debug(
    f"Calculated input_dim: {input_dim}, Flattened state shape: {state.shape}"
)

# Determine number of actions (for continuous action space)
if isinstance(env.action_space, gym.spaces.Box):
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
else:
    raise ValueError(
        "This script currently only supports continuous action spaces (gym.spaces.Box)"
    )

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


class PolicyNetwork(nn.Module):
    """
    Neural network for approximating the policy function for continuous actions.
    Outputs mean and log standard deviation for a Gaussian distribution.

    Attributes:
        layer1 (nn.Linear): First fully connected layer.
        layer2 (nn.Linear): Second fully connected layer.
        mean_layer (nn.Linear): Output layer for action means.
        log_std_layer (nn.Linear): Output layer for log standard deviations.
    """

    def __init__(self, input_dim, action_dim):
        """
        Initialize the PolicyNetwork.
n_actions
        Args:
            input_dim (int): Dimension of the input state.
            action_dim (int): Number of action dimensions.
        """
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)  # Output log_std

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log standard deviation of the action distribution.
        """
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        mean = self.mean_layer(x)
        # Clamp log_std for stability
        log_std = torch.clamp(self.log_std_layer(x), min=-20, max=2)
        return mean, log_std


def select_action(state, policy_net):
    """
    Select an action using the policy network for continuous action spaces.

    Args:
        state (np.ndarray): Current state.
        policy_net (PolicyNetwork): Policy network.

    Returns:
        np.ndarray: Selected action.
        torch.Tensor: Log probability of the selected action.
    """
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logging.debug(f"State tensor shape: {state_tensor.shape}")
    mean, log_std = policy_net(state_tensor)
    std = torch.exp(log_std)
    action_dist = distributions.Normal(mean, std)
    action = action_dist.sample()  # Sample action
    log_prob = action_dist.log_prob(action).sum(
        dim=-1
    )  # Sum log probs across action dimensions

    # Clip action to environment bounds
    action_np = action.cpu().detach().numpy().flatten()
    clipped_action = np.clip(action_np, action_low, action_high)

    # Note: Clipping might affect gradient calculation if not handled carefully.
    # For simplicity here, we use the log_prob of the unclipped action.
    # More advanced methods might re-evaluate log_prob or use transformations (e.g., Tanh).

    return clipped_action, log_prob


def compute_returns(rewards, gamma):
    """
    Compute discounted returns for a trajectory.

    Args:
        rewards (List[float]): List of rewards from the trajectory.
        gamma (float): Discount factor.

    Returns:
        List[float]: Discounted returns.
    """
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


def optimize_model(policy_net, optimizer, log_probs, returns):
    """
    Perform a single optimization step on the policy network.

    Args:
        policy_net (PolicyNetwork): Policy network.
        optimizer (torch.optim.Optimizer): Optimizer for the policy network.
        log_probs (List[torch.Tensor]): Log probabilities of actions taken.
        returns (List[float]): Discounted returns.
    """
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize returns

    loss = -torch.sum(torch.stack(log_probs) * returns)  # Policy gradient loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Create directories for saving models and metrics
SAVE_DIR = "results"
MODEL_DIR = os.path.join(SAVE_DIR, "models")
METRICS_DIR = os.path.join(SAVE_DIR, "metrics")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def record_agent(policy_net, env, video_folder, run_name, num_eval_episodes=4):
    """
    Record a video of the trained agent's performance.

    Args:
        policy_net (PolicyNetwork): Trained policy network.
        env (gym.Env): Environment to evaluate the agent.
        video_folder (str): Directory to save the video.
        run_name (str): Name of the current run for organizing videos.
        num_eval_episodes (int): Number of episodes to record.
    """
    # Create a subdirectory for the current run's videos
    run_video_folder = os.path.join(video_folder, run_name)
    os.makedirs(run_video_folder, exist_ok=True)
    logging.info(f"Recording videos to folder: {run_video_folder}")
    env = RecordVideo(
        env,
        video_folder=run_video_folder,
        name_prefix="eval",
        fps=10,
        episode_trigger=lambda episode_id: True,  # Record all episodes
    )
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    for episode_num in range(num_eval_episodes):
        logging.info(
            f"Starting evaluation episode {episode_num + 1}/{num_eval_episodes}"
        )
        obs, info = env.reset()
        obs = preprocess_state(obs).flatten()  # Ensure the observation is flattened
        logging.debug(f"Initial observation: {obs}")

        # Dynamically verify input dimension
        if obs.size != policy_net.layer1.in_features:
            raise ValueError(
                f"Mismatch between observation size ({obs.size}) and policy network input size ({policy_net.layer1.in_features})."
            )

        episode_over = False
        total_reward = 0
        step_count = 0

        while not episode_over:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).view(
                    1, -1
                )
                # Get mean action from the policy network for deterministic evaluation
                mean, _ = policy_net(obs_tensor)
                action = mean.cpu().detach().numpy().flatten()
                # Clip action to environment bounds
                action = np.clip(action, action_low, action_high)

            obs, reward, terminated, truncated, info = env.step(action)
            obs = preprocess_state(
                obs
            ).flatten()  # Ensure the next observation is flattened
            total_reward += reward
            step_count += 1
            logging.debug(
                f"Step {step_count}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
            )

            episode_over = terminated or truncated

        logging.info(
            f"Episode {episode_num + 1} completed. Total reward: {total_reward}, Steps: {step_count}"
        )

    env.close()
    logging.info(f"Video recording completed. Videos saved to: {run_video_folder}")
    print(f"Episode total rewards: {env.return_queue}")
    print(f"Episode lengths: {env.length_queue}")


def get_run_name(custom_name=None):
    """
    Generate a descriptive name for the current experiment run.

    Args:
        custom_name (str, optional): Custom name provided by the user. Defaults to None.

    Returns:
        str: Generated run name.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if custom_name:
        return f"{custom_name}_{timestamp}"
    return f"PolicyGradient_lr{LR}_gamma{GAMMA}_episodes{NUM_EPISODES}_{timestamp}"


# Global variable to track if training should stop
stop_training = False
training_started = False


def signal_handler(sig, frame):
    """
    Handle Ctrl+C signal to save a model checkpoint and stop training.
    """
    global stop_training
    global training_started
    # Set stop_training to True if training has started
    stop_training = True and training_started
    if stop_training:
        logging.info("Training has been interrupted. Saving model checkpoint...")


signal.signal(signal.SIGINT, signal_handler)


def list_checkpoints(model_dir):
    """
    List all available checkpoints in the model directory.

    Args:
        model_dir (str): Directory containing model checkpoints.

    Returns:
        List[str]: List of checkpoint filenames.
    """
    return [f for f in os.listdir(model_dir) if f.endswith(".pth")]


def load_checkpoint(policy_net, model_dir):
    """
    Prompt the user to select a checkpoint to load.

    Args:
        policy_net (PolicyNetwork): Policy network to load the checkpoint into.
        model_dir (str): Directory containing model checkpoints.

    Returns:
        PolicyNetwork: Policy network with loaded weights.
    """
    checkpoints = list_checkpoints(model_dir)
    if not checkpoints:
        logging.info("No checkpoints found. Starting from scratch.")
        return policy_net

    print("Available checkpoints:")
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i}. {checkpoint}")

    while True:
        try:
            choice = int(
                input("Select a checkpoint to load (enter number or 0 to skip): ")
            )
            if 1 <= choice <= len(checkpoints):
                checkpoint_path = os.path.join(model_dir, checkpoints[choice - 1])
                policy_net.load_state_dict(torch.load(checkpoint_path))
                logging.info(f"Loaded checkpoint: {checkpoints[choice - 1]}")
                return policy_net
            elif choice == 0:
                print("Starting from scratch without loading a checkpoint.")
                return policy_net
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    """
    Main training loop for the policy gradient agent.

    - Initializes the environment, policy network, and optimizer.
    - Runs episodes to train the agent.
    - Periodically saves metrics and the trained model.
    """
    global training_started

    policy_net = PolicyNetwork(input_dim, action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # Ask user for a custom run name
    custom_run_name = input(
        "Enter a custom run name (or press Enter to use default): "
    ).strip()
    run_name = get_run_name(custom_run_name if custom_run_name else None)

    # Initialize TensorBoard writer
    log_dir = os.path.join(SAVE_DIR, "tensorboard", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard run name: {run_name}")

    # Log hyperparameters to TensorBoard
    writer.add_hparams(
        {
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "num_episodes": NUM_EPISODES,
        },
        {},
    )

    # For tracking metrics
    all_episode_rewards = []
    avg_window = 20  # For moving average

    training_started = True
    for episode in range(1, NUM_EPISODES + 1):
        if stop_training:  # Check if Ctrl+C was pressed
            checkpoint_path = os.path.join(
                MODEL_DIR, f"checkpoint_episode_{episode}.pth"
            )
            torch.save(policy_net.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
            break

        logging.info(f"--- Episode {episode} started ---")
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False

        log_probs = []
        rewards = []

        while not done:
            action, log_prob = select_action(state, policy_net)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_state)

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            total_reward += reward

        # Compute returns and optimize the policy network
        returns = compute_returns(rewards, GAMMA)
        optimize_model(policy_net, optimizer, log_probs, returns)

        all_episode_rewards.append(total_reward)
        writer.add_scalar("Reward/Per_Episode", total_reward, episode)

        # Calculate moving average if possible
        if len(all_episode_rewards) >= avg_window:
            avg_reward = sum(all_episode_rewards[-avg_window:]) / avg_window
            writer.add_scalar("Reward/Moving_Avg", avg_reward, episode)
            print(
                f"Episode {episode} | Reward: {total_reward:.2f} | Avg({avg_window}): {avg_reward:.2f}"
            )
        else:
            print(f"Episode {episode} | Reward: {total_reward:.2f}")

    logging.info("Saving trained model")
    model_filename = f"policy_net_{run_name}.pth"
    torch.save(policy_net.state_dict(), os.path.join(MODEL_DIR, model_filename))
    env.close()

    # Save final metrics
    np.save(os.path.join(METRICS_DIR, "rewards.npy"), np.array(all_episode_rewards))

    # Close TensorBoard writer
    writer.close()

    # Record a video of the trained agent
    logging.info("")

    # Recreate environment for video recording
    env_eval = gym.make(ENV_NAME, render_mode="rgb_array")
    env_eval.unwrapped.configure(config_dict)

    record_agent(
        policy_net,
        env_eval,
        video_folder=os.path.join(SAVE_DIR, "videos"),
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
