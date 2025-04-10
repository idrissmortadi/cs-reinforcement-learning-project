"""
Deep Q-Learning implementation for a custom Gymnasium environment.

This script trains a Deep Q-Network (DQN) agent to interact with an environment
and learn an optimal policy. The environment and its configuration are defined
in the `configs.task_1_config` module.

Key Features:
- Preprocessing of state observations.
- Replay buffer for experience replay.
- Epsilon-greedy policy for action selection.
- Target network for stable training.
- Periodic saving of rewards, losses, and the trained model.

Usage:
    python task1.py
"""

import datetime
import logging
import os
import random
import sys

# For accessing the config module;
sys.path.append("..")

from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from torch.utils.tensorboard import SummaryWriter

from configs.task_1_config import ENVIRONEMNT, config_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Create environment using the config
ENV_NAME = ENVIRONEMNT
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
    return np.array(state, dtype=np.float32)


# DQN Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor for future rewards
LR = 1e-3  # Learning rate for the optimizer
EPS_START = 1.0  # Initial value of epsilon for the epsilon-greedy policy
EPS_END = 0.05  # Minimum value of epsilon for the epsilon-greedy policy
EPS_DECAY = 0.995  # Decay rate for epsilon after each episode
TARGET_UPDATE = 10  # Number of episodes between updates of the target network
MEMORY_SIZE = 10000  # Maximum capacity of the replay buffer
NUM_EPISODES = 500  # Total number of episodes for training

# Determine input dimension from a sample observation
state, _ = env.reset()
state = preprocess_state(state)
input_dim = (
    state.flatten().size
)  # Ensure the state is flattened before calculating size
logging.debug(f"Calculated input_dim: {input_dim}")

# Determine number of actions (for discrete action space)
n_actions = env.action_space.n

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


class QNetwork(nn.Module):
    """
    Neural network for approximating the Q-value function.

    Attributes:
        layer1 (nn.Linear): First fully connected layer.
        layer2 (nn.Linear): Second fully connected layer.
        output_layer (nn.Linear): Output layer for Q-values.
    """

    def __init__(self, input_dim, n_actions):
        """
        Initialize the QNetwork.

        Args:
            input_dim (int): Dimension of the input state.
            n_actions (int): Number of possible actions.
        """
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        logging.debug(f"Input to QNetwork forward pass: {x.shape}")
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.

    Attributes:
        buffer (deque): A deque to store experiences with a fixed capacity.
    """

    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Tuple[np.ndarray]: Batch of states, actions, rewards, next_states, and dones.
        """
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: Number of experiences in the buffer.
        """
        return len(self.buffer)


def select_action(state, policy_net, epsilon):
    """
    Select an action using an epsilon-greedy policy.

    Args:
        state (np.ndarray): Current state.
        policy_net (QNetwork): Policy network.
        epsilon (float): Exploration rate.

    Returns:
        int: Selected action.
    """
    logging.debug(f"State shape before converting to tensor: {state.shape}")
    if random.random() < epsilon:
        action = random.randrange(n_actions)
        logging.debug(f"Random action selected: {action}")
        return action
    else:
        with torch.no_grad():
            state_tensor = (
                torch.tensor(state, device=device).unsqueeze(0).view(1, -1)
            )  # Flatten the state
            logging.debug(f"State tensor shape: {state_tensor.shape}")
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            logging.debug(f"Greedy action selected: {action}")
            return action


def optimize_model(policy_net, target_net, optimizer, memory):
    """
    Perform a single optimization step on the policy network.

    Args:
        policy_net (QNetwork): Policy network.
        target_net (QNetwork): Target network.
        optimizer (torch.optim.Optimizer): Optimizer for the policy network.
        memory (ReplayBuffer): Replay buffer.

    Returns:
        float: Loss value if optimization is performed, otherwise None.
    """
    if len(memory) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states = torch.tensor(states, dtype=torch.float32, device=device).view(
        BATCH_SIZE, -1
    )  # Flatten and set dtype
    actions = torch.tensor(
        actions, dtype=torch.long, device=device
    )  # Actions should be long for indexing
    rewards = torch.tensor(
        rewards, dtype=torch.float32, device=device
    )  # Set dtype to float32
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device).view(
        BATCH_SIZE, -1
    )  # Flatten and set dtype
    dones = torch.tensor(
        dones, dtype=torch.float32, device=device
    )  # Set dtype to float32

    # Compute Q-values for current states
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute Q-values for next states from target network
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, expected_q_values.detach())
    loss_value = loss.item()
    logging.debug(f"Loss: {loss_value}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_value


# Create directories for saving models and metrics
SAVE_DIR = "results"
MODEL_DIR = os.path.join(SAVE_DIR, "models")
METRICS_DIR = os.path.join(SAVE_DIR, "metrics")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def record_agent(policy_net, env, video_folder, num_eval_episodes=4):
    """
    Record a video of the trained agent's performance.

    Args:
        policy_net (QNetwork): Trained policy network.
        env (gym.Env): Environment to evaluate the agent.
        video_folder (str): Directory to save the video.
        num_eval_episodes (int): Number of episodes to record.
    """
    # Ensure a unique video folder to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_video_folder = f"{video_folder}_{timestamp}"
    logging.info(f"Recording videos to folder: {unique_video_folder}")
    env = RecordVideo(
        env,
        video_folder=unique_video_folder,
        name_prefix="eval",
        episode_trigger=lambda x: True,
        step_trigger=lambda x: x % 2
        == 0,  # Record every second step to increase framerate
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
                )  # Ensure tensor matches input_dim
                logging.debug(f"Input to QNetwork forward pass: {obs_tensor.shape}")
                action = policy_net(obs_tensor).argmax().item()
                logging.debug(f"Step {step_count}: Action selected: {action}")

            obs, reward, terminated, truncated, info = env.step(action)
            obs = preprocess_state(
                obs
            ).flatten()  # Ensure the next observation is flattened
            total_reward += reward
            step_count += 1
            logging.debug(
                f"Step {step_count}: Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
            )

            episode_over = terminated or truncated

        logging.info(
            f"Episode {episode_num + 1} completed. Total reward: {total_reward}, Steps: {step_count}"
        )

    env.close()
    logging.info(f"Video recording completed. Videos saved to: {unique_video_folder}")
    print(f"Episode time taken: {env.time_queue}")
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
    return f"DQN_lr{LR}_bs{BATCH_SIZE}_gamma{GAMMA}_eps{EPS_START}-{EPS_END}_mem{MEMORY_SIZE}_{timestamp}"


def main():
    """
    Main training loop for the DQN agent.

    - Initializes the environment, networks, optimizer, and replay buffer.
    - Runs episodes to train the agent.
    - Periodically saves metrics and the trained model.
    - Records a video of the trained agent's performance.
    """
    policy_net = QNetwork(input_dim, n_actions).to(device)
    target_net = QNetwork(input_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPS_START

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
            "epsilon_start": EPS_START,
            "epsilon_end": EPS_END,
            "epsilon_decay": EPS_DECAY,
            "target_update": TARGET_UPDATE,
            "memory_size": MEMORY_SIZE,
            "num_episodes": NUM_EPISODES,
        },
        {},
    )

    # Also save the run configuration to a file for reference
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(f"Environment: {ENV_NAME}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Gamma: {GAMMA}\n")
        f.write(f"Epsilon Start: {EPS_START}\n")
        f.write(f"Epsilon End: {EPS_END}\n")
        f.write(f"Epsilon Decay: {EPS_DECAY}\n")
        f.write(f"Target Update: {TARGET_UPDATE}\n")
        f.write(f"Memory Size: {MEMORY_SIZE}\n")
        f.write(f"Episodes: {NUM_EPISODES}\n")

    # For tracking metrics
    all_episode_rewards = []
    all_losses = []
    avg_window = 20  # For moving average

    for episode in range(1, NUM_EPISODES + 1):
        logging.info(f"--- Episode {episode} started ---")
        state, _ = env.reset()
        state = torch.tensor(preprocess_state(state), device=device)
        total_reward = 0
        done = False
        episode_losses = []

        while not done:
            action = select_action(state.cpu().numpy(), policy_net, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(preprocess_state(next_state), device=device)
            memory.push(
                state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done
            )
            state = next_state
            total_reward += reward

            logging.debug(
                f"Step info: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}"
            )

            # Capture loss during optimization
            loss = optimize_model(policy_net, target_net, optimizer, memory)
            if loss is not None:
                episode_losses.append(loss)

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logging.info("Target network updated")

        # Save metrics
        all_episode_rewards.append(total_reward)
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            all_losses.append(avg_loss)
            writer.add_scalar("Loss/Per_Episode", avg_loss, episode)

        writer.add_scalar("Reward/Per_Episode", total_reward, episode)

        # Calculate moving average if possible
        if len(all_episode_rewards) >= avg_window:
            avg_reward = sum(all_episode_rewards[-avg_window:]) / avg_window
            writer.add_scalar("Reward/Moving_Avg", avg_reward, episode)
            print(
                f"Episode {episode} | Reward: {total_reward:.2f} | Avg({avg_window}): {avg_reward:.2f} | Epsilon: {epsilon:.3f}"
            )
        else:
            print(
                f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}"
            )

        # Optional: Save metrics periodically
        if episode % 50 == 0:
            np.save(
                os.path.join(METRICS_DIR, "rewards.npy"), np.array(all_episode_rewards)
            )
            if all_losses:
                np.save(os.path.join(METRICS_DIR, "losses.npy"), np.array(all_losses))

    logging.info("Saving trained model")
    torch.save(policy_net.state_dict(), os.path.join(MODEL_DIR, "dqn_policy_net.pth"))
    env.close()

    # Save final metrics
    np.save(os.path.join(METRICS_DIR, "rewards.npy"), np.array(all_episode_rewards))
    if all_losses:
        np.save(os.path.join(METRICS_DIR, "losses.npy"), np.array(all_losses))

    # Close TensorBoard writer
    writer.close()

    # Record a video of the trained agent
    logging.info("Recording agent's performance")

    # Recreate environment for video recording

    # Create environment using the config
    env_eval = gym.make(ENV_NAME, render_mode="rgb_array")
    env_eval.unwrapped.configure(config_dict)

    record_agent(
        policy_net,
        env_eval,
        video_folder=os.path.join(SAVE_DIR, "videos"),
    )


if __name__ == "__main__":
    main()
