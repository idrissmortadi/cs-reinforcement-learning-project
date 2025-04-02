import logging
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.task_1_config import ENVIRONEMNT, config_dict

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Create environment using the config
ENV_NAME = ENVIRONEMNT
env = gym.make(ENV_NAME, render_mode="rgb_array")
env.unwrapped.configure(config_dict)


# If observations are dicts (for OccupancyGrid), flatten them
def preprocess_state(state):
    if isinstance(state, dict):
        # Flatten state with sorted keys to ensure consistent ordering
        state = np.concatenate(
            [np.array(state[k]).flatten() for k in sorted(state.keys())]
        )
    return np.array(state, dtype=np.float32)


# DQN Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
NUM_EPISODES = 500

# Determine input dimension from a sample observation
state, _ = env.reset()
state = preprocess_state(state)
input_dim = state.shape[0]

# Determine number of actions (for discrete action space)
n_actions = env.action_space.n


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def select_action(state, policy_net, epsilon):
    logging.debug(f"Selecting action with epsilon: {epsilon}")
    if random.random() < epsilon:
        action = random.randrange(n_actions)
        logging.debug(f"Random action selected: {action}")
        return action
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            logging.debug(f"Greedy action selected: {action}")
            return action


def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states = torch.tensor(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute Q-values for current states
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute Q-values for next states from target network
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, expected_q_values.detach())
    logging.debug(f"Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    policy_net = QNetwork(input_dim, n_actions)
    target_net = QNetwork(input_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPS_START

    for episode in range(1, NUM_EPISODES + 1):
        logging.info(f"--- Episode {episode} started ---")
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_state)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            logging.debug(
                f"Step info: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}"
            )
            optimize_model(policy_net, target_net, optimizer, memory)

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logging.info("Target network updated")

        print(f"Episode {episode} Total Reward: {total_reward} Epsilon: {epsilon:.3f}")

    logging.info("Saving trained model")
    torch.save(policy_net.state_dict(), "dqn_policy_net.pth")
    env.close()


if __name__ == "__main__":
    main()
