import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class NetContinousActions(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_dim, hidden_size, act_dim, is_critic=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )

        self.is_critic = is_critic
        if not self.is_critic:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        if self.is_critic:
            return self.net(obs)
        else:
            mean = self.net(obs)
            std = self.log_std.exp() + 1e-6
            return mean, std


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), size=batch_size, replace=True)
        return [self.memory[idx] for idx in indices]
        # return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
