import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Determine the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )
        # Convert numpy arrays to tensors and move to the specified device
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in batch.items()
        }


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class DDPGAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        gamma=0.99,
        tau=0.005,
        lr=1e-3,
        buffer_size=100000,
    ):
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.device = device  # Store device

    def select_action(self, obs, deterministic=False):
        # Move observation to the correct device before converting to tensor
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():  # No need to track gradients for action selection
            act = (
                self.actor(obs_t).cpu().numpy()
            )  # Move action back to CPU for numpy conversion
        if not deterministic:
            act += np.random.normal(0, 0.1, size=act.shape)  # Add exploration noise
        return np.clip(act, -self.actor.act_limit, self.actor.act_limit)

    def store_transition(self, obs, act, rew, next_obs, done):
        # No device changes needed here as buffer stores numpy arrays
        self.replay_buffer.store(obs, act, rew, next_obs, done)

    def train(self, batch_size=64):
        if self.replay_buffer.size < batch_size:
            return
        batch = self.replay_buffer.sample_batch(
            batch_size
        )  # Batch tensors are already on the correct device
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"].unsqueeze(-1)
        next_obs = batch["next_obs"]
        done = batch["done"].unsqueeze(-1)

        # --- Calculate Critic Loss ---
        with torch.no_grad():
            next_act = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_act)
            target_q = rew + self.gamma * (1 - done) * target_q
        q = self.critic(obs, act)
        critic_loss = nn.MSELoss()(q, target_q)

        # --- Update Critic ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Calculate Actor Loss ---
        # Freeze critic parameters during actor loss calculation
        for p in self.critic.parameters():
            p.requires_grad = False

        actor_actions = self.actor(obs)
        actor_loss = -self.critic(obs, actor_actions).mean()

        # --- Update Actor ---
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for p in self.critic.parameters():
            p.requires_grad = True

        # --- Update Target Networks ---
        with torch.no_grad():
            for p, p_target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_target.data.copy_((1 - self.tau) * p_target.data + self.tau * p.data)
            for p, p_target in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                p_target.data.copy_((1 - self.tau) * p_target.data + self.tau * p.data)
