import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start = 0, 0
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        slice_ = slice(self.path_start, self.ptr)
        rews = np.append(self.rew_buf[slice_], last_val)
        vals = np.append(self.val_buf[slice_], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = 0
        for i in reversed(range(len(deltas))):
            adv = deltas[i] + self.gamma * self.lam * adv
            self.adv_buf[slice_.start + i] = adv
        # compute rewards-to-go
        self.ret_buf[slice_] = np.array(
            [
                sum(self.gamma**k * rews[k + i] for k in range(len(rews) - i))
                for i in range(len(rews) - 1)
            ]
        )
        self.path_start = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start = 0, 0
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 512)):
        super().__init__()

        # Actor network
        actor_layers = []
        prev = obs_dim
        for h in hidden_sizes:
            actor_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.actor_net = nn.Sequential(*actor_layers)
        self.pi_mu = nn.Linear(prev, act_dim)
        self.pi_logstd = nn.Parameter(-0.5 * torch.ones(act_dim))

        # Critic network
        critic_layers = []
        prev = obs_dim
        for h in hidden_sizes:
            critic_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.critic_net = nn.Sequential(*critic_layers)
        self.v = nn.Linear(prev, 1)

    def pi_parameters(self):
        # Return only policy parameters
        return (
            list(self.actor_net.parameters())
            + list(self.pi_mu.parameters())
            + [self.pi_logstd]
        )

    def v_parameters(self):
        # Return only value function parameters
        return list(self.critic_net.parameters()) + list(self.v.parameters())

    def forward(self, obs):
        # Actor forward pass
        actor_x = self.actor_net(obs)
        mu = self.pi_mu(actor_x)
        std = torch.exp(self.pi_logstd).clamp(1e-6, 1.0)

        # Critic forward pass
        critic_x = self.critic_net(obs)
        v = self.v(critic_x).squeeze(-1)

        return mu, std, v


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_low=-1.0,
        action_high=1.0,
        steps_per_epoch=4000,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.05,
        pi_lr=1e-5,
        vf_lr=1e-4,
        train_iters=128,
        minibatch_size=256,
        entropy_coef=0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = MLPActorCritic(obs_dim, act_dim).to(self.device)
        self.buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
        self.clip_ratio, self.train_iters = clip_ratio, train_iters
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.action_low = action_low
        self.action_high = action_high

        # Separate parameter groups for policy and value networks
        policy_params = self.ac.pi_parameters()
        value_params = self.ac.v_parameters()

        self.pi_optimizer = optim.Adam(policy_params, lr=pi_lr)
        self.vf_optimizer = optim.Adam(value_params, lr=vf_lr)

    def save(self, save_path):
        """
        Save the model parameters to a file.

        Args:
            filename (str): Name of the file (without extension)
        """
        # Save model state
        torch.save(
            {
                "actor_critic_state": self.ac.state_dict(),
                "pi_optimizer_state": self.pi_optimizer.state_dict(),
                "vf_optimizer_state": self.vf_optimizer.state_dict(),
            },
            save_path,
        )

        print(f"Model saved to {save_path}")

    def load(self, load_path):
        """
        Load model parameters from a file.

        Args:
            filename (str): Name of the file (without extension)
        """

        if not load_path.exists():
            raise FileNotFoundError(f"No model found at {load_path}")

        # Load model state
        checkpoint = torch.load(load_path, map_location=self.device)

        # Apply state to models and optimizers
        self.ac.load_state_dict(checkpoint["actor_critic_state"])
        self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer_state"])
        self.vf_optimizer.load_state_dict(checkpoint["vf_optimizer_state"])

        print(f"Model loaded from {load_path}")

    def select_action(self, obs):
        obs_t = torch.as_tensor(
            np.array(obs).flatten(),
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():
            mu, std, v = self.ac(obs_t)
            pi = Normal(mu, std)
            a = pi.sample()
            # Clip actions to valid range
            a = torch.clamp(a, self.action_low, self.action_high)
            logp = pi.log_prob(a).sum()

        return a.cpu().numpy(), v.cpu().item(), logp.cpu().item()

    def store(self, obs, act, rew, val, logp):
        self.buf.store(obs, act, rew, val, logp)

    def finish_path(self, last_val=0):
        self.buf.finish_path(last_val)

    def update(self):
        data = self.buf.get()
        obs, act, ret, adv, logp_old = (
            data["obs"].to(self.device),
            data["act"].to(self.device),
            data["ret"].to(self.device),
            data["adv"].to(self.device),
            data["logp"].to(self.device),
        )

        for _ in range(self.train_iters):
            idxs = torch.randperm(obs.size(0))
            for start in range(0, obs.size(0), self.minibatch_size):
                b = idxs[start : start + self.minibatch_size]
                mu, std, v = self.ac(obs[b])
                pi = Normal(mu, std)
                logp = pi.log_prob(act[b]).sum(axis=-1)
                ratio = torch.exp(logp - logp_old[b])
                clip_adv = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * adv[b]
                )

                # Add entropy bonus
                entropy = pi.entropy().mean()
                pi_loss = (
                    -(torch.min(ratio * adv[b], clip_adv)).mean()
                    - self.entropy_coef * entropy
                )
                v_loss = ((ret[b] - v) ** 2).mean()

                # Compute all gradients before updating any parameters
                self.pi_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                pi_loss.backward()
                v_loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)

                # Update parameters
                self.pi_optimizer.step()
                self.vf_optimizer.step()
