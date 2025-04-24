import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal


class PPOBufferPolicyOnly:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start = 0, 0
        self.max_size = size

    def store(self, obs, act, rew, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self):
        slice_ = slice(self.path_start, self.ptr)
        rews = self.rew_buf[slice_]
        # Compute rewards-to-go (returns)
        self.ret_buf[slice_] = np.array(
            [
                sum(self.gamma**k * rews[k + i] for k in range(len(rews) - i))
                for i in range(len(rews))
            ]
        )
        self.path_start = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start = 0, 0
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class MLPPolicyOnly(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 512)):
        super().__init__()
        # Actor network
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.actor_net = nn.Sequential(*layers)
        self.pi_mu = nn.Linear(prev, act_dim)
        self.pi_logstd = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, obs):
        actor_x = self.actor_net(obs)
        mu = self.pi_mu(actor_x)
        std = torch.exp(self.pi_logstd).clamp(1e-6, 1.0)
        return mu, std


class PPOPolicyOnlyAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_low=-1.0,
        action_high=1.0,
        steps_per_epoch=4000,
        gamma=0.99,
        clip_ratio=0.05,
        pi_lr=1e-5,
        train_iters=128,
        minibatch_size=256,
        entropy_coef=0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MLPPolicyOnly(obs_dim, act_dim).to(self.device)
        self.buf = PPOBufferPolicyOnly(obs_dim, act_dim, steps_per_epoch, gamma)
        self.clip_ratio, self.train_iters = clip_ratio, train_iters
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.action_low = action_low
        self.action_high = action_high

        self.pi_optimizer = optim.Adam(self.policy.parameters(), lr=pi_lr)

    def select_action(self, obs):
        obs_t = torch.as_tensor(
            np.array(obs).flatten(),
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():
            mu, std = self.policy(obs_t)
            pi = Normal(mu, std)
            a = pi.sample()
            # Clip actions to valid range
            a = torch.clamp(a, self.action_low, self.action_high)
            logp = pi.log_prob(a).sum()

        return a.cpu().numpy(), logp.cpu().item()

    def store(self, obs, act, rew, logp):
        self.buf.store(obs, act, rew, logp)

    def finish_path(self):
        self.buf.finish_path()

    def update(self):
        data = self.buf.get()
        obs, act, ret, logp_old = (
            data["obs"].to(self.device),
            data["act"].to(self.device),
            data["ret"].to(self.device),
            data["logp"].to(self.device),
        )

        # Normalize returns to reduce variance
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        for _ in range(self.train_iters):
            idxs = torch.randperm(obs.size(0))
            for start in range(0, obs.size(0), self.minibatch_size):
                b = idxs[start : start + self.minibatch_size]
                mu, std = self.policy(obs[b])
                pi = Normal(mu, std)
                logp = pi.log_prob(act[b]).sum(axis=-1)
                ratio = torch.exp(logp - logp_old[b])
                clip_adv = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * ret[b]
                )

                # Add entropy bonus
                entropy = pi.entropy().mean()
                pi_loss = (
                    -(torch.min(ratio * ret[b], clip_adv)).mean()
                    - self.entropy_coef * entropy
                )

                # Update policy
                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.pi_optimizer.step()

    def save(self, save_path):
        """
        Save the model parameters to a file.

        Args:
            save_path (str): Path to save the model.
        """
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "pi_optimizer_state": self.pi_optimizer.state_dict(),
            },
            save_path,
        )
        print(f"Policy-only PPO model saved to {save_path}")

    def load(self, load_path):
        """
        Load model parameters from a file.

        Args:
            load_path (str): Path to load the model from.
        """
        if not load_path.exists():
            raise FileNotFoundError(f"No model found at {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state"])
        self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer_state"])
        print(f"Policy-only PPO model loaded from {load_path}")
