import numpy as np
import torch
from torch import nn, optim
from torch.distributions import MultivariateNormal


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

        # Instead of individual logstd, output elements of a triangular matrix
        self.act_dim = act_dim
        tril_size = int(act_dim * (act_dim + 1) / 2)
        self.pi_tril = nn.Linear(prev, tril_size)

        # Initialize triangular matrix with reasonable values
        self.pi_tril.weight.data.zero_()
        self.pi_tril.bias.data.zero_()
        # Initialize diagonals to -0.5
        for i in range(act_dim):
            idx = int(i * (i + 1) / 2 + i)  # Index of diagonal element
            if idx < tril_size:
                self.pi_tril.bias.data[idx] = -0.5

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
            + list(self.pi_tril.parameters())
        )

    def v_parameters(self):
        # Return only value function parameters
        return list(self.critic_net.parameters()) + list(self.v.parameters())

    def _get_covariance_matrix(self, tril_elements):
        """
        Constructs a covariance matrix from lower triangular elements
        """
        batch_size = tril_elements.size(0)
        L = torch.zeros(
            batch_size, self.act_dim, self.act_dim, device=tril_elements.device
        )

        # Fill in the lower triangular part
        indices = torch.tril_indices(self.act_dim, self.act_dim)
        L[:, indices[0], indices[1]] = tril_elements

        # Add small value to diagonal for numerical stability
        L[:, range(self.act_dim), range(self.act_dim)] += 1e-6

        # Construct covariance matrix: L * L^T
        cov = L @ L.transpose(1, 2)

        # Ensure reasonable bounds on the variances
        cov = torch.clamp(cov, min=1e-6, max=10.0)

        return cov

    def forward(self, obs):
        # Actor forward pass
        actor_x = self.actor_net(obs)
        mu = self.pi_mu(actor_x)

        # Get lower triangular elements for covariance matrix
        tril_elements = self.pi_tril(actor_x)

        # Construct the covariance matrix
        cov = self._get_covariance_matrix(tril_elements)

        # Critic forward pass
        critic_x = self.critic_net(obs)
        v = self.v(critic_x).squeeze(-1)

        return mu, cov, v


class LSTMActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, num_layers=1):
        super().__init__()

        # Actor network with LSTM
        self.actor_lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.pi_mu = nn.Linear(hidden_size, act_dim)

        # Instead of independent logstd, output elements of a triangular matrix for covariance
        # We need n*(n+1)/2 elements for an nxn lower triangular matrix
        self.act_dim = act_dim
        tril_size = int(act_dim * (act_dim + 1) / 2)
        self.pi_tril = nn.Linear(hidden_size, tril_size)

        # Initialize with reasonable values (negative on diagonal means small variance)
        self.pi_tril.weight.data.zero_()
        self.pi_tril.bias.data.zero_()
        # Initialize diagonals to -0.5
        for i in range(act_dim):
            idx = int(i * (i + 1) / 2 + i)  # Index of diagonal element
            if idx < tril_size:
                self.pi_tril.bias.data[idx] = -0.5

        # Critic network with LSTM
        self.critic_lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.v = nn.Linear(hidden_size, 1)

        # Initialize hidden states
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reset_hidden_states()

    def reset_hidden_states(self, batch_size=1):
        # Reset hidden states (useful at the beginning of episodes)
        device = next(self.parameters()).device
        # Ensure hidden states are on the same device as model parameters
        self.actor_hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )
        self.critic_hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )

    def pi_parameters(self):
        # Return only policy parameters
        return (
            list(self.actor_lstm.parameters())
            + list(self.pi_mu.parameters())
            + list(self.pi_tril.parameters())
        )

    def v_parameters(self):
        # Return only value function parameters
        return list(self.critic_lstm.parameters()) + list(self.v.parameters())

    def _get_covariance_matrix(self, tril_elements):
        batch_size, m = tril_elements.shape
        D = self.act_dim
        L = tril_elements.new_zeros(batch_size, D, D)

        idx = torch.tril_indices(D, D, device=tril_elements.device)
        L[:, idx[0], idx[1]] = tril_elements
        L[:, range(D), range(D)] += 1e-6  # numeric stability on diag

        cov = L @ L.transpose(-1, -2)

        # clamp variances only:
        diag = cov[:, range(D), range(D)]
        diag = diag.clamp(min=1e-6, max=10.0)
        cov[:, range(D), range(D)] = diag

        return cov

    def forward(self, obs, reset=False):
        # Reshape input if it's not batched
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Add sequence dimension if not present
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [batch_size, seq_len=1, obs_dim]

        batch_size = obs.size(0)
        device = obs.device

        # Reset hidden states if requested or if batch size has changed
        if reset or self.actor_hidden[0].size(1) != batch_size:
            self.reset_hidden_states(batch_size)

        # Ensure hidden states are on the same device as input
        if self.actor_hidden[0].device != device:
            self.actor_hidden = (
                self.actor_hidden[0].to(device),
                self.actor_hidden[1].to(device),
            )
        if self.critic_hidden[0].device != device:
            self.critic_hidden = (
                self.critic_hidden[0].to(device),
                self.critic_hidden[1].to(device),
            )

        # Actor forward pass with LSTM
        actor_out, self.actor_hidden = self.actor_lstm(obs, self.actor_hidden)
        mu = self.pi_mu(actor_out[:, -1])  # Get output from last timestep

        # Get lower triangular elements for covariance matrix
        tril_elements = self.pi_tril(actor_out[:, -1])

        # Construct the covariance matrix
        cov = self._get_covariance_matrix(tril_elements)

        # Critic forward pass with LSTM
        critic_out, self.critic_hidden = self.critic_lstm(obs, self.critic_hidden)
        v = self.v(critic_out[:, -1]).squeeze(-1)  # Get output from last timestep

        return mu, cov, v


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_low=-1.0,
        action_high=1.0,
        steps_per_epoch=4_000,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.05,
        pi_lr=1e-5,
        vf_lr=1e-4,
        train_iters=128,
        minibatch_size=256,
        entropy_coef=0.01,
        model_type="lstm",  # "mlp" or "lstm"
        hidden_sizes=(512, 512),  # For MLP
        lstm_hidden_size=256,  # For LSTM
        lstm_num_layers=1,  # For LSTM
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Choose between MLP and LSTM actor-critic models
        self.model_type = model_type.lower()
        if self.model_type == "mlp":
            self.ac = MLPActorCritic(obs_dim, act_dim, hidden_sizes).to(self.device)
        elif self.model_type == "lstm":
            self.ac = LSTMActorCritic(
                obs_dim, act_dim, lstm_hidden_size, lstm_num_layers
            ).to(self.device)
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Choose 'mlp' or 'lstm'"
            )

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

    def select_action(self, obs, reset_lstm=False):
        # Ensure obs is correctly formatted
        obs_t = torch.as_tensor(
            np.array(obs).flatten(),
            dtype=torch.float32,
        ).to(self.device)

        # Always add batch dimension for consistent processing
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            if self.model_type == "lstm":
                mu, cov, v = self.ac(obs_t, reset=reset_lstm)
                pi = MultivariateNormal(mu, cov)
            else:
                # Updated to use covariance matrix instead of std
                mu, cov, v = self.ac(obs_t)
                pi = MultivariateNormal(mu, cov)

            a = pi.sample()
            # Clip actions to valid range
            a = torch.clamp(a, self.action_low, self.action_high)
            logp = pi.log_prob(a)  # No need for sum with MultivariateNormal

        return a.squeeze(0).cpu().numpy(), v.cpu().item(), logp.cpu().item()

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
            # For MLP, shuffle data. For LSTM, keep sequential order
            if self.model_type == "mlp":
                idxs = torch.randperm(obs.size(0))
            else:
                # For LSTM, process in sequential order to maintain temporal dependencies
                idxs = torch.arange(obs.size(0))

            for start in range(0, obs.size(0), self.minibatch_size):
                b = idxs[start : start + self.minibatch_size]

                # For LSTM, reset hidden states for each minibatch
                if self.model_type == "lstm":
                    self.ac.reset_hidden_states(batch_size=len(b))
                    mu, cov, v = self.ac(obs[b])
                    pi = MultivariateNormal(mu, cov)
                else:
                    mu, std, v = self.ac(obs[b])
                    pi = MultivariateNormal(mu, std)

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
