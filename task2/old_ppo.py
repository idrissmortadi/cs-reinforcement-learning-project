import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from net import NetContinousActions


class PPO:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        hidden_size,
        episode_batch_size,
        actor_learning_rate,
        critic_learning_rate,
        lambda_=0.95,
        writer=None,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = 0.05

        self.episode_batch_size = episode_batch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        self.loss_function = nn.MSELoss()
        self.writer = writer

        obs_size = np.prod(self.observation_space.shape)
        actions_dim = self.action_space.shape[0]

        self.actor = NetContinousActions(obs_size, hidden_size, actions_dim)
        self.critic = NetContinousActions(obs_size, hidden_size, 1, is_critic=True)

        self.optimizer = optim.Adam(
            params=self.actor.parameters(), lr=self.actor_learning_rate
        )

        self.critic_optimizer = optim.Adam(
            params=self.critic.parameters(), lr=self.critic_learning_rate
        )
        self.current_episode = []
        self.episode_reward = 0

        self.scores = []

        self.n_eps = 0
        self.total_steps = 0
        self.critic_updates = 0

    def sample_and_clamp_action(self, state_tensor):
        """
        Compute mean and std from the actor, sample an action from a normal distribution,
        and clamp it within the action space bounds.

        Args:
            state_tensor (torch.Tensor): The input state tensor.

        Returns:
            torch.distributions.Normal: The distribution object.
            torch.Tensor: Clamped action.
        """
        mean, std = self.actor(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(
            action, self.action_space.low[0], self.action_space.high[0]
        )
        return dist, action

    def get_action(self, state, epsilon=None):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # === Check for NaNs ===
            mean, std = self.actor(state_tensor)
            if torch.isnan(mean).any():
                raise ValueError("NaN detected in actor 'mean' output")
            if torch.isnan(std).any():
                raise ValueError("NaN detected in actor 'std' output")
            # =======================

            # Clamp the standard deviation (adjust range if needed)
            std = torch.clamp(std, min=1e-4, max=1.0)  # Slightly tighter max clamp

            # === Check std again after clamp ===
            if torch.isnan(std).any():
                raise ValueError("NaN detected in 'std' after clamping")
            # ===================================

            _, action = self.sample_and_clamp_action(state_tensor)
            return action.squeeze(0).numpy()

    def compute_GAE(self, rewards, terminateds, advantages):
        """
        Generalized Advantage Estimation
        """
        GAE = 0
        GAE_list = []
        for t in reversed(range(len(rewards))):
            GAE = (1 - terminateds[t]) * GAE
            GAE = advantages[t] + self.gamma * self.lambda_ * GAE
            GAE_list.append(GAE)
        GAEs = torch.tensor(GAE_list[::-1], dtype=torch.float32)

        if torch.isnan(GAEs).any():
            print(f"NaN detected in GAEs before normalization: {GAEs}")
            # Decide how to handle NaN GAEs, e.g., return zeros or skip update
            # For now, let's prevent normalization if NaN is present
            return GAEs

        # Avoid normalization if GAEs has size 1 or less
        if GAEs.size(0) > 1:
            GAEs = (GAEs - GAEs.mean()) / (GAEs.std() + 1e-8)

        return GAEs

    def compute_ppo_score(self):
        states, actions, rewards, terminals, next_states, old_log_probs = tuple(
            [torch.cat(data) for data in zip(*self.current_episode)]
        )

        with torch.no_grad():
            target_values = (
                rewards
                + self.gamma * (1 - terminals) * self.critic(next_states).squeeze()
            )
            values = self.critic(states).squeeze()
            advantages = target_values - values

        GAEs = self.compute_GAE(rewards, terminals, advantages)

        # Avoid normalization if GAEs has size 1 or less
        # if GAEs.size(0) > 1:
        #     GAEs = (GAEs - GAEs.mean()) / (GAEs.std() + 1e-8)  # Avoid div by 0

        dist, _ = self.sample_and_clamp_action(states)

        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        ratio = torch.exp(log_probs - old_log_probs)

        clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        ppo_clip_obj = torch.min(ratio * GAEs, clipped_ratio * GAEs)

        if self.writer:
            entropy = dist.entropy().sum(dim=-1).mean()
            self.writer.add_scalar("policy/entropy", entropy.item(), self.n_eps)

        return ppo_clip_obj.sum().unsqueeze(0)

    def train_reset(self):
        self.current_episode = []
        self.episode_reward = 0
        self.scores = []

    def update_critic(self, transition):
        state, _, reward, terminated, next_state, _ = transition

        values = self.critic.forward(state)
        with torch.no_grad():
            next_state_values = (1 - terminated) * self.critic.forward(next_state)
            targets = next_state_values * self.gamma + reward

        loss = self.loss_function(values, targets)
        if self.writer:
            self.writer.add_scalar("loss/critic", loss.item(), self.total_steps)

        self.critic_optimizer.zero_grad()
        loss.backward()

        # === Add Gradient Clipping for Critic ===
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        # ========================================

        self.critic_optimizer.step()

    def update(self, state, action, reward, terminated, next_state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            dist, _ = self.sample_and_clamp_action(state_tensor)
            old_log_probs = dist.log_prob(torch.tensor(action, dtype=torch.float32))
            old_log_probs = old_log_probs.sum(dim=-1, keepdim=True)

        transition = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor(action, dtype=torch.float32).unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
            old_log_probs,
        )

        self.total_steps += 1
        self.episode_reward += reward

        self.current_episode.append(transition)
        self.update_critic(transition)

        if terminated:
            self.writer.add_scalar("policy/reward", self.episode_reward, self.n_eps)
            self.episode_reward = 0
            self.n_eps += 1

            self.scores.append(self.compute_ppo_score())
            self.current_episode = []

            if (self.n_eps % self.episode_batch_size) == 0:
                self.optimizer.zero_grad()
                full_neg_score = -torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()

                # === Add Gradient Clipping for Actor ===
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                # =======================================

                self.optimizer.step()
                if self.writer:
                    self.writer.add_scalar(
                        "loss/actor", full_neg_score.item(), self.n_eps
                    )

                self.scores = []

    def sample_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, std = self.actor(state_tensor)  # Get mean and std from the actor
        _, action = self.sample_and_clamp_action(state_tensor)
        return action

    def save(self, save_path):
        """
        Save the PPO model, including actor, critic, optimizers, and training state.

        Args:
            save_path (str): Path to save the model checkpoint.
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "n_eps": self.n_eps,
            "total_steps": self.total_steps,
            "episode_batch_size": self.episode_batch_size,
            "scores": self.scores,
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path):
        """
        Load the PPO model, including actor, critic, optimizers, and training state.

        Args:
            load_path (str): Path to the saved model checkpoint.
        """
        checkpoint = torch.load(load_path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.n_eps = checkpoint["n_eps"]
        self.total_steps = checkpoint["total_steps"]
        self.episode_batch_size = checkpoint["episode_batch_size"]
        self.scores = checkpoint["scores"]
        print(f"Model loaded from {load_path}")
