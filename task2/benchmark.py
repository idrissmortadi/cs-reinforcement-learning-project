import os
import pickle
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

# Import your PPO implementation
sys.path.append(".")  # Ensure the directory with your PPO implementation is in path
import ppo  # Your PPO implementation

# Create a directory for saving results
os.makedirs("benchmark_results", exist_ok=True)

# Define benchmark environments
test_envs = ["CartPole-v1"]

# Training parameters
n_episodes = 2_000
eval_interval = 50
eval_episodes = 10


# Function to evaluate the agent's performance
def evaluate_policy(agent, env_name, n_eval_episodes=10):
    env = gym.make(env_name)
    total_rewards = []

    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        if isinstance(state, dict):
            state = np.concatenate([v.flatten() for v in state.values()])
        else:
            state = state.flatten()

        done = False
        episode_reward = 0

        while not done:
            action, _, _ = agent.select_action(state)

            # Handle discrete action spaces differently
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = np.argmax(action)

            next_state, reward, terminated, truncated, _ = env.step(action)

            if isinstance(next_state, dict):
                next_state = np.concatenate([v.flatten() for v in next_state.values()])
            else:
                next_state = next_state.flatten()

            done = terminated or truncated
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards), np.std(total_rewards)


# Function to record agent behavior and save video
def record_agent_behavior(agent, env_name, episode_num, n_steps=1000):
    # Create videos directory if it doesn't exist
    videos_dir = f"benchmark_results/videos/{env_name}"
    os.makedirs(videos_dir, exist_ok=True)

    # Create a wrapped environment for recording
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        videos_dir,
        name_prefix=f"episode_{episode_num}",
        episode_trigger=lambda x: True,  # Record every episode
    )

    state, _ = env.reset()
    if isinstance(state, dict):
        state = np.concatenate([v.flatten() for v in state.values()])
    else:
        state = state.flatten()

    total_reward = 0
    done = False
    step = 0

    while not done and step < n_steps:
        action, _, _ = agent.select_action(state)

        # Handle discrete action spaces differently
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = np.argmax(action)

        next_state, reward, terminated, truncated, _ = env.step(action)

        if isinstance(next_state, dict):
            next_state = np.concatenate([v.flatten() for v in next_state.values()])
        else:
            next_state = next_state.flatten()

        done = terminated or truncated
        total_reward += reward
        state = next_state
        step += 1

    env.close()
    print(f"Recorded video for episode {episode_num}. Total reward: {total_reward:.2f}")
    return total_reward


# Run benchmarks for each environment
results = {}

for env_name in test_envs:
    print(f"\n{'=' * 50}")
    print(f"Benchmarking on {env_name}")
    print(f"{'=' * 50}")

    env = gym.make(env_name)

    # Initialize observation and action dimensions
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_dim = sum(np.prod(space.shape) for space in env.observation_space.values())
    else:
        obs_dim = np.prod(env.observation_space.shape)

    # Handle both continuous and discrete action spaces
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n  # One-hot encoding for discrete actions
    else:
        act_dim = env.action_space.shape[0]

    # Initialize your PPO agent
    agent = ppo.PPOAgent(obs_dim=obs_dim, act_dim=act_dim)

    # Lists to store training progress
    episode_rewards = []
    eval_rewards = []
    eval_stds = []
    eval_episodes_list = []

    # Record initial behavior (before training)
    print("\nRecording initial behavior (untrained agent)...")
    record_agent_behavior(agent, env_name, episode_num=0)

    # Training loop
    progress_bar = tqdm(range(n_episodes), desc=f"Training on {env_name}")
    for episode in progress_bar:
        state, _ = env.reset()
        if isinstance(state, dict):
            state = np.concatenate([v.flatten() for v in state.values()])
        else:
            state = state.flatten()

        episode_reward = 0
        step_counter = 0
        done = False

        while not done:
            action, value, logp = agent.select_action(state)

            # Handle discrete action spaces differently
            if isinstance(env.action_space, gym.spaces.Discrete):
                env_action = np.argmax(action)
            else:
                env_action = action

            next_state, reward, terminated, truncated, _ = env.step(env_action)

            if isinstance(next_state, dict):
                next_state = np.concatenate([v.flatten() for v in next_state.values()])
            else:
                next_state = next_state.flatten()

            # Check if buffer is full before storing
            if agent.buf.ptr >= agent.buf.max_size:
                last_val = 0
                if not done:
                    _, last_val, _ = agent.select_action(state)
                agent.finish_path(last_val)
                agent.update()

            # Store in PPO buffer
            agent.store(state, action, reward, value, logp)

            state = next_state
            episode_reward += reward
            step_counter += 1
            done = terminated or truncated

            # If buffer is full, update - keep this as a backup
            if step_counter % agent.buf.max_size == 0:
                last_val = 0
                if not done:
                    _, last_val, _ = agent.select_action(state)
                agent.finish_path(last_val)
                agent.update()

        # End of episode handling
        if agent.buf.ptr > 0:  # Only finish path if there's data in the buffer
            agent.finish_path(0)
            # Only update if buffer is full
            if agent.buf.ptr == agent.buf.max_size:
                agent.update()

        episode_rewards.append(episode_reward)
        progress_bar.set_postfix(
            {"reward": episode_reward, "avg_reward": np.mean(episode_rewards[-20:])}
        )

        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            mean_reward, std_reward = evaluate_policy(
                agent, env_name, n_eval_episodes=eval_episodes
            )
            eval_rewards.append(mean_reward)
            eval_stds.append(std_reward)
            eval_episodes_list.append(episode + 1)

            print(
                f"\nEvaluation at episode {episode + 1}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}"
            )

            if mean_reward >= 450:
                break

    env.close()

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        agent, env_name, n_eval_episodes=eval_episodes
    )
    print(
        f"\nFinal evaluation on {env_name}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}"
    )

    # Record final behavior (after training)
    print("\nRecording final behavior (trained agent)...")
    record_agent_behavior(agent, env_name, episode_num=n_episodes)

    # Store results
    results[env_name] = {
        "episode_rewards": episode_rewards,
        "eval_rewards": eval_rewards,
        "eval_stds": eval_stds,
        "eval_episodes": eval_episodes_list,
        "final_mean": mean_reward,
        "final_std": std_reward,
    }

    # Plot learning curve
    plt.figure(figsize=(12, 6))

    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title(f"Episode Rewards - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Plot evaluation rewards
    plt.subplot(1, 2, 2)
    plt.errorbar(eval_episodes_list, eval_rewards, yerr=eval_stds, capsize=5)
    plt.title(f"Evaluation Rewards - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")

    plt.tight_layout()
    plt.savefig(f"benchmark_results/{env_name}_learning_curve.png")
    plt.close()

# Save results

with open("benchmark_results/benchmark_results.pkl", "wb") as f:
    pickle.dump(results, f)

# Compare all environments in a single plot
plt.figure(figsize=(10, 6))
for env_name, result in results.items():
    plt.errorbar(
        result["eval_episodes"],
        result["eval_rewards"],
        yerr=result["eval_stds"],
        label=env_name,
        capsize=5,
    )

plt.title("PPO Performance Across Environments")
plt.xlabel("Training Episodes")
plt.ylabel("Mean Evaluation Reward")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("benchmark_results/overall_comparison.png")
plt.close()

# Print summary
print("\n=== Benchmark Summary ===")
for env_name, result in results.items():
    print(f"{env_name}: {result['final_mean']:.2f} ± {result['final_std']:.2f}")

# Add information about recorded videos
print("\n=== Recorded Agent Behaviors ===")
print("Videos showing agent behavior at the beginning and end of training")
print("can be found in the 'benchmark_results/videos/' directory.")
print("Compare these videos to see how the agent's performance has improved.")
