import gymnasium as gym
from ddpg import DDPGAgent  # Import the DDPG agent

# Initialize the environment and agent
env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]  # Action limit for scaling
agent = DDPGAgent(obs_dim, act_dim, act_limit)

# Train the agent
for epoch in range(50):
    obs, _ = env.reset()  # Extract the observation from the reset tuple
    obs = obs.flatten()  # Flatten the observation
    epoch_reward = 0  # Initialize cumulative reward for the epoch
    done = False
    while not done:
        act = agent.select_action(obs)  # Select action
        next_obs, rew, done, _, _ = env.step(act)  # Step the environment
        next_obs = next_obs.flatten()  # Flatten the next observation
        agent.store_transition(obs, act, rew, next_obs, done)  # Store transition
        agent.train()  # Train the agent
        obs = next_obs
        epoch_reward += rew  # Accumulate reward
    print(f"Epoch {epoch + 1} completed, Total Reward: {epoch_reward:.2f}")

# Evaluate the trained agent
obs, _ = env.reset()  # Extract the observation from the reset tuple
obs = obs.flatten()  # Flatten the observation
done = False
play_reward = 0  # Initialize cumulative reward for visualization
while not done:
    env.render()  # Render the environment
    act = agent.select_action(obs, deterministic=True)  # Use deterministic actions
    next_obs, rew, done, _, _ = env.step(act)  # Step the environment
    obs = next_obs.flatten()  # Flatten the next observation
    play_reward += rew  # Accumulate reward

print(f"Total Reward during play: {play_reward:.2f}")
env.close()  # Close the environment after visualization
