import torch

# === Training Control ===
NUM_EPISODES = 500  # Total number of episodes to train the agent
BATCH_SIZE = 64  # Number of experiences sampled from the replay buffer for each optimization step
TARGET_UPDATE = 1_000  # Frequency (in *steps*) of updating the target network weights with the policy network weights

# === Q-Learning Parameters ===
GAMMA = 0.99  # Discount factor for future rewards. How much to value future rewards compared to immediate ones (0 to 1).
LR = 1e-4  # Learning rate for the Adam optimizer. Controls the step size during weight updates.

# === Epsilon-Greedy Exploration ===
EPS_START = 1.0  # Initial value of epsilon (probability of taking a random action). Starts with full exploration.
EPS_END = (
    0.05  # Minimum value of epsilon. Ensures some exploration even late in training.
)
EPS_DECAY = 0.995  # Multiplicative factor by which epsilon is reduced after each episode (e.g., 1.0 -> 0.995 -> 0.990, etc.).

# === Replay Buffer ===
MEMORY_SIZE = 500_000  # Maximum number of experiences (transitions) to store in the replay buffer.

# === Device Configuration ===
# Automatically select CUDA (NVIDIA GPU) if available, otherwise use CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
