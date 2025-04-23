import random
from collections import deque, namedtuple

import numpy as np

# Define a named tuple 'Experience' for better code readability and structure
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """
    A fixed-size replay buffer to store experience tuples.

    Experiences are stored as named tuples `(state, action, reward, next_state, done)`.
    The buffer uses a deque (double-ended queue) for efficient appending and popping
    when the capacity is reached.
    """

    def __init__(self, capacity):
        """
        Initialize the ReplayBuffer.

        Args:
            capacity (int): The maximum number of experiences the buffer can hold.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        # logging.info(f"ReplayBuffer initialized with capacity {capacity}") # Optional: Add logging if desired

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        If the buffer is full, the oldest experience is automatically removed.

        Args:
            state (np.ndarray): The state observed.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state observed.
            done (bool): True if the episode terminated, False otherwise.
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing NumPy arrays for states, actions, rewards,
                   next_states, and dones, corresponding to the sampled batch.
                   Returns None if the buffer contains fewer experiences than batch_size.

        Raises:
            ValueError: If batch_size is larger than the current buffer size.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Requested batch_size ({batch_size}) is larger than current buffer size ({len(self.buffer)})"
            )

        # Randomly sample 'batch_size' experiences from the buffer
        sampled_experiences = random.sample(self.buffer, batch_size)

        # Unzip the batch of experiences into separate lists/tuples
        states, actions, rewards, next_states, dones = zip(*sampled_experiences)

        # Convert the lists into NumPy arrays for efficient processing
        # Ensure correct dtypes, especially for actions (long) and dones (float for calculations)
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)

        return states_np, actions_np, rewards_np, next_states_np, dones_np

    def __len__(self):
        """
        Return the current number of experiences stored in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return len(self.buffer)
