import datetime
import logging
import os
import random

import numpy as np
import torch
from hyperparameters import (
    BATCH_SIZE,
    EPS_END,
    EPS_START,
    GAMMA,
    LR,
    MEMORY_SIZE,
    device,
)

# Global flags for signal handling
stop_training = False
training_started = False


def signal_handler(sig, frame):
    """
    Handle Ctrl+C signal (SIGINT) to gracefully stop training and save a checkpoint.

    Sets the `stop_training` flag to True if training has already started.
    """
    global stop_training
    global training_started
    # Set stop_training to True only if training has actually begun
    if training_started:
        stop_training = True
        logging.warning("SIGINT received. Training will stop after this episode.")
    else:
        logging.warning("SIGINT received, but training hasn't started yet.")


def get_run_name(custom_name=None):
    """
    Generate a descriptive name for the current experiment run, including hyperparameters and a timestamp.

    Args:
        custom_name (str, optional): A custom prefix for the run name. Defaults to None.

    Returns:
        str: The generated run name (e.g., "DQN_lr0.0001_bs64_gamma0.99_eps1.0-0.05_mem10000_20231027-120000").
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    hyperparam_str = (
        f"lr{LR}_bs{BATCH_SIZE}_gamma{GAMMA}_eps{EPS_START}-{EPS_END}_mem{MEMORY_SIZE}"
    )
    if custom_name:
        return f"{custom_name}_{hyperparam_str}_{timestamp}"
    return f"DQN_{hyperparam_str}_{timestamp}"


def select_action(state, policy_net, epsilon, n_actions):
    """
    Select an action using an epsilon-greedy policy.

    With probability epsilon, selects a random action (exploration).
    Otherwise, selects the action with the highest Q-value predicted by the policy network (exploitation).

    Args:
        state (np.ndarray): The current environment state.
        policy_net (QNetwork): The policy network used for Q-value prediction.
        epsilon (float): The current exploration rate.
        n_actions (int): The total number of possible actions.

    Returns:
        int: The selected action index.
    """
    sample = random.random()
    if sample < epsilon:
        # Exploration: Select a random action
        action = random.randrange(n_actions)
        logging.debug(
            f"Exploration: Random action selected: {action} (epsilon={epsilon:.3f})"
        )
        return action
    else:
        # Exploitation: Select the best action based on Q-values
        with torch.no_grad():
            # Convert state to tensor, add batch dimension, and flatten if necessary
            state_tensor = (
                torch.tensor(state, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .view(1, -1)
            )
            logging.debug(
                f"State tensor shape for Q-value prediction: {state_tensor.shape}"
            )
            q_values = policy_net(state_tensor)
            # Select the action with the maximum Q-value
            action = q_values.argmax().item()
            logging.debug(
                f"Exploitation: Greedy action selected: {action} (epsilon={epsilon:.3f})"
            )
            return action


def preprocess_state(state):
    """
    Preprocess the state observation. If it's a dictionary, flatten it into a NumPy array.

    Ensures the state is always a flat NumPy array of type float32, suitable for network input.

    Args:
        state (Union[dict, np.ndarray]): The raw state observation from the environment.

    Returns:
        np.ndarray: The preprocessed state as a flattened NumPy array (dtype=np.float32).
    """
    if isinstance(state, dict):
        # Flatten dictionary state, ensuring consistent order by sorting keys
        logging.debug("Preprocessing dictionary state.")
        processed_state = np.concatenate(
            [np.array(state[k]).flatten() for k in sorted(state.keys())]
        )
    else:
        # Assume state is already a NumPy array or compatible type
        logging.debug("Preprocessing non-dictionary state.")
        processed_state = np.array(state)

    # Ensure the final state is a flat float32 NumPy array
    return processed_state.flatten().astype(np.float32)


def list_checkpoints(model_dir):
    """
    List all available model checkpoint files (ending with '.pth') in the specified directory.

    Args:
        model_dir (str): The directory containing model checkpoints.

    Returns:
        List[str]: A sorted list of checkpoint filenames found in the directory. Returns an empty list if the directory doesn't exist or contains no checkpoints.
    """
    if not os.path.isdir(model_dir):
        logging.warning(f"Model directory not found: {model_dir}")
        return []
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    checkpoints.sort()  # Ensure consistent order
    logging.info(f"Found {len(checkpoints)} checkpoints in {model_dir}.")
    return checkpoints


def load_checkpoint(policy_net, model_dir):
    """
    Load model weights from a checkpoint file into the policy network.

    Prompts the user to select a checkpoint from the available files in `model_dir`.
    If no checkpoints are found or the user chooses not to load, the original network is returned.

    Args:
        policy_net (QNetwork): The policy network instance to load the weights into.
        model_dir (str): The directory where checkpoint files are stored.

    Returns:
        QNetwork: The policy network, potentially with loaded weights.
    """
    checkpoints = list_checkpoints(model_dir)
    if not checkpoints:
        logging.info(
            f"No checkpoints found in {model_dir}. Starting training from scratch."
        )
        return policy_net

    print("\nAvailable checkpoints:")
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"  {i}. {checkpoint}")
    print("  0. Start from scratch (do not load)")

    while True:
        try:
            choice_str = input(
                f"Select a checkpoint to load [1-{len(checkpoints)}, or 0 to skip]: "
            )
            choice = int(choice_str)

            if 1 <= choice <= len(checkpoints):
                selected_checkpoint = checkpoints[choice - 1]
                checkpoint_path = os.path.join(model_dir, selected_checkpoint)
                logging.info(f"Loading checkpoint: {checkpoint_path}")
                try:
                    # Load the state dict, ensuring it's mapped to the correct device
                    policy_net.load_state_dict(
                        torch.load(checkpoint_path, map_location=device)
                    )
                    logging.info(
                        f"Successfully loaded checkpoint '{selected_checkpoint}'."
                    )
                    return policy_net
                except Exception as e:
                    logging.error(
                        f"Error loading checkpoint {selected_checkpoint}: {e}"
                    )
                    print(
                        f"Error loading checkpoint: {e}. Please try again or select 0."
                    )
            elif choice == 0:
                logging.info("User chose to start from scratch. No checkpoint loaded.")
                return policy_net
            else:
                print(
                    f"Invalid choice '{choice}'. Please enter a number between 0 and {len(checkpoints)}."
                )
        except ValueError:
            print(f"Invalid input '{choice_str}'. Please enter a number.")
        except EOFError:  # Handle Ctrl+D or unexpected end of input
            logging.warning("Input stream closed unexpectedly. Starting from scratch.")
            return policy_net
