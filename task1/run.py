"""
Main script for training a Deep Q-Network (DQN) agent on a Highway-Env environment.

This script orchestrates the training process, including:
- Environment setup using configurations from `configs.task_1_config`.
- Initialization of the policy and target Q-networks, optimizer, and replay buffer.
- Handling of potential checkpoint loading to resume training.
- The main training loop, involving interaction with the environment, action selection (epsilon-greedy),
  storing experiences, and optimizing the policy network.
- Periodic updates of the target network.
- Logging of training progress (rewards, losses, epsilon) using TensorBoard and console output.
- Saving the final trained model and training metrics.
- Recording evaluation episodes of the trained agent.
"""

import logging
import os
import sys

# Add project root to Python path for module access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from hyperparameters import (
    BATCH_SIZE,
    EPS_DECAY,
    EPS_END,
    EPS_START,
    GAMMA,
    LR,
    MEMORY_SIZE,
    NUM_EPISODES,
    TARGET_UPDATE,
    device,
)
from optimize_model import optimize_model
from q_network import QNetwork
from record_agent import record_agent
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# Import utility functions
from utils import (
    get_run_name,
    load_checkpoint,
    preprocess_state,
    select_action,
    stop_training,
)

# Import environment configuration
from configs.task_1_config import ENVIRONEMNT, config_dict

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,  # Set default logging level to INFO
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to standard output
    ],
)
logging.info("--- Starting DQN Training Script ---")

# --- Environment Setup ---
try:
    ENV_NAME = ENVIRONEMNT
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env.unwrapped.configure(config_dict)
    logging.info(f"Successfully created and configured environment: {ENV_NAME}")
except Exception as e:
    logging.error(f"Failed to create or configure environment '{ENV_NAME}': {e}")
    sys.exit(1)


# --- Determine State and Action Dimensions ---
try:
    state, _ = env.reset()
    # Preprocess the initial state to get the correct shape
    processed_state = preprocess_state(state)
    # The input dimension for the Q-network is the size of the flattened state vector
    input_dim = processed_state.size
    logging.info(f"State preprocessed. Input dimension for Q-network: {input_dim}")

    # Get the number of discrete actions from the environment's action space
    n_actions = env.action_space.n
    logging.info(f"Number of discrete actions: {n_actions}")
except Exception as e:
    logging.error(f"Failed to get state/action dimensions from environment: {e}")
    env.close()
    sys.exit(1)

# --- Device Selection ---
logging.info(f"Using device: {device}")
if device == "cuda":
    logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# --- Directory Setup ---
SAVE_DIR = "results"
MODEL_DIR = os.path.join(SAVE_DIR, "models")
METRICS_DIR = os.path.join(SAVE_DIR, "metrics")
VIDEO_DIR = os.path.join(SAVE_DIR, "videos")
TENSORBOARD_DIR = os.path.join(SAVE_DIR, "tensorboard")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
logging.info(
    f"Ensured directories exist: {MODEL_DIR}, {METRICS_DIR}, {VIDEO_DIR}, {TENSORBOARD_DIR}"
)


# --- Main Training Function ---
def main():
    """
    Main function to run the DQN training loop.

    Initializes networks, optimizer, memory buffer, and TensorBoard.
    Handles checkpoint loading. Executes the training episodes, performs optimization,
    updates the target network, logs progress, and saves results. Finally, records
    evaluation episodes.
    """
    # Use global flags defined in utils.py for signal handling
    global training_started
    global stop_training

    # --- Initialization ---
    logging.info("Initializing networks, optimizer, and replay buffer...")
    # Policy network: learns the Q-values and decides actions
    policy_net = QNetwork(input_dim, n_actions).to(device)
    # Target network: provides stable targets for Q-value updates
    target_net = QNetwork(input_dim, n_actions).to(device)

    # Load checkpoint if available and user chooses to
    policy_net = load_checkpoint(policy_net, MODEL_DIR)

    # Initialize target network with the same weights as the policy network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Set target network to evaluation mode (no gradient calculation)
    logging.info("Policy and Target networks initialized.")

    # Optimizer for updating the policy network's weights
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    logging.info(f"Optimizer initialized: Adam with LR={LR}")

    # Replay buffer to store experiences (state, action, reward, next_state, done)
    memory = ReplayBuffer(MEMORY_SIZE)
    logging.info(f"Replay buffer initialized with capacity: {MEMORY_SIZE}")

    # Initial exploration rate
    epsilon = EPS_START

    # --- TensorBoard Setup ---
    # Ask user for a custom run name (optional)
    custom_run_name = input(
        "Enter a custom run name (or press Enter for default): "
    ).strip()
    run_name = get_run_name(custom_run_name if custom_run_name else None)
    log_dir = os.path.join(TENSORBOARD_DIR, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logging initialized. Run name: {run_name}")
    logging.info(f"Logs will be saved to: {log_dir}")
    print(f"--- TensorBoard Run Name: {run_name} ---")  # Also print for easy copying

    # Log hyperparameters to TensorBoard for tracking experiment setup
    hparams = {
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "epsilon_start": EPS_START,
        "epsilon_end": EPS_END,
        "epsilon_decay": EPS_DECAY,
        "target_update_freq": TARGET_UPDATE,
        "memory_size": MEMORY_SIZE,
        "num_episodes": NUM_EPISODES,
        "environment": ENV_NAME,
    }
    # Using add_text to log hyperparameters as markdown for better readability in TensorBoard
    hparams_table = "| Hyperparameter | Value |\n|---|---|\n"
    for key, value in hparams.items():
        hparams_table += f"| {key} | {value} |\n"
    writer.add_text("Hyperparameters", hparams_table, 0)
    logging.info("Logged hyperparameters to TensorBoard.")

    # Save the run configuration details to a text file within the log directory
    config_log_path = os.path.join(log_dir, "config_log.txt")
    try:
        with open(config_log_path, "w") as f:
            f.write(f"Run Name: {run_name}\n")
            f.write(f"Environment: {ENV_NAME}\n")
            f.write("--- Hyperparameters ---\n")
            for key, value in hparams.items():
                f.write(f"{key}: {value}\n")
            f.write("\n--- Environment Configuration ---\n")
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")
        logging.info(f"Saved run configuration to {config_log_path}")
    except IOError as e:
        logging.error(f"Failed to write configuration log file: {e}")

    # --- Training Loop ---
    logging.info(f"--- Starting Training for {NUM_EPISODES} Episodes ---")
    all_episode_rewards = []
    all_avg_losses = []  # Store average loss per episode
    avg_reward_window = 50  # Window size for calculating moving average reward
    total_steps = 0  # Global step counter across all episodes

    training_started = True  # Set flag now that loop is starting

    for episode in range(1, NUM_EPISODES + 1):
        # --- Check for Interrupt Signal ---
        if stop_training:
            logging.warning(
                f"Training interrupted at the beginning of episode {episode}. Saving checkpoint..."
            )
            checkpoint_path = os.path.join(
                MODEL_DIR, f"checkpoint_interrupt_ep{episode - 1}_{run_name}.pth"
            )
            try:
                torch.save(policy_net.state_dict(), checkpoint_path)
                logging.info(f"Checkpoint saved due to interrupt: {checkpoint_path}")
            except Exception as e:
                logging.error(f"Failed to save interrupt checkpoint: {e}")
            break  # Exit the training loop

        # --- Episode Start ---
        state, _ = env.reset()
        state = torch.tensor(
            preprocess_state(state), dtype=torch.float32, device=device
        )
        episode_reward = 0
        episode_losses = []  # Track losses within this episode
        done = False
        episode_steps = 0
        logging.info(f"--- Episode {episode}/{NUM_EPISODES} Started ---")

        # --- Inner Episode Loop (Steps) ---
        while not done:
            # Select action using epsilon-greedy policy
            action = select_action(state.cpu().numpy(), policy_net, epsilon, n_actions)

            # Execute action in the environment
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Episode ends if terminated or truncated

            # Preprocess the next state
            next_state = torch.tensor(
                preprocess_state(next_state_raw), dtype=torch.float32, device=device
            )

            # Store experience in replay buffer
            # Convert tensors back to numpy for storage if needed by buffer implementation
            memory.push(
                state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done
            )

            # Move to the next state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # --- Optimize Model ---
            # Perform one step of optimization on the policy network
            loss = optimize_model(
                policy_net, target_net, optimizer, memory, writer, total_steps, logging
            )
            if loss is not None:
                episode_losses.append(loss)
                # Log loss per step to TensorBoard
                writer.add_scalar("Loss/Per_Step", loss, total_steps)

            # --- Target Network Update ---
            # Update the target network every TARGET_UPDATE steps
            if total_steps % TARGET_UPDATE == 0:
                logging.info(f"Updating target network at step {total_steps}")
                target_net.load_state_dict(policy_net.state_dict())

            # Check for interrupt signal within the episode loop as well
            if stop_training:
                logging.warning(
                    "Interrupt signal detected during episode step. Finishing episode..."
                )
                # Don't break here, let the episode finish naturally or by the outer loop check

        # --- Episode End ---
        # Decay epsilon after each episode
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        writer.add_scalar("Progress/Epsilon", epsilon, episode)

        # Log episode metrics
        all_episode_rewards.append(episode_reward)
        writer.add_scalar("Reward/Per_Episode", episode_reward, episode)
        writer.add_scalar("Progress/Episode_Length", episode_steps, episode)

        # Calculate and log average loss for the episode
        avg_loss = 0.0  # Initialize avg_loss
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            all_avg_losses.append(avg_loss)
            writer.add_scalar("Loss/Average_Per_Episode", avg_loss, episode)
            logging.debug(f"Episode {episode}: Average Loss = {avg_loss:.4f}")
        else:
            all_avg_losses.append(
                avg_loss  # Append the initialized value (0.0)
            )  # Append 0 if no optimization steps occurred (e.g., buffer not full)

        # Calculate and log moving average reward
        # Prepare the average loss string for logging
        avg_loss_str = f"{avg_loss:.4f}" if episode_losses else "N/A"

        if episode >= avg_reward_window:
            avg_reward = np.mean(all_episode_rewards[-avg_reward_window:])
            writer.add_scalar(
                f"Reward/Moving_Avg_{avg_reward_window}", avg_reward, episode
            )
            # Use the pre-formatted avg_loss_str
            logging.info(
                f"Episode {episode}/{NUM_EPISODES} | Steps: {episode_steps} | Reward: {episode_reward:.2f} | Avg Reward ({avg_reward_window}): {avg_reward:.2f} | Avg Loss: {avg_loss_str} | Epsilon: {epsilon:.3f}"
            )
        else:
            # Use the pre-formatted avg_loss_str
            logging.info(
                f"Episode {episode}/{NUM_EPISODES} | Steps: {episode_steps} | Reward: {episode_reward:.2f} | Avg Loss: {avg_loss_str} | Epsilon: {epsilon:.3f}"
            )

    # --- End of Training ---
    training_started = False  # Reset flag
    logging.info("--- Training Finished ---")

    # --- Save Final Model ---
    final_model_filename = f"dqn_policy_net_{run_name}_final.pth"
    final_model_path = os.path.join(MODEL_DIR, final_model_filename)
    try:
        torch.save(policy_net.state_dict(), final_model_path)
        logging.info(f"Saved final trained model to: {final_model_path}")
    except Exception as e:
        logging.error(f"Failed to save final model: {e}")

    # Close the environment
    env.close()
    logging.info("Training environment closed.")

    # --- Save Final Metrics ---
    try:
        rewards_path = os.path.join(METRICS_DIR, f"rewards_{run_name}.npy")
        losses_path = os.path.join(METRICS_DIR, f"losses_{run_name}.npy")
        np.save(rewards_path, np.array(all_episode_rewards))
        np.save(losses_path, np.array(all_avg_losses))
        logging.info(f"Saved final rewards to: {rewards_path}")
        logging.info(f"Saved final losses to: {losses_path}")
    except IOError as e:
        logging.error(f"Failed to save metrics files: {e}")

    # Close TensorBoard writer
    writer.close()
    logging.info("TensorBoard writer closed.")

    # --- Record Agent Performance ---
    logging.info("--- Starting Agent Evaluation and Recording ---")
    try:
        # Create a separate environment instance for evaluation/recording
        env_eval = gym.make(ENV_NAME, render_mode="rgb_array")
        env_eval.unwrapped.configure(config_dict)
        logging.info(f"Created evaluation environment: {ENV_NAME}")

        record_agent(
            policy_net=policy_net,
            env=env_eval,
            video_folder=VIDEO_DIR,
            run_name=run_name,
            num_eval_episodes=5,
        )
        env_eval.close()
        logging.info("Evaluation environment closed.")
    except Exception as e:
        logging.error(f"An error occurred during agent recording: {e}", exc_info=True)

    logging.info("--- Script Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)
        sys.exit(1)
