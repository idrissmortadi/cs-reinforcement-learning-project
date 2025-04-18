import logging
import os

import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from hyperparameters import device
from utils import preprocess_state


def record_agent(policy_net, env, video_folder, run_name, num_eval_episodes=5):
    """
    Evaluate the trained agent and record videos of its performance.

    Wraps the environment with video recording and statistics tracking wrappers.
    Runs the agent for a specified number of episodes using the learned policy
    (exploitation mode, no exploration). Saves videos to a run-specific subfolder.

    Args:
        policy_net (QNetwork): The trained policy network (in evaluation mode).
        env (gym.Env): The Gym environment instance (should have `render_mode="rgb_array"`).
        video_folder (str): The base directory to save videos. A subfolder named `run_name` will be created.
        run_name (str): The identifier for the current training run, used for the subfolder name.
        num_eval_episodes (int): The number of evaluation episodes to run and record. Defaults to 5.
    """
    logging.info(f"Starting agent evaluation for {num_eval_episodes} episodes.")
    policy_net.eval()  # Ensure the network is in evaluation mode

    # Create a subdirectory specific to this run for cleaner organization
    run_video_folder = os.path.join(video_folder, run_name)
    os.makedirs(run_video_folder, exist_ok=True)
    logging.info(f"Videos will be saved to: {run_video_folder}")

    # --- Environment Wrappers ---
    # Wrapper to record videos of episodes
    try:
        env = RecordVideo(
            env,
            video_folder=run_video_folder,
            name_prefix=f"eval-{run_name}",  # Include run name in video filenames
            # episode_trigger=lambda ep_id: ep_id < num_eval_episodes, # Record only the specified number
            episode_trigger=lambda ep_id: True,  # Record all episodes run through this wrapper
            disable_logger=True,  # Disable default RecordVideo logger if desired
            fps=16,  # Set frames per second for video playback
        )
        # Wrapper to automatically track episode statistics (return, length)
        env = RecordEpisodeStatistics(env, num_eval_episodes)
        logging.info("Applied RecordVideo and RecordEpisodeStatistics wrappers.")
    except Exception as e:
        logging.error(f"Error applying environment wrappers: {e}", exc_info=True)
        # Attempt to close env even if wrappers failed
        try:
            env.close()
        except Exception:
            pass  # Ignore errors during cleanup
        return  # Stop execution if wrappers fail

    # --- Evaluation Loop ---
    all_rewards = []
    all_lengths = []
    for episode_num in range(num_eval_episodes):
        logging.info(
            f"--- Starting Evaluation Episode {episode_num + 1}/{num_eval_episodes} ---"
        )
        try:
            obs, info = env.reset()
            # Preprocess the initial observation and ensure it's flat
            obs = preprocess_state(obs).flatten()
            logging.debug(f"Initial observation shape: {obs.shape}")

            # --- Input Dimension Check (Optional but Recommended) ---
            # Check if the flattened observation size matches the network's input layer size
            expected_input_dim = policy_net.layer1.in_features
            if obs.size != expected_input_dim:
                logging.error(
                    f"Observation size mismatch! Observation size: {obs.size}, Network input size: {expected_input_dim}."
                )
                # Handle mismatch
                raise ValueError(
                    f"Observation size ({obs.size}) does not match network input dimension ({expected_input_dim}). Check preprocessing."
                )

            done = False
            truncated = False
            episode_reward = 0
            step_count = 0

            while not (done or truncated):
                # Select action greedily using the policy network
                with torch.no_grad():  # Disable gradient calculation for inference
                    # Convert observation to tensor, add batch dim, ensure correct device and shape
                    obs_tensor = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    ).view(1, -1)
                    # Get Q-values from the network
                    q_values = policy_net(obs_tensor)
                    # Choose the action with the highest Q-value
                    action = q_values.argmax().item()
                    logging.debug(
                        f"Step {step_count}: State shape: {obs_tensor.shape}, Action: {action}"
                    )

                # Execute action in the environment
                obs_next_raw, reward, done, truncated, info = env.step(action)

                # Preprocess the next observation
                obs = preprocess_state(obs_next_raw).flatten()

                episode_reward += reward
                step_count += 1
                logging.debug(
                    f"Step {step_count}: Reward: {reward:.3f}, Done: {done}, Truncated: {truncated}"
                )
                # Log the info dictionary if the episode ended to understand why
                if done or truncated:
                    logging.info(f"Episode ended. Info: {info}")

            logging.info(
                f"--- Evaluation Episode {episode_num + 1} Finished --- Reward: {episode_reward:.2f}, Steps: {step_count}"
            )
            # Statistics are automatically collected by the wrapper, access them later

        except Exception as e:
            logging.error(
                f"Error during evaluation episode {episode_num + 1}: {e}", exc_info=True
            )

    # --- Post-Evaluation ---
    try:
        all_rewards = list(env.return_queue)
        logging.info(f"Evaluation Episode Rewards: {all_rewards}")
        print(f"Evaluation Episode Rewards: {all_rewards}")

        all_lengths = list(env.length_queue)
        logging.info(f"Evaluation Episode Lengths: {all_lengths}")
        print(f"Evaluation Episode Lengths: {all_lengths}")

        if all_rewards:
            avg_reward = sum(all_rewards) / len(all_rewards)
            logging.info(
                f"Average Evaluation Reward over {len(all_rewards)} episodes: {avg_reward:.2f}"
            )
            print(f"Average Evaluation Reward: {avg_reward:.2f}")
        if all_lengths:
            avg_length = sum(all_lengths) / len(all_lengths)
            logging.info(
                f"Average Evaluation Length over {len(all_lengths)} episodes: {avg_length:.2f}"
            )
            print(f"Average Evaluation Length: {avg_length:.2f}")

    except Exception as e:
        logging.error(
            f"Error accessing or logging evaluation statistics: {e}", exc_info=True
        )

    # Close the environment (this also finalizes video saving)
    env.close()
    logging.info(f"Video recording complete. Videos saved in: {run_video_folder}")
