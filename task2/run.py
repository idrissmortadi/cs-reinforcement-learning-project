import argparse
import logging
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import ppo
import torch

# For accessing the config module;
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.task_2_config import ENVIRONMENT, config_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def train(
    env: gym.Env,
    agent: ppo.PPOAgent,
    N_episodes: int,
    writer: SummaryWriter,
    run_name: str,
    reward_threshold: int = 400,
    record_every_ep: int = 8,
    checkpoint_every_ep: int = 50,
    eval_every_ep: int = 20,
    patience: int = 512,
):
    total_env_steps = 0
    all_episode_rewards = []
    episode_pbar = tqdm(range(N_episodes))

    # Early stopping variables
    best_avg_reward = float("-inf")
    episodes_without_improvement = 0

    os.makedirs(f"results/models/{run_name}", exist_ok=True)

    for ep in episode_pbar:
        state, _ = env.reset()
        state = state.flatten()
        ep_reward = 0
        ep_steps = 0
        ep_speed = 0
        ep_crash = False
        ep_action_sum = np.zeros(env.action_space.shape)
        ep_coll = ep_lane = ep_high = ep_onroad = 0

        done = False
        episode_pbar.set_description("Collecting observations...")
        while not done:
            action, value, logp = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            if info["rewards"].get("on_road_reward", 1) == 0:
                reward += -1.0

            agent.store(state, action, reward, value, logp)

            ep_reward += reward
            ep_steps += 1
            ep_speed += info.get("speed", 0)
            if info.get("crashed", False):
                ep_crash = True
            ep_action_sum += action
            rinfo = info.get("rewards", {})
            ep_coll += rinfo.get("collision_reward", 0)
            ep_lane += rinfo.get("right_lane_reward", 0)
            ep_high += rinfo.get("high_speed_reward", 0)
            ep_onroad += rinfo.get("on_road_reward", 0)

            state = next_state
            total_env_steps += 1

            if done:
                last_val = 0
                agent.finish_path(last_val)

            if total_env_steps % agent.buf.max_size == 0:
                agent.finish_path(last_val if not done else 0)
                episode_pbar.set_description("Updating agent...")
                agent.update()

        all_episode_rewards.append(ep_reward)
        moving_avg = (
            np.mean(all_episode_rewards[-20:])
            if len(all_episode_rewards) >= 20
            else np.mean(all_episode_rewards)
        )
        episode_pbar.set_postfix(moving_avg=moving_avg)

        writer.add_scalar("Episode/TotalReward", ep_reward, ep + 1)
        writer.add_scalar("Episode/Length", ep_steps, ep + 1)
        if ep_steps:
            writer.add_scalar("Episode/AvgSpeed", ep_speed / ep_steps, ep + 1)
            writer.add_scalar("Episode/Crashed", int(ep_crash), ep + 1)
            avg_act = ep_action_sum / ep_steps
            writer.add_scalar("Episode/AvgAction_0", avg_act[0], ep + 1)
            if avg_act.size > 1:
                writer.add_scalar("Episode/AvgAction_1", avg_act[1], ep + 1)
            writer.add_scalar("Episode/AvgReward_Collision", ep_coll / ep_steps, ep + 1)
            writer.add_scalar("Episode/AvgReward_RightLane", ep_lane / ep_steps, ep + 1)
            writer.add_scalar("Episode/AvgReward_HighSpeed", ep_high / ep_steps, ep + 1)
            writer.add_scalar("Episode/AvgReward_OnRoad", ep_onroad / ep_steps, ep + 1)

        if len(all_episode_rewards) >= 20:
            writer.add_scalar("Reward/Moving_Avg", moving_avg, ep + 1)

            # Check for improvement for early stopping
            if moving_avg > best_avg_reward:
                best_avg_reward = moving_avg
                episodes_without_improvement = 0
                # Save best model
                agent.save(f"results/models/{run_name}/best_model.pth")
                logging.info(
                    f"New best model saved with average reward: {best_avg_reward:.2f}"
                )
            else:
                episodes_without_improvement += 1
                if episodes_without_improvement >= patience:
                    logging.info(
                        f"Early stopping after {patience} episodes without improvement"
                    )
                    break

        # Periodic checkpoints
        if (ep + 1) % checkpoint_every_ep == 0:
            agent.save(f"results/models/{run_name}/checkpoint_ep_{ep + 1}.pth")
            logging.info(f"Checkpoint saved at episode {ep + 1}")

        # Periodic recording
        if ep % record_every_ep == 0:
            record_agent(agent, env, run_name)

        # Periodic evaluation
        if ep % eval_every_ep == 0:
            eval_reward, eval_std = evaluate_agent(agent, env, num_episodes=5)
            writer.add_scalar("Evaluation/AvgReward", eval_reward, ep + 1)
            logging.info(
                f"Evaluation at episode {ep + 1}: Average reward = {eval_reward:.2f} ({eval_std:.2f})"
            )

        if moving_avg >= reward_threshold:
            logging.info(
                f"Reward threshold {reward_threshold} reached at episode {ep + 1}"
            )
            break


def evaluate_agent(agent: ppo.PPOAgent, env: gym.Env, num_episodes: int = 5) -> float:
    """
    Evaluate the agent without recording videos

    Args:
        agent: The PPO agent
        env: The environment
        num_episodes: Number of episodes to evaluate

    Returns:
        Average reward across episodes
    """
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action, _, _ = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            state = state.flatten()

            if info["rewards"].get("on_road_reward", 1) == 0:
                reward += -1.0

            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards)


def record_agent(agent: ppo.PPOAgent, env, run_name, num_eval_episodes=4):
    """
    Record a video of the trained agent's performance.

    Args:
        agent (ppo.PPO): Trained PPO agent.
        env (gym.Env): Environment to evaluate the agent.
        video_folder (str): Directory to save the video.
        run_name (str): Name of the current run for organizing videos.
        num_eval_episodes (int): Number of episodes to record.
    """
    # Create a subdirectory for the current run's videos
    video_folder = "results/videos/"
    run_video_folder = os.path.join(video_folder, run_name)
    os.makedirs(run_video_folder, exist_ok=True)
    logging.info(f"Recording videos to folder: {run_video_folder}")

    # Wrap the environment with video recording
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=run_video_folder,
        name_prefix="eval",
        episode_trigger=lambda _: True,  # Record all episodes
        fps=16,
    )

    for episode_num in range(num_eval_episodes):
        logging.info(
            f"Starting evaluation episode {episode_num + 1}/{num_eval_episodes}"
        )
        state, _ = env.reset()
        state = state.flatten()  # Ensure the observation is flattened
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action, _, _ = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)

            if info["rewards"].get("on_road_reward", 1) == 0:
                reward += -1.0

            state = state.flatten()
            total_reward += reward
            done = terminated or truncated

        logging.info(
            f"Episode {episode_num + 1} completed. Total reward: {total_reward}"
        )

    env.close()
    logging.info(f"Video recording completed. Videos saved to: {run_video_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="PPO agent training for RL tasks")
    parser.add_argument(
        "--run_name", type=str, default="", help="Name for this training run"
    )
    parser.add_argument(
        "--episodes", type=int, default=1280, help="Number of training episodes"
    )
    parser.add_argument(
        "--reward_threshold",
        type=int,
        default=400,
        help="Reward threshold for early stopping",
    )
    parser.add_argument(
        "--record_every", type=int, default=32, help="Record video every N episodes"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--eval_every", type=int, default=20, help="Evaluate agent every N episodes"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=128,
        help="Episodes without improvement before early stopping",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--actor_lr", type=float, default=1e-3, help="Actor learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=1e-4, help="Critic learning rate"
    )
    parser.add_argument(
        "--lambda_", type=float, default=0.95, help="GAE lambda parameter"
    )
    parser.add_argument(
        "--load_model", type=str, default="", help="Path to load a pre-trained model"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # === Get run name ===
    run_name = args.run_name
    if run_name == "":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"PPO_{timestamp}"

    print("Run name: ", run_name)

    # Create necessary directories
    os.makedirs("results/tensorboard", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/videos", exist_ok=True)

    # Tensorboard writer:
    writer = SummaryWriter(log_dir=f"results/tensorboard/{run_name}")

    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    env.unwrapped.configure(config_dict)
    _ = env.reset()

    action_space = env.action_space
    observation_space = env.observation_space

    gamma = args.gamma
    actor_learning_rate = args.actor_lr
    critic_learning_rate = args.critic_lr
    lambda_ = args.lambda_

    action_space = env.action_space
    observation_space = env.observation_space
    print("Observation space shape: ", observation_space.shape)

    agent = ppo.PPOAgent(
        np.prod(observation_space.shape),
        action_space.shape[0],
        gamma=gamma,
        lam=lambda_,
        pi_lr=actor_learning_rate,
        vf_lr=critic_learning_rate,
        clip_ratio=0.05,
    )

    # Load pre-trained model if specified
    if args.load_model:
        if os.path.exists(args.load_model):
            agent.load(args.load_model)
            logging.info(f"Loaded pre-trained model from {args.load_model}")
        else:
            logging.warning(
                f"Model file {args.load_model} not found. Starting with a new model."
            )

    train(
        env,
        agent,
        writer=writer,
        run_name=run_name,
        N_episodes=args.episodes,
        reward_threshold=args.reward_threshold,
        record_every_ep=args.record_every,
        checkpoint_every_ep=args.checkpoint_every,
        eval_every_ep=args.eval_every,
        patience=args.patience,
    )

    # Save the final model
    agent.save(f"results/models/{run_name}/final_model.pth")
    logging.info(f"Final model saved to results/models/{run_name}/final_model.pth")

    # Record videos of the trained agent
    record_agent(agent, env, run_name, num_eval_episodes=3)

    # Final evaluation
    final_reward, final_std = evaluate_agent(agent, env, num_episodes=10)
    logging.info(f"Final evaluation: Average reward = {final_reward:.2f}")
    writer.add_scalar("Evaluation/FinalAvgReward", final_reward)

    writer.close()


if __name__ == "__main__":
    main()
