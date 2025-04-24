# Highway Reinforcement Learning Project

A comprehensive implementation of various reinforcement learning algorithms applied to autonomous driving scenarios using the `highway-env` suite. This project explores different aspects of RL, from discrete to continuous action spaces and parameter sensitivity analysis.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Task 1: Discrete DQN Agent](#task-1-discrete-dqn-agent)
  - [Overview](#task-1-overview)
  - [Running the Training](#task-1-running-the-training)
  - [Evaluating the Agent](#task-1-evaluating-the-agent)
- [Task 2: Continuous PPO Agent](#task-2-continuous-ppo-agent)
  - [Overview](#task-2-overview)
  - [Running the Training](#task-2-running-the-training)
  - [Benchmarking](#task-2-benchmarking)
- [Task 3: Racetrack Environment](#task-3-racetrack-environment)
  - [Overview](#task-3-overview)
  - [Training Methods](#task-3-training-methods)
  - [Testing the Agent](#task-3-testing-the-agent)
- [Task 4: Reward Parameter Analysis](#task-4-reward-parameter-analysis)
  - [Overview](#task-4-overview)
  - [Running the Experiments](#task-4-running-the-experiments)
  - [Analyzing Results](#task-4-analyzing-results)
- [Visualization Tools](#visualization-tools)
- [Contributors](#contributors)

## Environment Setup

1. Create and activate the conda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate RL
   ```

2. Verify installation by running one of the test scripts:
   ```bash
   python configs/test_configs.py
   ```

## Project Structure

The project is organized around four main tasks, each exploring different aspects of reinforcement learning:

```
configs/                  # Environment configurations
├── task_1_config.py      # Discrete highway config
├── task_2_config.py      # Continuous highway config
├── task_3_config.py      # Racetrack environment config
├── membres_groupe.txt    # Contributors list
task_1/                   # DQN implementation
task_2/                   # PPO implementation
task_3/                   # Racetrack environment tasks
task_4/                   # Parameter analysis experiments
```

## Task 1: Discrete DQN Agent {#task-1-discrete-dqn-agent}

### Overview {#task-1-overview}

Task 1 implements a Deep Q-Network (DQN) agent to navigate a highway environment with discrete actions. The agent uses:

- Experience replay buffer to store and sample past interactions
- Target network to stabilize training
- Epsilon-greedy exploration strategy
- Occupancy grid observations of the surrounding area

The environment (`highway-fast-v0`) features:

- 4 lane highway with traffic
- Discrete action space (lane changes, speed adjustments)
- Rewards for right-lane preference, high speed, and penalties for collisions

### Running the Training {#task-1-running-the-training}

To train the DQN agent:

```bash
cd task_1
python run.py
```

The script will:

- Initialize the environment and networks
- Train the agent for a specified number of episodes (default: 1000)
- Save the model at regular intervals and after training
- Log metrics to TensorBoard
- Record evaluation videos after training

Key hyperparameters can be found in `task_1/hyperparameters.py`.

### Evaluating the Agent {#task-1-evaluating-the-agent}

To evaluate a trained agent and record its performance:

```bash
python task_1/play_or_record.py --model results/models/dqn_policy_net_task1_final.pth --video_folder results/videos --num_episodes 5
```

Add the `--live` flag to watch the agent in real-time instead of recording videos.

## Task 2: Continuous PPO Agent {#task-2-continuous-ppo-agent}

### Overview {#task-2-overview}

Task 2 explores continuous control using Proximal Policy Optimization (PPO) in the same highway environment. Key features:

- Continuous steering and acceleration actions
- Policy and value networks for actor-critic architecture
- GAE (Generalized Advantage Estimation) for advantage calculation
- Support for both actor-critic and policy-only implementations

The environment is configured with:

- 8 lanes for more complex navigation
- Detailed reward shaping for safe and efficient driving
- Higher simulation frequency for smoother control

### Running the Training {#task-2-running-the-training}

To train the PPO agent:

```bash
cd task_2
python run.py --episodes 1280 --record_every 32
```

Options include:

- `--run_name`: Custom name for the training run
- `--gamma`: Discount factor (default: 0.99)
- `--actor_lr`: Learning rate for the actor network
- `--critic_lr`: Learning rate for the critic network
- `--lambda_`: GAE lambda parameter

For the version without actor-critic:

```bash
python run_no_ac.py
```

### Benchmarking {#task-2-benchmarking}

Task 2 includes benchmarking capabilities:

```bash
python task_2/benchmark.py
```

This will compare the performance against other algorithms on standard environments.

## Task 3: Racetrack Environment {#task-3-racetrack-environment}

### Overview {#task-3-overview}

Task 3 moves to a more challenging `racetrack-v0` environment, requiring precise control to navigate curved tracks. Features:

- Complex racetrack navigation with lane centering and collision avoidance
- Off-road detection and termination
- More challenging reward structure with penalties for off-track driving

### Training Methods {#task-3-training-methods}

Two training approaches are implemented:

1. PPO (Proximal Policy Optimization):

```bash
cd task_3
python ppo_plot_figs.py
```

2. A2C (Advantage Actor-Critic):

```bash
python ac_plot_figs.py
```

Both implementations utilize:

- Parallelized training with multiple environments
- TensorBoard logging with detailed metrics
- Visualization tools for training progress

### Testing the Agent {#task-3-testing-the-agent}

To evaluate a trained agent on the racetrack:

```bash
python task_3/test_agent.py
```

This script loads a saved model and runs evaluation episodes, recording videos of the agent's performance.

## Task 4: Reward Parameter Analysis {#task-4-reward-parameter-analysis}

### Overview {#task-4-overview}

Task 4 conducts systematic experiments on how reward parameters affect agent behavior. Using the DQN algorithm from Task 1, it analyzes:

- The impact of `right_lane_reward` parameter (-0.5, 0, 0.5, 1.0)
- How reward shaping influences lane preference, speed, and collision rates
- Trade-offs between safety (avoiding collisions) and efficiency (maintaining speed)

### Running the Experiments {#task-4-running-the-experiments}

The experiments are implemented in a Jupyter notebook:

```bash
jupyter lab task_4/task4.ipynb
```

The notebook will:

1. Train DQN agents with different reward configurations
2. Evaluate each agent in controlled test environments
3. Collect metrics on lane occupancy, speed, and collision rates

### Analyzing Results {#task-4-analyzing-results}

The notebook generates visualizations including:

- Lane distribution plots showing preferred lanes for each reward setting
- Collision rate analysis across different reward parameters
- Speed-safety trade-off scatter plots
- Behavior profile radar charts

These visualizations are saved in the `task_4/figures/` directory.

## Visualization Tools

Several tools for visualizing training progress and results:

1. **TensorBoard**:

   ```bash
   tensorboard --logdir task[i]/results/tensorboard
   ```

2. **Exporting Figures** (Task 1):

   ```bash
   python task_1/export_run_figures.py <path_to_event_file> <run_name>
   ```

3. **Training Plots** (Task 3):

   ```bash
   python task_3/ppo_plot_figs.py  # For PPO metrics
   python task_3/ac_plot_figs.py   # For A2C metrics
   ```

## Contributors

Project contributors are listed in `configs/membres_groupe.txt`:

- Idriss MORTADI
- Abdellah OUMIDA
- Abdelaziz GUELFANE
- Aymane LOTFI
