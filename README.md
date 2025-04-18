# Reinforcement Learning Project

This repository contains the implementation of a Reinforcement Learning (RL) project. The project is structured into multiple tasks, each focusing on a specific aspect of RL. Currently, only **Task 1** is implemented, with plans to add **Task 2**, **Task 3**, and **Task 4** in the future.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Task 1: Training a DQN Agent](#task-1-training-a-dqn-agent)
  - [Overview](#overview)
  - [Running the Training Script](#running-the-training-script)
  - [Evaluating the Agent](#evaluating-the-agent)
  - [Visualizing Results](#visualizing-results)
- [Future Tasks](#future-tasks)

---

## Environment Setup

1. Install the required dependencies using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate RL
   ```

2. Ensure all configurations are properly set up in the `configs/` directory.

---

## Task 1: Training a DQN Agent

### Overview
In Task 1, we train a Deep Q-Network (DQN) agent to perform well in the `highway-fast-v0` environment. The agent learns to navigate a highway while maximizing rewards and avoiding collisions. The environment is configured using the `configs/task_1_config.py` file.

### Running the Training Script

To train the DQN agent, run the following command:
```bash
python task1/run.py
```

- The script will:
  - Initialize the environment and networks.
  - Train the agent for the specified number of episodes.
  - Save the trained model and metrics in the `results/` directory.

### Evaluating the Agent

To evaluate the trained agent and record its performance, use the `play_or_record.py` script:
```bash
python task1/play_or_record.py --model <path_to_model> --video_folder <path_to_videos> --num_episodes 5
```
- Replace `<path_to_model>` with the path to the saved model (e.g., `results/models/dqn_policy_net_task1_final.pth`).
- Replace `<path_to_videos>` with the directory where videos should be saved.
- Add the `--live` flag to play the environment live instead of recording.

### Visualizing Results

1. **TensorBoard**:
   - Training metrics (e.g., loss, rewards) are logged to TensorBoard.
   - To view the logs, run:
     ```bash
     tensorboard --logdir results/tensorboard
     ```
   - Open the provided URL in your browser.

2. **Exporting Figures**:
   - Use the `export_run_figures.py` script to generate plots from TensorBoard logs:
     ```bash
     python task1/export_run_figures.py <path_to_event_file> <run_name>
     ```
   - Replace `<path_to_event_file>` with the path to the TensorBoard event file.
   - Replace `<run_name>` with a name for the output directory.

---

## Future Tasks

### Task 2: Continuous Action Space
- Train an agent in a continuous action space environment (`highway-fast-v0`).
- Use the configuration in `configs/task_2_config.py`.

### Task 3: Advanced Environment
- Train an agent in a more complex environment (`racetrack-v0`).
- Use the configuration in `configs/task_3_config.py`.

### Task 4: TBD
- Details will be added in the future.

---

For any questions or issues, feel free to contact the contributors listed in `configs/membres_groupe.txt`.