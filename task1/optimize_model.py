import torch
from hyperparameters import BATCH_SIZE, GAMMA, device
from torch import nn


def optimize_model(
    policy_net, target_net, optimizer, memory, writer=None, step=None, logging=None
):
    """
    Performs a single step of optimization for the DQN policy network.

    Samples a batch of experiences from the replay buffer, calculates the loss
    (using the Bellman equation and the target network), and updates the
    policy network's weights via backpropagation.

    Args:
        policy_net (QNetwork): The network being trained (predicts Q(s, a)).
        target_net (QNetwork): A separate network providing stable targets (predicts max_a' Q(s', a')).
                               Its weights are periodically updated from policy_net.
        optimizer (torch.optim.Optimizer): The optimizer used to update policy_net's weights (e.g., Adam).
        memory (ReplayBuffer): The replay buffer storing past experiences.
        writer (SummaryWriter, optional): TensorBoard writer for logging metrics. Defaults to None.
        step (int, optional): The current global training step number for logging. Defaults to None.
        logging (logging.Logger, optional): Logger instance for detailed logging. Defaults to None.

    Returns:
        float: The calculated loss value for this optimization step. Returns None if the
               replay buffer doesn't contain enough samples yet (less than BATCH_SIZE).
    """
    # Only perform optimization if the buffer has enough samples for a full batch
    if len(memory) < BATCH_SIZE:
        if logging:
            logging.debug(
                f"Skipping optimization: Buffer size ({len(memory)}) < Batch size ({BATCH_SIZE})"
            )
        return None

    # --- 1. Sample a Batch of Experiences ---
    try:
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
        if logging:
            logging.debug(f"Sampled batch of size {BATCH_SIZE} from replay buffer.")
    except ValueError as e:
        if logging:
            logging.error(f"Error sampling from buffer: {e}")
        return None  # Cannot proceed if sampling fails

    # --- 2. Convert data to PyTorch Tensors ---
    # Ensure data is on the correct device (CPU or GPU) and has the correct dtype.
    # States and next_states need to be flattened if they aren't already.
    # Actions need to be LongTensor for gather(). Dones need to be float for calculations.
    try:
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device).view(
            BATCH_SIZE, -1
        )
        actions_tensor = torch.tensor(
            actions, dtype=torch.long, device=device
        ).unsqueeze(1)  # Add dimension for gather
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_tensor = torch.tensor(
            next_states, dtype=torch.float32, device=device
        ).view(BATCH_SIZE, -1)
        dones_tensor = torch.tensor(
            dones, dtype=torch.float32, device=device
        )  # 1.0 if done, 0.0 otherwise
        if logging:
            logging.debug(
                f"Converted batch data to tensors. State shape: {states_tensor.shape}, Action shape: {actions_tensor.shape}"
            )
    except Exception as e:
        if logging:
            logging.error(f"Error converting batch data to tensors: {e}", exc_info=True)
        return None

    # --- 3. Calculate Current Q-values: Q(s, a) ---
    # Get Q-values for all actions from the policy_net for the sampled states
    # Then, select the Q-value corresponding to the action actually taken in the experience
    q_values_all = policy_net(states_tensor)
    # gather() selects the Q-value based on the action index in actions_tensor
    q_values_current = q_values_all.gather(1, actions_tensor).squeeze(
        1
    )  # Remove the added dimension

    # --- 4. Calculate Target Q-values: r + γ * max_a' Q_target(s', a') ---
    # Get the maximum Q-value for the next states from the target_net
    # We use target_net for stability. No gradients are needed here.
    with torch.no_grad():
        next_q_values_all = target_net(next_states_tensor)
        # Find the maximum Q-value among all possible actions in the next state
        next_q_values_max = next_q_values_all.max(1)[
            0
        ]  # .max(1) returns (values, indices)
        # Calculate the target value using the Bellman equation:
        # If the state is terminal (done=1), the target is just the reward.
        # Otherwise, it's reward + discounted max future Q-value.
        expected_q_values = rewards_tensor + (
            GAMMA * next_q_values_max * (1 - dones_tensor)
        )

    # --- 5. Calculate Loss ---
    # Use Mean Squared Error (MSE) loss between current Q-values and target Q-values
    # Other losses like Smooth L1 (Huber Loss) can also be used.
    loss_fn = nn.MSELoss()
    loss = loss_fn(q_values_current, expected_q_values)
    loss_value = loss.item()  # Get the scalar value of the loss

    if logging:
        logging.debug(f"Loss calculated: {loss_value:.4f}")
        # Log detailed Q-value stats occasionally if needed
        # if step % 100 == 0: # Example: Log every 100 steps
        #     logging.debug(f"Step {step} Q-value stats: Current Mean={q_values_current.mean().item():.3f}, Target Mean={expected_q_values.mean().item():.3f}")

    # --- 6. Backpropagation and Optimization Step ---
    optimizer.zero_grad()  # Reset gradients before backpropagation
    loss.backward()  # Compute gradients of the loss w.r.t. policy_net parameters

    # --- 7. Gradient Clipping (Optional but Recommended) ---
    # Prevents exploding gradients, improving training stability.
    # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0) # Clip by norm
    torch.nn.utils.clip_grad_value_(
        policy_net.parameters(), clip_value=100.0
    )  # Clip by value
    if logging:
        # You could add more detailed gradient logging here if needed
        # total_norm = 0
        # for p in policy_net.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # logging.debug(f"Gradient norm before clipping: {total_norm:.4f}") # Requires clip_grad_norm_
        logging.debug("Applied gradient clipping by value (100.0).")

    optimizer.step()  # Update policy_net weights based on computed gradients

    # --- 8. Logging to TensorBoard ---
    if writer and step is not None:
        # Log loss per optimization step
        # writer.add_scalar("Loss/Per_Step", loss_value, step) # Already logged in run.py loop

        # Log Q-value statistics to understand the scale and evolution of predicted values
        writer.add_scalar("Q-Values/Current_Max", q_values_current.max().item(), step)
        writer.add_scalar("Q-Values/Current_Min", q_values_current.min().item(), step)
        writer.add_scalar("Q-Values/Current_Mean", q_values_current.mean().item(), step)
        writer.add_scalar(
            "Q-Values/Target_Mean", expected_q_values.mean().item(), step
        )  # Log mean target Q

    return loss_value
