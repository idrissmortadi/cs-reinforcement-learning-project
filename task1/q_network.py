import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    A simple Feedforward Neural Network for Q-value approximation (Deep Q-Network).

    Architecture:
        Input Layer -> Linear(input_dim, 128) -> ReLU ->
        Hidden Layer 1 -> Linear(128, 128) -> ReLU ->
        Output Layer -> Linear(128, output_dim)

    Takes a state representation as input and outputs the estimated Q-values
    for each possible action in that state.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize the QNetwork layers.

        Args:
            input_dim (int): The dimensionality of the input state space.
                             This should match the size of the preprocessed state vector.
            output_dim (int): The number of possible discrete actions (dimensionality of the output).
        """
        super(QNetwork, self).__init__()
        # First hidden layer: transforms input state to a 128-dimensional representation
        self.layer1 = nn.Linear(input_dim, 128)
        # Second hidden layer: further processes the representation
        self.layer2 = nn.Linear(128, 128)
        # Output layer: produces Q-value estimates for each action
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): A batch of input states. Shape: (batch_size, input_dim).

        Returns:
            torch.Tensor: The predicted Q-values for each action for each state in the batch.
                          Shape: (batch_size, output_dim).
        """
        # Ensure input is float32
        if x.dtype != torch.float32:
            x = x.float()

        # Pass through the first hidden layer and apply ReLU activation
        x = F.relu(self.layer1(x))
        # Pass through the second hidden layer and apply ReLU activation
        x = F.relu(self.layer2(x))
        # Pass through the output layer (no activation function here, as we want raw Q-values)
        q_values = self.output_layer(x)
        return q_values
