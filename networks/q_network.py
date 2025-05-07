"""
Authors: Charankamal Brar, Johan Hernandez, Brittney Jones, Lucas Perry, Corey Young
Date: 05/07/2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    QNetwork class used for Deep Q-Learning with PyTorch.
    This class defines a neural network with three fully connected layers.
    The network is used to approximate the Q-values in Q-learning.
    """

    def __init__(
        self, input_dims: int, n_actions: int, fc1_dims: int, fc2_dims: int
    ) -> None:
        """
        Initialize the QNetwork class.

        - input_dims: int, the number of input features (state dimensions).
        - n_actions: int, the number of possible actions (outputs).
        - fc1_dims: int, the number of units in the first fully connected layer.
        - fc2_dims: int, the number of units in the second fully connected layer.
        """

        # Initialize the parent class
        super(QNetwork, self).__init__()

        # Define the first and second fully connected layer
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        # Define the output layer, where the number of units equals the number of possible actions
        self.out = nn.Linear(fc2_dims, n_actions)

        # Set the device to CPU
        self.device = torch.device("cpu")

        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QNetwork.

        - state: torch.Tensor, the input tensor representing the state.

        Returns The output tensor representing the Q-values for all possible actions.
        """

        # Pass the state through the first and second fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.out(x)
