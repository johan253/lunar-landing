import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
  """QNetwork class used for Deep Q-Learning with PyTorch.
  This class defines a neural network with three fully connected layers."""

  def __init__(self, input_dims: int, n_actions: int, fc1_dims: int, fc2_dims: int) -> None:
    """Initialize the QNetwork"""

    super(QNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dims, fc1_dims)
    self.fc2 = nn.Linear(fc1_dims, fc2_dims)
    self.out = nn.Linear(fc2_dims, n_actions)
    self.device = torch.device('cpu')
    self.to(self.device)

  def forward(self, state: torch.Tensor) -> torch.Tensor:
    """Forward pass through the network."""
    
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    return self.out(x)