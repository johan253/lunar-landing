import random
from collections import deque, namedtuple
from typing import Tuple, Deque, NamedTuple
import numpy as np

class Transition(NamedTuple):
  """A named tuple representing a single transition in the replay buffer."""

  state: np.ndarray
  action: int
  reward: float
  next_state: np.ndarray
  done: bool

class ReplayBuffer:
  def __init__(self, buffer_size: int, batch_size: int) -> None:
    """Initialize the ReplayBuffer with a given size and batch size."""
    
    self.memory: Deque[Transition] = deque(maxlen=buffer_size)
    self.batch_size = batch_size

  def add(
    self,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool
  ) -> None:
    """Add a transition to the replay buffer."""
    
    self.memory.append(Transition(state, action, reward, next_state, done))

  def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a batch of transitions from the replay buffer."""
    
    transitions = random.sample(self.memory, self.batch_size)
    batch = Transition(*zip(*transitions))
    return (
      np.array(batch.state),
      np.array(batch.action),
      np.array(batch.reward),
      np.array(batch.next_state),
      np.array(batch.done)
    )
  
  def __len__(self) -> int:
    """Return the current size of the replay buffer."""
    
    return len(self.memory)