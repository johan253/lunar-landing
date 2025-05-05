"""
Authors: Charankamal Brar, Johan Hernandez, Brittney Jones, Lucas Perry, Corey Young
Date: 05/07/2025
"""
import random
from collections import deque, namedtuple
from typing import Tuple, Deque, NamedTuple
import numpy as np

class Transition(NamedTuple):
    """
    A named tuple representing a single transition in the replay buffer.
    
    This class holds a single transition, which includes:
    - state: The current state of the agent.
    - action: The action taken by the agent in that state.
    - reward: The reward received after taking the action.
    - next_state: The state of the environment after the action is taken.
    - done: A boolean indicating whether the episode has ended.
    """
    
    state: np.ndarray     # The state before taking the action.
    action: int           # The action taken in the given state.
    reward: float         # The reward received after taking the action.
    next_state: np.ndarray # The state of the environment after the action.
    done: bool            # A boolean indicating if the episode is finished.

class ReplayBuffer:
    """
    ReplayBuffer class used for storing and sampling transitions for experience replay.
    
    This class stores a collection of transitions, which can later be sampled to 
    train a reinforcement learning agent. The buffer has a fixed size, and the 
    oldest transitions are discarded when the buffer reaches its capacity.
    """

    def __init__(self, buffer_size: int, batch_size: int) -> None:
        """
        Initialize the ReplayBuffer with a given size and batch size.
        
        - buffer_size: The maximum size of the buffer, representing the 
          maximum number of transitions it can store.
        - batch_size: The size of the batches to be sampled during training.
        """
        
        # Initialize the buffer as a deque with a maximum size of buffer_size
        self.memory: Deque[Transition] = deque(maxlen=buffer_size)
        
        # Set the batch size used for sampling transitions
        self.batch_size = batch_size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the replay buffer.
        
        This method stores the current state, action, reward, next state, and 
        done flag as a transition object in the buffer.
        
        - state: The current state of the environment.
        - action: The action taken by the agent.
        - reward: The reward received after taking the action.
        - next_state: The next state of the environment after the action.
        - done: Whether the episode has ended (True or False).
        """
        
        # Create a Transition object and add it to the memory
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions from the replay buffer.
        
        Randomly samples a batch of transitions from the buffer for training the agent.
        
        Returns A tuple of arrays containing the states, actions, rewards, next states, 
          and done flags from the sampled transitions.
        """
        
        # Randomly sample a batch of transitions from the memory
        transitions = random.sample(self.memory, self.batch_size)
        
        # Unzip the batch of transitions into separate arrays for each element
        batch = Transition(*zip(*transitions))
        
        # Convert the lists into numpy arrays and return them
        return (
            np.array(batch.state),      # Array of states
            np.array(batch.action),     # Array of actions
            np.array(batch.reward),     # Array of rewards
            np.array(batch.next_state), # Array of next states
            np.array(batch.done)        # Array of done flags
        )
  
    def __len__(self) -> int:
        """
        The current size of the replay buffer.
        
        This is helpful to check how many transitions are currently stored in 
        the buffer.
        
        Returns The number of transitions currently stored in the buffer.
        """
        
        # Return the length of the memory deque, i.e., the current size of the buffer
        return len(self.memory)