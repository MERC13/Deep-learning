"""Experience replay buffer for DQN training."""

import random
import numpy as np
from collections import deque
from typing import Tuple, List, Any


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions.
    
    This buffer stores state transitions and provides random sampling
    for training the DQN to break correlation between consecutive samples.
    """
    
    def __init__(self, capacity: int):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Store a single experience tuple.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state observed
            done: Whether the episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            
        Raises:
            ValueError: If batch_size is larger than buffer size
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer of size {len(self.buffer)}")
        
        # Sample random experiences
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack and stack into arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards),
            np.stack(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling.
        
        Args:
            batch_size: Required batch size
            
        Returns:
            True if buffer can provide the required batch size
        """
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()