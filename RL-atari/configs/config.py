"""Configuration settings for Atari RL training."""

import dataclasses
from typing import Tuple, Optional


@dataclasses.dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    
    # Environment settings
    env_name: str = 'ALE/Breakout-v5'
    render_mode: Optional[str] = None  # Set to "human" for visualization
    frameskip: int = 4
    
    # Network architecture
    frame_size: Tuple[int, int] = (84, 84)
    frame_stack: int = 4
    conv_channels: Tuple[int, int, int] = (32, 64, 64)
    conv_kernels: Tuple[int, int, int] = (8, 4, 3)
    conv_strides: Tuple[int, int, int] = (4, 2, 1)
    hidden_size: int = 512
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000000
    batch_size: int = 32
    buffer_size: int = 100000
    target_update_frequency: int = 10000
    
    # Training settings
    n_episodes: int = 100
    max_steps_per_episode: int = 10000
    save_frequency: int = 100
    log_frequency: int = 10
    
    # Paths
    model_save_dir: str = "saved_models"
    results_dir: str = "results"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 <= self.gamma <= 1, "Gamma must be between 0 and 1"
        assert self.epsilon_start >= self.epsilon_end, "Epsilon start must be >= epsilon end"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.buffer_size > self.batch_size, "Buffer size must be > batch size"