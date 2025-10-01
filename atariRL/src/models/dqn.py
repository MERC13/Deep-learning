"""Deep Q-Network implementation for Atari games."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class DQN(nn.Module):
    """Deep Q-Network for Atari games.
    
    This network processes stacked grayscale frames through convolutional layers
    followed by fully connected layers to output Q-values for each action.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_actions: int,
        conv_channels: Tuple[int, int, int] = (32, 64, 64),
        conv_kernels: Tuple[int, int, int] = (8, 4, 3),
        conv_strides: Tuple[int, int, int] = (4, 2, 1),
        hidden_size: int = 512
    ):
        """Initialize the DQN.
        
        Args:
            input_shape: Shape of input frames (channels, height, width)
            n_actions: Number of possible actions
            conv_channels: Number of channels for each conv layer
            conv_kernels: Kernel sizes for each conv layer
            conv_strides: Stride sizes for each conv layer
            hidden_size: Size of the hidden fully connected layer
        """
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_channels[0], 
                     kernel_size=conv_kernels[0], stride=conv_strides[0]),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1],
                     kernel_size=conv_kernels[1], stride=conv_strides[1]),
            nn.ReLU(),
            nn.Conv2d(conv_channels[1], conv_channels[2],
                     kernel_size=conv_kernels[2], stride=conv_strides[2]),
            nn.ReLU()
        )
        
        # Calculate the size of flattened conv features
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Fully connected layers for decision making
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        """Calculate the output size of convolutional layers.
        
        Args:
            shape: Input shape (channels, height, width)
            
        Returns:
            Flattened size of conv layer output
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            conv_out = self.conv_layers(dummy_input)
            return int(np.prod(conv_out.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action of shape (batch_size, n_actions)
        """
        # Extract spatial features
        conv_features = self.conv_layers(x)
        
        # Flatten for fully connected layers
        flattened = conv_features.view(conv_features.size(0), -1)
        
        # Output Q-values
        q_values = self.fc_layers(flattened)
        
        return q_values