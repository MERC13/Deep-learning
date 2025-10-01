"""DQN Agent implementation for Atari games."""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List

from ..models.dqn import DQN
from ..utils.replay_buffer import ReplayBuffer
from ..utils.preprocessing import AtariPreprocessor


class DQNAgent:
    """Deep Q-Network agent for playing Atari games.
    
    This agent implements the DQN algorithm with experience replay,
    target network, and epsilon-greedy exploration.
    """
    
    def __init__(self, env, config):
        """Initialize the DQN agent.
        
        Args:
            env: Gymnasium environment
            config: Configuration object with hyperparameters
        """
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Environment information
        self.n_actions = env.action_space.n
        self.frame_shape = (config.frame_stack, config.frame_size[1], config.frame_size[0])
        
        # Initialize networks
        self._setup_networks()
        
        # Initialize components
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.preprocessor = AtariPreprocessor(config.frame_size, config.frame_stack)
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
    
    def _setup_networks(self) -> None:
        """Initialize Q-network and target network."""
        self.q_network = DQN(
            input_shape=self.frame_shape,
            n_actions=self.n_actions,
            conv_channels=self.config.conv_channels,
            conv_kernels=self.config.conv_kernels,
            conv_strides=self.config.conv_strides,
            hidden_size=self.config.hidden_size
        ).to(self.device)
        
        self.target_network = DQN(
            input_shape=self.frame_shape,
            n_actions=self.n_actions,
            conv_channels=self.config.conv_channels,
            conv_kernels=self.config.conv_kernels,
            conv_strides=self.config.conv_strides,
            hidden_size=self.config.hidden_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_epsilon(self) -> float:
        """Calculate current epsilon for epsilon-greedy policy.
        
        Returns:
            Current epsilon value
        """
        epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                 np.exp(-self.step_count / self.config.epsilon_decay)
        return epsilon
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (stacked frames)
            
        Returns:
            Selected action index
        """
        if random.random() < self.get_epsilon():
            return random.randrange(self.n_actions)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def update_network(self) -> Optional[float]:
        """Update Q-network using experience replay.
        
        Returns:
            Training loss if update was performed, None otherwise
        """
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_episode(self) -> Tuple[float, List[float]]:
        """Train for one episode.
        
        Returns:
            Tuple of (total_reward, episode_losses)
        """
        obs, _ = self.env.reset()
        state = self.preprocessor.reset(obs)
        total_reward = 0.0
        episode_losses = []
        
        for step in range(self.config.max_steps_per_episode):
            # Select and perform action
            action = self.select_action(state)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Preprocess next state
            next_state = self.preprocessor.step(next_obs)
            
            # Store experience
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update network
            loss = self.update_network()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update target network periodically
            if self.step_count % self.config.target_update_frequency == 0:
                self.update_target_network()
            
            state = next_state
            total_reward += reward
            self.step_count += 1
            
            if done:
                break
        
        return total_reward, episode_losses
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        # PyTorch 2.6 defaults torch.load(weights_only=True), which blocks unpickling
        # of custom classes like our DQNConfig saved in the checkpoint. We'll try a
        # few strategies to remain compatible and safe:
        # 1) Attempt safe allowlisting of DQNConfig if supported
        # 2) Fall back to weights_only=False (only if you trust the checkpoint source)
        checkpoint = None
        load_errors = []
        
        # Attempt 1: safe allowlist + default load
        try:
            try:
                # Only available in newer PyTorch versions
                from torch.serialization import add_safe_globals  # type: ignore
                # Import here to avoid circulars at module import time
                from configs.config import DQNConfig as _DQNConfig  # type: ignore
                add_safe_globals([_DQNConfig])
            except Exception as _e:
                # Safe globals may not exist or import may fail; record and proceed
                load_errors.append(_e)
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e1:
            load_errors.append(e1)
            
            # Attempt 2: explicitly disable weights_only for backward compatibility
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)  # type: ignore
            except TypeError:
                # Older PyTorch without weights_only parameter
                checkpoint = torch.load(filepath, map_location=self.device)
            except Exception as e2:
                load_errors.append(e2)
                # Re-raise the original error with context
                raise e1
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        
        print(f"Model loaded from {filepath}")