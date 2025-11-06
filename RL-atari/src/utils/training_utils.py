"""Training utilities and visualization functions."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from datetime import datetime


class TrainingLogger:
    """Logger for training metrics and visualization."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the logger.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Training metrics
        self.scores = []
        self.losses = []
        self.epsilons = []
        self.episode_lengths = []
        
        # Running averages for smoother plots
        self.avg_scores = []
        self.avg_losses = []
    
    def log_episode(
        self, 
        episode: int, 
        score: float, 
        episode_losses: List[float], 
        epsilon: float,
        episode_length: int
    ) -> None:
        """Log metrics for a single episode.
        
        Args:
            episode: Episode number
            score: Total episode reward
            episode_losses: List of losses from the episode
            epsilon: Current epsilon value
            episode_length: Number of steps in the episode
        """
        self.scores.append(score)
        self.epsilons.append(epsilon)
        self.episode_lengths.append(episode_length)
        
        # Log average loss for the episode
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        self.losses.append(avg_loss)
        
        # Calculate running averages (last 100 episodes)
        window_size = min(100, len(self.scores))
        self.avg_scores.append(np.mean(self.scores[-window_size:]))
        self.avg_losses.append(np.mean(self.losses[-window_size:]))
    
    def print_progress(self, episode: int, log_frequency: int = 10) -> None:
        """Print training progress.
        
        Args:
            episode: Current episode number
            log_frequency: How often to print progress
        """
        if episode % log_frequency == 0 and episode > 0:
            recent_scores = self.scores[-log_frequency:]
            avg_score = np.mean(recent_scores)
            avg_loss = self.avg_losses[-1] if self.avg_losses else 0.0
            current_epsilon = self.epsilons[-1] if self.epsilons else 0.0
            avg_length = np.mean(self.episode_lengths[-log_frequency:])
            
            print(f"Episode {episode:4d} | "
                  f"Avg Score: {avg_score:7.2f} | "
                  f"Avg Loss: {avg_loss:.6f} | "
                  f"Epsilon: {current_epsilon:.3f} | "
                  f"Avg Length: {avg_length:.1f}")
    
    def plot_training_results(self, save_path: str = None) -> None:
        """Plot training results.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot scores
        episodes = range(len(self.scores))
        ax1.plot(episodes, self.scores, alpha=0.3, label='Episode Score')
        if self.avg_scores:
            ax1.plot(episodes, self.avg_scores, label='Running Average (100 episodes)')
        ax1.set_title('Training Scores')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot losses
        ax2.plot(episodes, self.losses, alpha=0.3, label='Episode Loss')
        if self.avg_losses:
            ax2.plot(episodes, self.avg_losses, label='Running Average (100 episodes)')
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot epsilon decay
        ax3.plot(episodes, self.epsilons)
        ax3.set_title('Epsilon Decay')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True, alpha=0.3)
        
        # Plot episode lengths
        ax4.plot(episodes, self.episode_lengths, alpha=0.3, label='Episode Length')
        if len(self.episode_lengths) > 0:
            window_size = min(100, len(self.episode_lengths))
            running_avg_lengths = [
                np.mean(self.episode_lengths[max(0, i-window_size):i+1]) 
                for i in range(len(self.episode_lengths))
            ]
            ax4.plot(episodes, running_avg_lengths, label='Running Average (100 episodes)')
        ax4.set_title('Episode Lengths')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f"training_results_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training plot saved to {save_path}")
    
    def save_metrics(self, filepath: str = None) -> None:
        """Save training metrics to JSON file.
        
        Args:
            filepath: Optional path to save metrics
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.results_dir, f"training_metrics_{timestamp}.json")
        
        metrics = {
            'scores': self.scores,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'episode_lengths': self.episode_lengths,
            'avg_scores': self.avg_scores,
            'avg_losses': self.avg_losses,
            'total_episodes': len(self.scores),
            'final_avg_score': self.avg_scores[-1] if self.avg_scores else 0.0,
            'best_score': max(self.scores) if self.scores else 0.0
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Training metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str) -> None:
        """Load training metrics from JSON file.
        
        Args:
            filepath: Path to metrics file
        """
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        
        self.scores = metrics['scores']
        self.losses = metrics['losses']
        self.epsilons = metrics['epsilons']
        self.episode_lengths = metrics['episode_lengths']
        self.avg_scores = metrics['avg_scores']
        self.avg_losses = metrics['avg_losses']
        
        print(f"Training metrics loaded from {filepath}")


def evaluate_agent(agent, env, n_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
    """Evaluate a trained agent.
    
    Args:
        agent: Trained DQN agent
        env: Environment for evaluation
        n_episodes: Number of evaluation episodes
        render: Whether to render the environment
        
    Returns:
        Dictionary with evaluation results
    """
    scores = []
    episode_lengths = []
    
    # Save current epsilon and set to 0 for evaluation (no exploration)
    original_epsilon_start = agent.config.epsilon_start
    original_epsilon_end = agent.config.epsilon_end
    agent.config.epsilon_start = 0.0
    agent.config.epsilon_end = 0.0
    
    try:
        for episode in range(n_episodes):
            obs, _ = env.reset()
            state = agent.preprocessor.reset(obs)
            total_reward = 0.0
            steps = 0
            
            while True:
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = agent.preprocessor.step(next_obs)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            scores.append(total_reward)
            episode_lengths.append(steps)
            print(f"Evaluation episode {episode + 1}: Score = {total_reward:.2f}, Length = {steps}")
    
    finally:
        # Restore original epsilon values
        agent.config.epsilon_start = original_epsilon_start
        agent.config.epsilon_end = original_epsilon_end
    
    results = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'mean_length': np.mean(episode_lengths),
        'scores': scores,
        'episode_lengths': episode_lengths
    }
    
    return results