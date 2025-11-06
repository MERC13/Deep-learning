"""Evaluation script for trained DQN models."""

import os
import sys
import argparse
import gymnasium as gym
import ale_py

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from configs.config import DQNConfig
from src.agents.dqn_agent import DQNAgent
from src.utils.training_utils import evaluate_agent

# Register Atari environments
gym.register_envs(ale_py)


def evaluate_model(model_path: str, n_episodes: int = 10, render: bool = False) -> None:
    """Evaluate a trained DQN model.
    
    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    """
    print(f"Evaluating model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("-" * 50)
    
    # Load configuration (you might want to save this with the model)
    config = DQNConfig()
    if render:
        config.render_mode = "human"
    
    # Create environment
    env = gym.make(
        config.env_name,
        render_mode=config.render_mode,
        frameskip=config.frameskip
    )
    
    # Create agent and load model
    agent = DQNAgent(env, config)
    agent.load_model(model_path)
    
    try:
        # Evaluate
        results = evaluate_agent(agent, env, n_episodes=n_episodes, render=render)
        
        print("\n" + "=" * 50)
        print("Evaluation Results:")
        print(f"  Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
        print(f"  Min Score: {results['min_score']:.2f}")
        print(f"  Max Score: {results['max_score']:.2f}")
        print(f"  Mean Episode Length: {results['mean_length']:.1f}")
        
        print(f"\nIndividual episode scores:")
        for i, score in enumerate(results['scores'], 1):
            print(f"  Episode {i}: {score:.2f}")
            
    finally:
        env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model")
    parser.add_argument("model_path", help="Path to the saved model file")
    parser.add_argument("--episodes", "-e", type=int, default=10, 
                       help="Number of episodes to evaluate (default: 10)")
    parser.add_argument("--render", "-r", action="store_true",
                       help="Render the environment during evaluation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        sys.exit(1)
    
    evaluate_model(args.model_path, args.episodes, args.render)


if __name__ == "__main__":
    main()