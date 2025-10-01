"""Main script for Atari DQN training."""

import os
import sys
import gymnasium as gym
import ale_py

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from configs.config import DQNConfig
    from src.agents.dqn_agent import DQNAgent
    from src.utils.training_utils import TrainingLogger, evaluate_agent

    # Register Atari environments
    gym.register_envs(ale_py)

    def main():
        """Main training function."""
        print("Starting Atari DQN Training...")
        print("-" * 50)
        
        # Load configuration
        config = DQNConfig()
        
        # Create directories
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        
        # Create environment
        env = gym.make(
            config.env_name,
            render_mode=config.render_mode,
            frameskip=config.frameskip
        )
        
        # Create agent and logger
        agent = DQNAgent(env, config)
        logger = TrainingLogger(config.results_dir)
        
        try:
            # Training loop
            for episode in range(config.n_episodes):
                # Train one episode
                total_reward, episode_losses = agent.train_episode()
                
                # Log metrics
                logger.log_episode(
                    episode=episode,
                    score=total_reward,
                    episode_losses=episode_losses,
                    epsilon=agent.get_epsilon(),
                    episode_length=len(episode_losses)
                )
                
                # Print progress
                logger.print_progress(episode, config.log_frequency)
                
                # Save model periodically
                if episode % config.save_frequency == 0 and episode > 0:
                    model_path = os.path.join(
                        config.model_save_dir, 
                        f"dqn_{config.env_name.replace('/', '_')}_episode_{episode}.pth"
                    )
                    agent.save_model(model_path)
                
                agent.episode_count += 1
            
            print("\n" + "=" * 50)
            print("Training completed!")
            
            # Save final model
            final_model_path = os.path.join(
                config.model_save_dir,
                f"dqn_{config.env_name.replace('/', '_')}_final.pth"
            )
            agent.save_model(final_model_path)
            
            # Save training metrics and plot results
            logger.save_metrics()
            logger.plot_training_results()
            
            # Evaluate the trained agent
            print("\nEvaluating trained agent...")
            eval_results = evaluate_agent(agent, env, n_episodes=5)
            print(f"Mean Score: {eval_results['mean_score']:.2f} ± {eval_results['std_score']:.2f}")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            
            # Save current model
            interrupt_model_path = os.path.join(
                config.model_save_dir,
                f"dqn_{config.env_name.replace('/', '_')}_interrupted_episode_{episode}.pth"
            )
            agent.save_model(interrupt_model_path)
            logger.save_metrics()
            
        finally:
            env.close()

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease install required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    print("Make sure all dependencies are installed and try again.")
    sys.exit(1)