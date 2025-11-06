"""Minimal script to run a DQNAgent as a policy without any CLI.

How to use:
  - Adjust the constants below if you want to change the environment, model path, episodes, or rendering.
  - Run: python simple.py
"""

import os
import sys
import gymnasium as gym
import ale_py

# Ensure local packages are importable when running this script directly
sys.path.append(os.path.dirname(__file__))

from configs.config import DQNConfig
from src.agents.dqn_agent import DQNAgent


# ======== Minimal knobs (no CLI) ========
ENV_NAME = "ALE/Breakout-v5"
EPISODES = 1  # keep small for a quick run
RENDER = False  # set True to visualize

# Try to use a saved model if it exists; otherwise run with random-initialized weights
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "saved_models", "dqn_ALE_Breakout-v5_final.pth"
)
MODEL_PATH = DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else None


def run():
    # Register Atari environments
    gym.register_envs(ale_py)

    # Build config
    config = DQNConfig()
    config.env_name = ENV_NAME
    config.render_mode = "human" if RENDER else None

    # Create env
    env = gym.make(config.env_name, render_mode=config.render_mode, frameskip=config.frameskip)

    # Create agent
    agent = DQNAgent(env, config)

    # Optionally load trained weights
    if MODEL_PATH is not None:
        print(f"Loading model from: {MODEL_PATH}")
        agent.load_model(MODEL_PATH)
    else:
        print("No model found; running with untrained weights (greedy policy).")

    # Evaluation mode: turn off exploration
    original_eps_start, original_eps_end = agent.config.epsilon_start, agent.config.epsilon_end
    agent.config.epsilon_start = 0.0
    agent.config.epsilon_end = 0.0

    try:
        for ep in range(EPISODES):
            obs, _ = env.reset()
            state = agent.preprocessor.reset(obs)
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = agent.preprocessor.step(next_obs)
                total_reward += reward
                steps += 1

            print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")
    finally:
        # restore epsilon in case this script is reused
        agent.config.epsilon_start, agent.config.epsilon_end = original_eps_start, original_eps_end
        env.close()


if __name__ == "__main__":
    run()