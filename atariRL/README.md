# Atari Deep Q-Network (DQN) Implementation

A clean, modular implementation of Deep Q-Network (DQN) for playing Atari games using PyTorch and Gymnasium.

## Project Structure

```
atariRL/
├── configs/
│   └── config.py              # Configuration settings
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── dqn_agent.py       # DQN agent implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── dqn.py             # Deep Q-Network model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Frame preprocessing
│   │   ├── replay_buffer.py   # Experience replay buffer
│   │   └── training_utils.py  # Training utilities and logging
│   └── __init__.py
├── saved_models/              # Directory for saved models
├── results/                   # Directory for training results
├── requirements.txt           # Python dependencies
├── main.py                    # Main training script
├── evaluate.py                # Model evaluation script
└── README.md                  # This file
```

## Installation

1. **Clone or download the project**

**(Optional) Download CUDA**  
Refer to the [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) for installation instructions. After downloading CUDA, import using ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install torch torchvision gymnasium[atari] ale-py numpy opencv-python matplotlib Pillow
   ```

## Usage

### Training

Train a DQN agent on Atari Breakout:

```bash
python main.py
```

The training script will:
- Create a DQN agent with default configuration
- Train for the specified number of episodes
- Save models periodically during training
- Generate training plots and metrics
- Evaluate the final trained model

### Configuration

Modify training parameters in `configs/config.py`:

```python
@dataclasses.dataclass
class DQNConfig:
    # Environment settings
    env_name: str = 'ALE/Breakout-v5'
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000000
    
    # Training settings
    n_episodes: int = 2000
    batch_size: int = 32
    # ... more parameters
```

### Evaluation

Evaluate a trained model:

```bash
# Basic evaluation (10 episodes, no rendering)
python evaluate.py saved_models/dqn_ALE_Breakout-v5_final.pth

# Evaluate with rendering
python evaluate.py saved_models/dqn_ALE_Breakout-v5_final.pth --render

# Evaluate for more episodes
python evaluate.py saved_models/dqn_ALE_Breakout-v5_final.pth --episodes 50
```

## Key Components

### DQNAgent (`src/agents/dqn_agent.py`)
The main reinforcement learning agent that:
- Manages the Q-network and target network
- Implements epsilon-greedy action selection
- Handles experience replay and network updates
- Provides model saving/loading functionality

### DQN Model (`src/models/dqn.py`)
Convolutional neural network that:
- Processes stacked grayscale frames (84x84x4)
- Uses three convolutional layers for feature extraction
- Outputs Q-values for each possible action

### Frame Preprocessor (`src/utils/preprocessing.py`)
Handles frame preprocessing:
- Converts RGB frames to grayscale
- Resizes frames to 84x84 pixels
- Stacks 4 consecutive frames for temporal information
- Normalizes pixel values to [0, 1]

### Replay Buffer (`src/utils/replay_buffer.py`)
Experience replay mechanism:
- Stores state transitions (s, a, r, s', done)
- Provides random sampling for training
- Breaks correlation between consecutive samples

### Training Logger (`src/utils/training_utils.py`)
Training utilities:
- Logs training metrics (scores, losses, epsilon values)
- Generates training plots and visualizations
- Saves metrics to JSON files
- Provides evaluation functionality


## Customization

### Different Atari Games

Change the environment in `configs/config.py`:
```python
env_name: str = 'ALE/Pong-v5'  # or any other Atari game
```

### Network Architecture

Modify the DQN architecture in `src/models/dqn.py` or adjust parameters in the config:
```python
conv_channels: Tuple[int, int, int] = (32, 64, 64)
hidden_size: int = 512
```

### Training Parameters

Adjust hyperparameters in the configuration:
```python
learning_rate: float = 5e-5  # Lower for more stable training
batch_size: int = 64         # Larger for more stable gradients
```

## Reference
@article{farebrother2024cale,
  title={C{ALE}: Continuous Arcade Learning Environment},
  author={Jesse Farebrother and Pablo Samuel Castro},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}