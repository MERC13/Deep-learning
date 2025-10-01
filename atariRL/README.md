# Atari Deep Q-Network (DQN) Implementation

A clean, modular implementation of Deep Q-Network (DQN) for playing Atari games using PyTorch and Gymnasium.

## Key Improvements & Features

This codebase provides a **professional, production-ready** implementation with the following improvements over typical single-file scripts:

### 🏗️ **Modular Architecture**
- Clean separation into logical modules (`models`, `agents`, `utils`, `configs`)
- Proper imports with `__init__.py` files
- Type hints and comprehensive docstrings throughout
- Clean interfaces between components

### ⚙️ **Configuration Management**
- Centralized configuration in `configs/config.py`
- Dataclass-based configuration with validation
- Easy parameter tuning without code changes
- Environment-specific settings

### 🎯 **Enhanced Features**
- **Deep Q-Network**: Convolutional neural network for processing Atari frames
- **Experience Replay**: Stable training through experience replay buffer
- **Target Network**: Separate target network for stable Q-learning
- **Frame Preprocessing**: Grayscale conversion, resizing, and frame stacking
- **Epsilon-Greedy Exploration**: Decaying exploration strategy
- **Training Visualization**: Real-time plotting of training metrics
- **Model Persistence**: Save and load trained models with checkpoints
- **Evaluation Tools**: Dedicated evaluation script for testing trained agents

### 💻 **Code Quality**
- Consistent naming conventions and error handling
- Memory management considerations
- GPU acceleration support
- Interrupt handling during training
- Professional documentation and setup

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

## Why This Implementation?

### 🔧 **Maintainability**
- Easy to modify individual components
- Clear separation of concerns
- Consistent code style throughout
- Proper documentation for future reference

### 🚀 **Extensibility**
- Easy to add new RL algorithms
- Pluggable components (different networks, buffers, etc.)
- Environment-agnostic design
- Configurable hyperparameters

### 🛡️ **Reliability**
- Error handling and validation
- Type safety with hints
- Tested interfaces between components
- Proper resource management

### 👥 **Usability**
- Simple command-line interface
- Clear configuration options
- Comprehensive documentation
- Easy installation and setup

## Installation

1. **Clone or download the project**

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

## Training Tips

1. **GPU Acceleration**: The code automatically detects and uses GPU if available
2. **Memory Requirements**: Training requires significant RAM for the replay buffer
3. **Training Time**: Full training can take several hours to days depending on hardware
4. **Hyperparameter Tuning**: Adjust learning rate, epsilon decay, and network architecture as needed
5. **Early Stopping**: Monitor training plots and stop if performance plateaus

## Results

During training, you'll see:
- Real-time training metrics printed to console
- Training plots showing scores, losses, and epsilon decay
- Saved models at regular intervals
- Final evaluation results

Expected performance on Breakout:
- Random agent: ~1-2 points
- Trained DQN: 200-400+ points (depending on training duration)

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