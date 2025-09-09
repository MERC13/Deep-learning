# gymtest

Simple Q-learning with discretization on CartPole-v1, using Gym. Renders to a window.

## Setup

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install gym numpy matplotlib pygame
```

If render fails, ensure `pygame` is installed and display drivers are available.

## Run

```powershell
python main.py
```
