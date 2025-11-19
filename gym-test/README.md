# gymtest

Simple Q-learning with discretization on CartPole-v1, using Gym. Renders to a window.

## Setup (PowerShell)

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If rendering fails, ensure `pygame` is installed and your display/drivers are available (WSL users may need an X server).

If activation fails due to execution policy, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run

```powershell
python main.py
```
