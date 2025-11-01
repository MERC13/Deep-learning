# LSTMstock

Keras LSTM model for single-ticker forecasting using yfinance. Includes a Dockerfile.

## Setup (PowerShell)

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation fails, temporarily relax PowerShell execution policy for the session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run locally

```powershell
python train.py
```

## Docker

```powershell
# Build
docker build -t lstmstock .
# Run
docker run --rm -it lstmstock
```
