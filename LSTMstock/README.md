# LSTMstock

Keras LSTM model for single-ticker forecasting using yfinance. Includes a Dockerfile.

## Setup

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
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
