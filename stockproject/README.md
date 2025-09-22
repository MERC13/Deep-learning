# stockproject

Stock price forecasting pipeline plus a simple Flask app to serve predictions and charts.

## Layout

- `train.py`: trains model(s) and writes artifacts to `static/` and `logs/`
- `app.py`: Flask app exposing `/` and `/predict`
- `dataprocessing.py`: feature engineering and dataset prep

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python train.py
```

## Serve

```powershell
$env:FLASK_APP = "app.py"; python app.py
```

App runs on http://localhost:5001
