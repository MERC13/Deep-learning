# stockproject

Stock price forecasting pipeline plus a simple Flask app to serve predictions and charts.

## Layout

- `train.py`: trains model(s) and writes artifacts to `static/` and `logs/`
- `app.py`: Flask app exposing `/` and `/predict`
- `dataprocessing.py`: feature engineering and dataset prep

## Setup (PowerShell)

Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation fails due to execution policy, enable it for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Train

```powershell
python train.py
```

## Serve (PowerShell)

Run the Flask app (the server listens on port 5001 by default):

```powershell
$env:FLASK_APP = 'app.py'; python app.py
```

Open http://localhost:5001 in your browser.

## Misc

Force retraining / overwrite the model for a run (PowerShell):

```powershell
$env:FORCE_NEW_MODEL = '1'
```