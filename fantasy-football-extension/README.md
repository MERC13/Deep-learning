# Fantasy Football Deep Learning Extension

Transformer-powered fantasy football predictions with cleaned NFL data and a Chrome/Edge extension overlay.

## Note to self
Next steps: real-time prediction, expert opinions, more data, relative value

## Overview

This repo contains:
- FT-Transformer/Temporal Transformer models for tabular/sequential NFL data (QB/RB/WR/TE)
- A Flask API that serves predictions from saved checkpoints
- A Chrome/Edge extension that overlays predictions on fantasy sites
- Data collection and preprocessing scripts (feature engineering optional)

Pretrained checkpoints for each position are included under `models/` so you can get predictions without training.

## Features

- Position-specific models (QB, RB, WR, TE)
- Time-aware splits and robust preprocessing (scaling + label encoding)
- Sequence-aware inference for players using prior-week history when available
- API endpoints for single-player and weekly-batch predictions
- Browser extension injection on Yahoo and ESPN (experimental Sleeper support)

## Project structure

```
fantasy-football-extension/
├─ api/                         # Flask API for predictions
│  └─ app.py                    # /predict, /predictions/weekly, /health
├─ data/
│  ├─ collect_data.py          # Download raw data from sources
│  ├─ export_heads.py          # Export schema/head samples
│  ├─ heads/                   # Small CSV heads and schemas
│  ├─ raw/                     # Raw data (parquet/csv)
│  └─ processed/               # Cleaned parquet used by the API
├─ preprocessing/
│  ├─ clean_data.py            # Clean/merge into processed parquet files
│  └─ engineer_features.py     # Optional feature engineering (disabled by default)
├─ models/
│  ├─ ft_transformer.py        # FT-Transformer (tabular)
│  ├─ temporal_transformer.py  # Transformer for sequences
│  ├─ train_position.py        # Train script (per-position)
│  ├─ *_transformer_complete.pt# Checkpoints with model+preprocessing artifacts
│  └─ best_model_*.pt          # Optional best snapshots
└─ extension/
     ├─ manifest.json            # MV3 manifest
     ├─ background.js            # Fetch + cache weekly predictions
     ├─ content.js               # Inject badges on supported sites
     └─ styles.css               # UI styles for injected elements
```

## Quick start (Windows / PowerShell)

The following uses the provided pretrained models.

1) Create and activate a virtual env (PowerShell)

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
```

If activation is blocked, allow it for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Prepare processed data (required by API)

```powershell
python preprocessing/clean_data.py
```

4) Start the API (default: http://127.0.0.1:5000)

```powershell
python api/app.py
```

5) Load the extension in Chrome/Edge

- Navigate to chrome://extensions (or edge://extensions)
- Toggle Developer mode
- Load unpacked and select the `extension/` folder

6) Visit Yahoo or ESPN fantasy pages. The overlay should display predicted points next to player names.

Tip: The extension fetches weekly predictions from the API automatically and caches them.

## API usage

Base URL: `http://127.0.0.1:5000`

- GET `/health`
    - Returns status, device (cpu/cuda), and loaded positions.

- POST `/predict`
    - Predict a single player. If the model is sequence-based and you do not pass `history`, the API will derive prior weeks from processed data.
    - Body (example):

```json
{
    "player_name": "Patrick Mahomes",
    "position": "QB",
    "week": 8,
    "season": 2025,
    "features": {},
    "history": [
        { "time_to_throw": 2.6, "recent_team": "KC", "opponent_team": "LAC", "season": 2025, "week": 6 }
    ]
}
```

- GET `/predictions/weekly?week=<int>&season=<int>`
    - Returns predictions for all relevant players for the given week.
    - Note: The current implementation internally uses the previous season for this route. Passing `season=2025` will operate on 2024 data if that’s what’s available.

## Data pipeline

1) Collect data (optional if you already have raw data)

```powershell
python data/collect_data.py
```

2) Clean/merge to produce processed parquet used by the API

```powershell
python preprocessing/clean_data.py
```

Optional: feature engineering is available via `preprocessing/engineer_features.py` but is disabled by default in training/inference.

## Training (optional)

If you want to retrain the models:

```powershell
python models/train_position.py
```

Default training uses historical splits (train/val/test by season). Hyperparameters in code (typical defaults):

```python
{
    'd_token': 192,
    'n_layers': 3,
    'n_heads': 8,
    'dropout': 0.15,
    'lr': 1e-4,
    'batch_size': 128
}
```

Hyperparameter search (Optuna) is available in the script (commented by default).

## Browser extension

- Supports Yahoo and ESPN content injection out of the box. Sleeper selectors are included but considered experimental.
- Requires the local API running at `http://127.0.0.1:5000` (or `http://localhost:5000`).
- The background worker refreshes predictions daily and on browser start.

## Requirements

See `requirements.txt`. PyTorch wheels are configured for CUDA 12.6 via the extra index; CPU-only installs are also supported (PyTorch will fall back to CPU if CUDA isn’t available).

## Troubleshooting

- API won’t start: ensure `preprocessing/clean_data.py` has produced files under `data/processed/` (e.g., `QB_data.parquet`).
- Extension shows no badges: verify `/predictions/weekly` returns data, and that you’re on a supported site (Yahoo/ESPN). Try the “Refresh predictions” path by reloading the page.
- CUDA errors: install a CUDA-compatible PyTorch build or run on CPU; the API will auto-select CPU if CUDA is unavailable.
- Port conflicts: change the port in `api/app.py` (last line) and update `extension/background.js` `API_URL` accordingly.

## Acknowledgments

- nfl_data_py (NFL data access)
- FT-Transformer (Gorishniy et al.) and rtdl library
- NFL Next Gen Stats (AWS player tracking)
