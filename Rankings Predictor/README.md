# Rankings Predictor

Predicts team win counts from tournament pairings using a Keras LSTM.

## Data shape

- Input: tournaments as rounds of pairings (padded to MAX_PAIRINGS_PER_ROUND x 2)
- Output: per-team win counts (padded to MAX_TEAMS)

See `data.py` for example structures.

## Setup

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train / run

```powershell
python main.py
```

Model file: `rankings_predictor_model.h5`
