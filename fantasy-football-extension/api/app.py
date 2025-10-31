# api/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Tuple

# Ensure we can import model definition
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(MODELS_DIR)
from ft_transformer import FTTransformer

app = Flask(__name__)
CORS(app)

def _build_model_from_checkpoint(checkpoint: Dict) -> FTTransformer:
    """Construct FTTransformer using saved hyperparameters when available."""
    hparams = checkpoint.get('hyperparameters', {})
    return FTTransformer(
        num_continuous=checkpoint['num_continuous'],
        cat_cardinalities=checkpoint['cat_cardinalities'],
        d_token=hparams.get('d_token', 192),
        n_layers=hparams.get('n_layers', 3),
        n_heads=hparams.get('n_heads', 8),
        dropout=hparams.get('dropout', 0.15),
    )


# Load models for each position
models: Dict[str, FTTransformer] = {}
artifacts: Dict[str, Dict] = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

positions = ['QB', 'RB', 'WR', 'TE']
loaded = 0
for pos in positions:
    ckpt_path = os.path.join(MODELS_DIR, f'{pos}_transformer_complete.pt')
    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint not found for {pos} at {ckpt_path}")
        continue
    # Load model artifacts
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Initialize and load model
    model = _build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    models[pos] = model
    artifacts[pos] = checkpoint
    loaded += 1

print(f"Loaded models for {loaded} positions")

def _encode_categorical_value(le, val: str) -> int:
    """Encode a categorical value with a LabelEncoder, mapping unknowns to 0.
    We precompute a dict for speed and robustness."""
    # Build mapping once and cache on the encoder
    if not hasattr(le, '_class_to_idx'):
        le._class_to_idx = {c: i for i, c in enumerate(le.classes_)}  # type: ignore[attr-defined]
    return le._class_to_idx.get(str(val), 0)  # type: ignore[attr-defined]


def _prepare_batch(df: pd.DataFrame, artifact: Dict) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """Prepare numeric and categorical matrices using saved scaler/encoders.

    Returns (X_num, X_cat, index) where index can be used to map predictions back to rows.
    """
    feature_names = artifact['feature_names']
    cont_cols = [c for c in feature_names['continuous'] if c in df.columns]
    cat_cols = [c for c in feature_names['categorical'] if c in df.columns]

    # Fill NaNs similar to training
    cont_df = df[cont_cols].copy()
    for c in cont_cols:
        if cont_df[c].isna().any():
            cont_df[c] = cont_df[c].fillna(cont_df[c].median())

    # Scale numeric
    scaler = artifact['scaler']
    X_num = scaler.transform(cont_df.values.astype(float)) if len(cont_cols) else np.zeros((len(df), 0))

    # Encode categoricals
    X_cat_list = []
    for c in cat_cols:
        le = artifact['label_encoders'][c]
        vals = df[c].astype(str).values
        enc = np.array([_encode_categorical_value(le, v) for v in vals], dtype=np.int64)
        X_cat_list.append(enc)
    X_cat = np.vstack(X_cat_list).T if X_cat_list else np.zeros((len(df), 0), dtype=np.int64)

    return X_num, X_cat, df.index


def load_weekly_player_data(position: str, week: int, season: int) -> pd.DataFrame:
    """Load cleaned data for a given position/week/season (no engineered features).
    If the requested week is unavailable, fallback to the latest available and log it."""
    fp = os.path.join(DATA_DIR, 'processed', f'{position}_data.parquet')
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Processed data not found for {position}: {fp}")
    df = pd.read_parquet(fp)
    # Filter by season/week
    subset = df[(df['season'] == season) & (df['week'] == week)].copy()
    if subset.empty:
        # Fallback: use most recent available
        latest_season = df['season'].max()
        latest_week = df[df['season'] == latest_season]['week'].max()
        print(f"No data for season={season}, week={week}. Using latest available season={latest_season}, week={latest_week}.")
        subset = df[(df['season'] == latest_season) & (df['week'] == latest_week)].copy()
    return subset


@app.route('/predict', methods=['POST'])
def predict_player():
    """
    Predict fantasy points for a single player
    
    Request body:
    {
        "player_name": "Patrick Mahomes",
        "position": "QB",
        "week": 8,
        "season": 2025,
        "features": {
            "fantasy_points_rolling_3": 25.4,
            "opponent": "LAC",
            "is_home": 1,
            ...
        }
    }
    """
    data = request.json
    position = data['position']
    features = data['features']
    
    # Get model and artifacts
    if position not in models:
        return jsonify({'error': f'No model loaded for position {position}'}), 400
    model = models[position]
    artifact = artifacts[position]
    scaler = artifact['scaler']
    label_encoders = artifact['label_encoders']
    feature_names = artifact['feature_names']
    
    # Prepare features
    X_num = []
    for feat in feature_names['continuous']:
        X_num.append(features.get(feat, 0))
    X_num = np.array([X_num])
    X_num = scaler.transform(X_num)
    
    X_cat = []
    for feat in feature_names['categorical']:
        val = features.get(feat, 'UNK')
        X_cat.append(_encode_categorical_value(label_encoders[feat], val))
    X_cat = np.array([X_cat], dtype=np.int64)
    
    # Predict
    with torch.no_grad():
        X_num_tensor = torch.FloatTensor(X_num).to(device)
        X_cat_tensor = torch.LongTensor(X_cat).to(device)
        prediction = model(X_num_tensor, X_cat_tensor)
        predicted_points = float(prediction.cpu().numpy()[0][0])
    
    return jsonify({
        'player_name': data['player_name'],
        'position': position,
        'week': data['week'],
        'predicted_points': round(predicted_points, 1),
        'confidence': 'medium'  # Add uncertainty quantification later
    })


@app.route('/predictions/weekly', methods=['GET'])
def weekly_predictions():
    """
    Get predictions for all fantasy-relevant players for current week
    """
    week = int(request.args.get('week', 1))
    season = int(request.args.get('season', 2025))
    
    # Aggregate predictions for all loaded positions
    all_predictions: Dict[str, Dict] = {}

    for pos in positions:
        if pos not in models:
            continue
        # Load player data for this week
        pos_data = load_weekly_player_data(pos, week, season)

        # Prepare batch
        artifact = artifacts[pos]
        # Keep identifiers for response
        id_cols = ['player_display_name', 'player_name', 'recent_team', 'opponent_team']
        present_id_cols = [c for c in id_cols if c in pos_data.columns]
        X_num, X_cat, idx = _prepare_batch(pos_data, artifact)

        # Predict in batches
        model = models[pos]
        with torch.no_grad():
            X_num_tensor = torch.from_numpy(X_num).float().to(device)
            X_cat_tensor = torch.from_numpy(X_cat).long().to(device)
            preds = model(X_num_tensor, X_cat_tensor).cpu().numpy().reshape(-1)

        # Build response mapping by player name(s)
        for i, row_idx in enumerate(idx):
            row = pos_data.loc[row_idx]
            pred_obj = {
                'position': pos,
                'team': row['recent_team'] if 'recent_team' in pos_data.columns else None,
                'opponent': row['opponent_team'] if 'opponent_team' in pos_data.columns else None,
                'predicted_points': float(np.round(preds[i], 1)),
            }
            # Prefer display name, but add both keys to maximize match rate
            disp_name = row['player_display_name'] if 'player_display_name' in pos_data.columns else None
            raw_name = row['player_name'] if 'player_name' in pos_data.columns else None
            if disp_name:
                all_predictions[disp_name] = pred_obj
            if raw_name and raw_name != disp_name:
                all_predictions[raw_name] = pred_obj

    return jsonify(all_predictions)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'device': device,
        'positions': list(models.keys()),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
