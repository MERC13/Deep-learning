# api/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Tuple, List, Optional

# Ensure we can import model definition
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(MODELS_DIR)
# Import model classes lazily via importlib inside builder to avoid static resolution issues
import importlib

app = Flask(__name__)
CORS(app)

def _allowlist_sklearn_for_torch_load():
    """Allowlist sklearn classes for torch.load safe unpickler (PyTorch 2.6+).

    This mirrors the training script setup to ensure we can load scaler/encoders
    stored in checkpoints.
    """
    try:
        from sklearn.preprocessing import StandardScaler, LabelEncoder  # noqa: F401
        import torch.serialization as _ts  # type: ignore
        _ts.add_safe_globals([StandardScaler, LabelEncoder])  # type: ignore[attr-defined]
    except Exception:
        pass


def _torch_load_compat(path, map_location=None, weights_only=None):
    """Compatibility wrapper for torch.load handling weights_only across versions."""
    try:
        if weights_only is None:
            return torch.load(path, map_location=map_location)
        else:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _build_model_from_checkpoint(checkpoint: Dict):
    """Construct model (FTTransformer or TemporalTransformer) based on checkpoint."""
    hparams = checkpoint.get('hyperparameters', {})
    num_continuous = checkpoint['num_continuous']

    # New sequence checkpoints use cat_cardinalities_padded
    if 'cat_cardinalities_padded' in checkpoint:
        d_model = hparams.get('d_token', 192)
        # Lazy import to avoid linter resolution issues
        TT = importlib.import_module('temporal_transformer').TemporalTransformer
        return TT(
            num_continuous=num_continuous,
            cat_cardinalities_padded=checkpoint['cat_cardinalities_padded'],
            d_model=d_model,
            n_layers=hparams.get('n_layers', 3),
            n_heads=hparams.get('n_heads', 8),
            dropout=hparams.get('dropout', 0.15),
            max_len=64,
        )

    # Backward compatibility: older flat FTTransformer checkpoints
    FT = importlib.import_module('ft_transformer').FTTransformer
    return FT(
        num_continuous=num_continuous,
        cat_cardinalities=checkpoint['cat_cardinalities'],
        d_token=hparams.get('d_token', 192),
        n_layers=hparams.get('n_layers', 3),
        n_heads=hparams.get('n_heads', 8),
        dropout=hparams.get('dropout', 0.15),
    )


# Load models for each position
models: Dict[str, torch.nn.Module] = {}
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
    _allowlist_sklearn_for_torch_load()
    checkpoint = _torch_load_compat(ckpt_path, map_location=device, weights_only=False)

    # Initialize and load model
    model = _build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    models[pos] = model
    # Mark whether this is a sequence model and compute pad indices for cats
    # Determine sequence model by presence of sequence-specific key in checkpoint
    is_sequence = 'cat_cardinalities_padded' in checkpoint
    artifact = dict(checkpoint)
    artifact['is_sequence'] = is_sequence
    if is_sequence:
        # Build pad indices from saved label encoders (length of classes per categorical feat)
        fns = artifact['feature_names']
        cat_feats = fns.get('categorical', [])
        pad_idx = [len(artifact['label_encoders'][c].classes_) for c in cat_feats]
        artifact['cat_pad_indices'] = np.array(pad_idx, dtype=np.int64)
    artifacts[pos] = artifact
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


def _prepare_sequence_for_player(
    df_all: pd.DataFrame,
    player_id: str,
    season: int,
    target_week: int,
    artifact: Dict,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build a single player's prior-weeks sequence up to (but not including) target_week.

    Returns (X_num_seq, X_cat_seq) or None if no prior weeks available.
    """
    fns = artifact['feature_names']
    cont_cols = [c for c in fns['continuous'] if c in df_all.columns]
    cat_cols = [c for c in fns['categorical'] if c in df_all.columns]

    # Identify player column used in preprocessing/training
    pid_col = 'player_display_name' if 'player_display_name' in df_all.columns else (
        'player' if 'player' in df_all.columns else 'full_name'
    )

    mask = (df_all[pid_col] == player_id) & (df_all['season'] == season) & (df_all['week'] < target_week)
    hist = df_all.loc[mask].sort_values('week').copy()
    if hist.empty:
        return None

    # Fill NaNs similar to training: median for continuous, mode default for cats handled by encoder mapping
    for c in cont_cols:
        if hist[c].isna().any():
            med = df_all[c].median() if c in df_all.columns else hist[c].median()
            hist[c] = hist[c].fillna(med)

    # Scale numeric using saved scaler
    scaler = artifact['scaler']
    X_num_seq = scaler.transform(hist[cont_cols].values.astype(float)) if len(cont_cols) else np.zeros((len(hist), 0))

    # Encode categoricals with saved label encoders, unknown -> 0
    X_cat_list: List[np.ndarray] = []
    for c in cat_cols:
        le = artifact['label_encoders'][c]
        vals = hist[c].astype(str).values
        enc = np.array([_encode_categorical_value(le, v) for v in vals], dtype=np.int64)
        X_cat_list.append(enc)
    X_cat_seq = np.stack(X_cat_list, axis=1) if X_cat_list else np.zeros((len(hist), 0), dtype=np.int64)

    return X_num_seq.astype(np.float32), X_cat_seq.astype(np.int64)


def _collate_sequences_for_inference(
    seq_num_list: List[np.ndarray],
    seq_cat_list: List[np.ndarray],
    pad_cat_indices: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences for inference.

    Returns X_num (B,T,Cn), X_cat (B,T,Cc), key_padding_mask (B,T)
    """
    B = len(seq_num_list)
    T_max = max(s.shape[0] for s in seq_num_list) if B > 0 else 0
    Cn = seq_num_list[0].shape[1] if B > 0 else 0
    Cc = seq_cat_list[0].shape[1] if (B > 0 and seq_cat_list[0].ndim == 2) else 0

    xnum = torch.zeros((B, T_max, Cn), dtype=torch.float32)
    xcat = torch.zeros((B, T_max, Cc), dtype=torch.long)
    kpm = torch.ones((B, T_max), dtype=torch.bool)

    for i, (xn, xc) in enumerate(zip(seq_num_list, seq_cat_list)):
        T = xn.shape[0]
        xnum[i, :T, :] = torch.from_numpy(xn)
        if Cc > 0 and xc.size > 0:
            xcat[i, :T, :] = torch.from_numpy(xc)
            if T < T_max:
                pad_row = torch.tensor(pad_cat_indices, dtype=torch.long).unsqueeze(0).expand(T_max - T, -1)
                xcat[i, T:, :] = pad_row
        kpm[i, :T] = False

    return xnum, xcat, kpm


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
    feature_names = artifact['feature_names']

    # Sequence model path
    if artifact.get('is_sequence', False):
        seq_num_list: List[np.ndarray] = []
        seq_cat_list: List[np.ndarray] = []

        history = data.get('history')  # optional explicit prior weeks
        if isinstance(history, list) and len(history) > 0:
            # Build from provided history objects
            cont_cols = feature_names['continuous']
            cat_cols = feature_names['categorical']
            # Build a DataFrame-like list to reuse encoders/scaler
            # Fill missing numeric with zeros then scale
            Xn = []
            for h in history:
                Xn.append([h.get(c, 0) for c in cont_cols])
            Xn = np.array(Xn, dtype=float)
            Xn = artifact['scaler'].transform(Xn) if len(cont_cols) else np.zeros((len(history), 0))

            Xc_cols: List[np.ndarray] = []
            for c in cat_cols:
                le = artifact['label_encoders'][c]
                vals = [str(h.get(c, 'UNK')) for h in history]
                enc = np.array([_encode_categorical_value(le, v) for v in vals], dtype=np.int64)
                Xc_cols.append(enc)
            Xc = np.stack(Xc_cols, axis=1) if Xc_cols else np.zeros((len(history), 0), dtype=np.int64)
            seq_num_list.append(Xn.astype(np.float32))
            seq_cat_list.append(Xc.astype(np.int64))
        else:
            # Derive history from processed data using player_name/season/week
            player_name = data.get('player_name')
            season = int(data.get('season'))
            week = int(data.get('week'))
            df_all = pd.read_parquet(os.path.join(DATA_DIR, 'processed', f"{position}_data.parquet"))
            res = _prepare_sequence_for_player(df_all, player_name, season, week, artifact)
            if res is None:
                return jsonify({'error': 'No prior history available to build sequence for prediction.'}), 400
            xn, xc = res
            seq_num_list.append(xn)
            seq_cat_list.append(xc)

        # Collate and predict
        Xn_t, Xc_t, kpm_t = _collate_sequences_for_inference(seq_num_list, seq_cat_list, artifact['cat_pad_indices'])
        with torch.no_grad():
            pred = model(Xn_t.to(device), Xc_t.to(device), kpm_t.to(device))
            predicted_points = float(pred.cpu().numpy().reshape(-1)[0])
    else:
        # Flat model path (backward-compatible)
        scaler = artifact['scaler']
        label_encoders = artifact['label_encoders']
        # Prepare features
        X_num = [features.get(feat, 0) for feat in feature_names['continuous']]
        X_num = scaler.transform(np.array([X_num])) if len(X_num) else np.zeros((1, 0))
        X_cat = [
            _encode_categorical_value(label_encoders[feat], features.get(feat, 'UNK'))
            for feat in feature_names['categorical']
        ]
        X_cat = np.array([X_cat], dtype=np.int64) if len(X_cat) else np.zeros((1, 0), dtype=np.int64)
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
    season -= 1
    
    # Aggregate predictions for all loaded positions
    all_predictions: Dict[str, Dict] = {}

    for pos in positions:
        if pos not in models:
            continue
        # Load player data for this week
        pos_data = load_weekly_player_data(pos, week, season)

        artifact = artifacts[pos]
        model = models[pos]

        if artifact.get('is_sequence', False):
            # Build sequences from prior weeks for players in this week's data
            df_all = pd.read_parquet(os.path.join(DATA_DIR, 'processed', f"{pos}_data.parquet"))
            # Identify player id column
            pid_col = 'player_display_name' if 'player_display_name' in pos_data.columns else (
                'player' if 'player' in pos_data.columns else 'full_name'
            )
            players = pos_data[pid_col].astype(str).unique().tolist()

            seq_num_list: List[np.ndarray] = []
            seq_cat_list: List[np.ndarray] = []
            player_keys: List[str] = []

            for pid in players:
                res = _prepare_sequence_for_player(df_all, pid, season, week, artifact)
                if res is None:
                    continue
                xn, xc = res
                if xn.shape[0] == 0:
                    continue
                seq_num_list.append(xn)
                seq_cat_list.append(xc)
                player_keys.append(pid)

            if len(seq_num_list) == 0:
                continue

            Xn_t, Xc_t, kpm_t = _collate_sequences_for_inference(seq_num_list, seq_cat_list, artifact['cat_pad_indices'])
            with torch.no_grad():
                preds = model(Xn_t.to(device), Xc_t.to(device), kpm_t.to(device)).cpu().numpy().reshape(-1)

            # Build response mapping
            for pid, pred_val in zip(player_keys, preds):
                # Find row in pos_data to extract team/opponent (first match)
                row = pos_data[pos_data[pid_col] == pid].iloc[0]
                all_predictions[pid] = {
                    'position': pos,
                    'team': row['recent_team'] if 'recent_team' in pos_data.columns else None,
                    'opponent': row['opponent_team'] if 'opponent_team' in pos_data.columns else None,
                    'predicted_points': float(np.round(pred_val, 1)),
                }
        else:
            # Flat model path
            X_num, X_cat, idx = _prepare_batch(pos_data, artifact)
            with torch.no_grad():
                X_num_tensor = torch.from_numpy(X_num).float().to(device)
                X_cat_tensor = torch.from_numpy(X_cat).long().to(device)
                preds = model(X_num_tensor, X_cat_tensor).cpu().numpy().reshape(-1)

        # Build response mapping by player name(s)
        if not artifact.get('is_sequence', False):
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
