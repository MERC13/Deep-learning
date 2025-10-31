# models/train_position.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import optuna
from ft_transformer import FTTransformer
from temporal_transformer import TemporalTransformer


class FantasyDataset(Dataset):
    """PyTorch Dataset for flat (non-sequence) fantasy football data"""
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.FloatTensor(X_num)
        self.X_cat = torch.LongTensor(X_cat)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


class SequenceFantasyDataset(Dataset):
    """Dataset returning variable-length sequences of prior weeks per player-week.

    Each item:
        X_num_seq: (T, Cn) float tensor of prior weeks' numeric features
        X_cat_seq: (T, Cc) long tensor of prior weeks' categorical features
        y:         (1,) float tensor fantasy points for week n
    """
    def __init__(self, seq_X_num: list, seq_X_cat: list, y: np.ndarray):
        assert len(seq_X_num) == len(seq_X_cat) == len(y)
        self.seq_X_num = seq_X_num
        self.seq_X_cat = seq_X_cat
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xnum = torch.tensor(self.seq_X_num[idx], dtype=torch.float32)
        xcat = torch.tensor(self.seq_X_cat[idx], dtype=torch.long)
        return xnum, xcat, self.y[idx]


def sequence_collate_fn(batch, pad_cat_indices: np.ndarray):
    """Pad variable-length sequences in a batch.

    Args:
        batch: list of (xnum, xcat, y). xnum=(T,Cn), xcat=(T,Cc)
        pad_cat_indices: array of length num_categorical giving padding index per categorical feature
    Returns:
        xnum_pad: (B, T_max, Cn)
        xcat_pad: (B, T_max, Cc)
        y: (B,1)
        key_padding_mask: (B, T_max) bool (True for pads)
    """
    xs_num, xs_cat, ys = zip(*batch)
    B = len(batch)
    T_max = max(x.shape[0] for x in xs_num)
    Cn = xs_num[0].shape[1]
    Cc = xs_cat[0].shape[1] if xs_cat[0].ndim == 2 else 0

    xnum_pad = torch.zeros((B, T_max, Cn), dtype=torch.float32)
    xcat_pad = torch.zeros((B, T_max, Cc), dtype=torch.long)
    key_padding_mask = torch.ones((B, T_max), dtype=torch.bool)  # start as all pad

    for i, (xn, xc) in enumerate(zip(xs_num, xs_cat)):
        T = xn.shape[0]
        xnum_pad[i, :T, :] = xn
        if Cc > 0:
            xcat_pad[i, :T, :] = xc
            # pad tail with per-feature pad indices
            if T < T_max:
                pad_row = torch.tensor(pad_cat_indices, dtype=torch.long).unsqueeze(0).expand(T_max - T, -1)
                xcat_pad[i, T:, :] = pad_row
        key_padding_mask[i, :T] = False  # False for valid tokens

    y = torch.stack(ys).view(-1, 1)
    return xnum_pad, xcat_pad, y, key_padding_mask



def prepare_data_for_position(position):
    """
    Load and prepare data for specific position.
    Aligns with preprocessing/engineering pipeline.
    """
    # Load data
    data = pd.read_parquet(f'data/processed/{position}_data.parquet')
    
    # Remove rows with missing target (fantasy_points_ppr, not fantasy_points)
    # data = data[data['fantasy_points_ppr'].notna()].copy()
    
    # Define continuous features (based on actual preprocessing output)
    continuous_features = [
        # NGS Receiving metrics (WR/TE primarily, but also tracked for others)
        'avg_cushion', 'avg_separation', 'avg_intended_air_yards',
        'percent_share_of_intended_air_yards', 'avg_yac', 'avg_expected_yac',
        'avg_yac_above_expectation', 'catch_percentage', 'receptions',
        'targets', 'yards', 'rec_touchdowns',

        # NGS Rushing metrics
        'efficiency', 'percent_attempts_gte_eight_defenders', 'avg_time_to_los',
        'avg_rush_yards', 'rush_yards_over_expected', 'rush_attempts',
        'rush_yards', 'rush_touchdowns', 'expected_rush_yards',

        # NGS Passing metrics
        'avg_time_to_throw', 'avg_completed_air_yards', 'avg_intended_air_yards_pass',
        'avg_air_yards_differential', 'aggressiveness', 'completion_percentage',
        'avg_air_distance', 'max_air_distance', 'attempts', 'pass_yards',
        'pass_touchdowns', 'interceptions', 'passer_rating', 'completions',

        # Workload & opportunity
        'offense_snaps', 'offense_pct', 'st_snaps', 'st_pct'
    ]
    
    # Define categorical features
    categorical_features = [
        'recent_team', 'opponent_team', 'position',
        # Injury-related categorical fields (from raw merges)
        'report_status', 'practice_status', 'report_primary_injury',
        # Depth chart team when available
        'depth_team'
    ]
    
    # Filter to available columns (handle position-specific features)
    continuous_features = [f for f in continuous_features if f in data.columns]
    categorical_features = [f for f in categorical_features if f in data.columns]
    
    print(f"Using {len(continuous_features)} continuous features")
    print(f"Using {len(categorical_features)} categorical features")
    
    # Handle missing values before encoding/scaling
    # Fill remaining NaNs in continuous features with median
    for feat in continuous_features:
        if data[feat].isna().any():
            median_val = data[feat].median()
            data[feat] = data[feat].fillna(median_val)
            print(f"  Filled {feat} NaNs with median: {median_val:.3f}")
    
    # Fill categorical NaNs with mode
    for feat in categorical_features:
        if data[feat].isna().any():
            mode_val = data[feat].mode()[0]
            data[feat] = data[feat].fillna(mode_val)
            print(f"  Filled {feat} NaNs with mode: {mode_val}")
    
    # Encode categorical variables
    label_encoders = {}
    for feat in categorical_features:
        le = LabelEncoder()
        # Handle any remaining non-string values
        data[feat] = data[feat].astype(str)
        data[feat] = le.fit_transform(data[feat])
        label_encoders[feat] = le
        print(f"  Encoded {feat}: {len(le.classes_)} unique values")
    
    # Scale continuous variables (AFTER any NaN filling)
    scaler = StandardScaler()
    data_cont = data[continuous_features].copy()
    data[continuous_features] = scaler.fit_transform(data_cont)
    
    # Build sequences: for week n sample, inputs are weeks [< n] for that player
    # We'll split by the TARGET season (week n row's season)
    def build_sequences(df: pd.DataFrame):
        seq_X_num, seq_X_cat, ys, meta = [], [], [], []
        # We require identifiers
        pid_col = 'player_display_name' if 'player_display_name' in df.columns else 'player' if 'player' in df.columns else 'full_name'
        for (pid, season), g in df.groupby([pid_col, 'season'], dropna=False):
            g = g.sort_values('week')
            Xn = g[continuous_features].values
            Xc = g[categorical_features].values if categorical_features else np.zeros((len(g), 0), dtype=np.int64)
            yt = g['fantasy_points_ppr'].values.reshape(-1, 1)
            weeks = g['week'].values
            # For each index i >=1, build a sample with prior weeks 0..i-1 -> target y_i
            for i in range(1, len(g)):
                seq_X_num.append(Xn[:i, :].astype(np.float32))
                seq_X_cat.append(Xc[:i, :].astype(np.int64))
                ys.append(yt[i])
                meta.append({'player': pid, 'season': season, 'week': int(weeks[i])})
        return seq_X_num, seq_X_cat, np.array(ys).astype(np.float32), meta

    # Split by season of the TARGET week (the week being predicted)
    seasons = data['season'].values
    train_mask = data['season'].isin([2019, 2020, 2021, 2022])
    val_mask = data['season'] == 2023
    test_mask = data['season'] == 2024

    print(f"\nConstructing sequences (prior weeks only) ...")
    seq_X_num_all, seq_X_cat_all, ys_all, meta_all = build_sequences(data)
    # Convert meta to arrays for season-based split
    meta_df = pd.DataFrame(meta_all)
    train_idx = meta_df['season'].isin([2019, 2020, 2021, 2022]).values
    val_idx = (meta_df['season'] == 2023).values
    test_idx = (meta_df['season'] == 2024).values

    seq_X_num_train = [seq_X_num_all[i] for i in range(len(seq_X_num_all)) if train_idx[i]]
    seq_X_cat_train = [seq_X_cat_all[i] for i in range(len(seq_X_cat_all)) if train_idx[i]]
    y_train = ys_all[train_idx]

    seq_X_num_val = [seq_X_num_all[i] for i in range(len(seq_X_num_all)) if val_idx[i]]
    seq_X_cat_val = [seq_X_cat_all[i] for i in range(len(seq_X_cat_all)) if val_idx[i]]
    y_val = ys_all[val_idx]

    seq_X_num_test = [seq_X_num_all[i] for i in range(len(seq_X_num_all)) if test_idx[i]]
    seq_X_cat_test = [seq_X_cat_all[i] for i in range(len(seq_X_cat_all)) if test_idx[i]]
    y_test = ys_all[test_idx]

    # Get cardinalities for categorical features and add +1 for padding index
    cat_cardinalities = [len(label_encoders[feat].classes_) for feat in categorical_features]
    cat_cardinalities_padded = [c + 1 for c in cat_cardinalities]

    print(f"\nTarget variable (fantasy_points_ppr) by split:")
    for name, arr in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        if len(arr) > 0:
            print(f"  {name}: n={len(arr)} mean={arr.mean():.2f}, std={arr.std():.2f}")
        else:
            print(f"  {name}: n=0")

    return {
        'train_seq': (seq_X_num_train, seq_X_cat_train, y_train),
        'val_seq': (seq_X_num_val, seq_X_cat_val, y_val),
        'test_seq': (seq_X_num_test, seq_X_cat_test, y_test),
        'num_continuous': len(continuous_features),
        'cat_cardinalities_padded': cat_cardinalities_padded,
        'cat_pad_indices': np.array(cat_cardinalities, dtype=np.int64),  # original max index + 1 used as pad
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': {
            'continuous': continuous_features,
            'categorical': categorical_features
        }
    }



def train_model(model, train_loader, val_loader, position='model', epochs=50, lr=1e-4, device='cpu', is_sequence: bool = False):
    """
    Train FT-Transformer model with improved monitoring
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            if is_sequence:
                X_num, X_cat, y, key_padding_mask = batch
                X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                key_padding_mask = key_padding_mask.to(device)
                preds = model(X_num, X_cat, key_padding_mask)
            else:
                X_num, X_cat, y = batch
                X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                preds = model(X_num, X_cat)

            optimizer.zero_grad()
            predictions = preds
            loss = criterion(predictions, y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        train_loss /= batch_count
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if is_sequence:
                    X_num, X_cat, y, key_padding_mask = batch
                    X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                    key_padding_mask = key_padding_mask.to(device)
                    predictions = model(X_num, X_cat, key_padding_mask)
                else:
                    X_num, X_cat, y = batch
                    X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                    predictions = model(X_num, X_cat)
                loss = criterion(predictions, y)
                val_loss += loss.item()
                batch_count += 1
        
        val_loss /= batch_count
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping with patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/best_model_{position}.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch} (patience exceeded)")
                break
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Best={best_val_loss:.4f}")
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    return model, train_losses, val_losses



def evaluate_model(model, test_loader, device='cpu', is_sequence: bool = False):
    """
    Evaluate model on test set with detailed metrics
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            if is_sequence:
                X_num, X_cat, y, key_padding_mask = batch
                X_num, X_cat = X_num.to(device), X_cat.to(device)
                key_padding_mask = key_padding_mask.to(device)
                pred = model(X_num, X_cat, key_padding_mask)
                actuals.extend(y.numpy().flatten())
            else:
                X_num, X_cat, y = batch
                X_num, X_cat = X_num.to(device), X_cat.to(device)
                pred = model(X_num, X_cat)
                actuals.extend(y.numpy().flatten())
            predictions.extend(pred.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / (np.abs(actuals) + 1))) * 100
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - actuals.mean()) ** 2)
    
    # Percentile errors
    percentile_errors = np.percentile(np.abs(predictions - actuals), [50, 90, 95])
    
    print(f"\n{'='*50}")
    print(f"Test Results")
    print(f"{'='*50}")
    print(f"  MAE:  {mae:.2f} PPR")
    print(f"  RMSE: {rmse:.2f} PPR")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.3f}")
    print(f"\n  Median Absolute Error: {percentile_errors[0]:.2f} PPR")
    print(f"  90th Percentile Error: {percentile_errors[1]:.2f} PPR")
    print(f"  95th Percentile Error: {percentile_errors[2]:.2f} PPR")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }



def hyperparameter_optimization(data_dict, position, n_trials=20):
    """
    Use Optuna for hyperparameter tuning
    """
    def objective(trial):
        # Suggest hyperparameters
        d_token = trial.suggest_categorical('d_token', [96, 128, 192, 256])
        n_layers = trial.suggest_int('n_layers', 2, 5)
        n_heads = trial.suggest_categorical('n_heads', [4, 8])
        dropout = trial.suggest_float('dropout', 0.05, 0.25)
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        
        # Create model
        model = FTTransformer(
            num_continuous=data_dict['num_continuous'],
            cat_cardinalities=data_dict['cat_cardinalities'],
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Create dataloaders
        train_dataset = FantasyDataset(*data_dict['train'])
        val_dataset = FantasyDataset(*data_dict['val'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, _ = train_model(model, train_loader, val_loader, position=position, epochs=30, lr=lr, device=device)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        criterion = nn.MSELoss()
        batch_count = 0
        
        with torch.no_grad():
            for X_num, X_cat, y in val_loader:
                X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                pred = model(X_num, X_cat)
                val_loss += criterion(pred, y).item()
                batch_count += 1
        
        return val_loss / batch_count
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest hyperparameters for {position}:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")
    
    return study.best_params



if __name__ == '__main__':
    # Train model for each position
    positions = ['QB', 'RB', 'WR', 'TE']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("Note: Training on CPU. This will be slower than GPU training.\n")
    else:
        print()
    
    results_summary = {}
    
    for position in positions:
        print(f"\n{'='*60}")
        print(f"Training {position} Model")
        print(f"{'='*60}\n")
        
        try:
            # Prepare data
            data_dict = prepare_data_for_position(position)
            
            # Optional: Hyperparameter optimization
            # best_params = hyperparameter_optimization(data_dict, position, n_trials=20)

            # Resume-from-checkpoint logic
            complete_ckpt_path = f"models/{position}_transformer_complete.pt"
            best_only_path = f"models/best_model_{position}.pt"

            loaded_from_checkpoint = False
            checkpoint = None

            if os.path.exists(complete_ckpt_path):
                print(f"Found existing checkpoint: {complete_ckpt_path}. Resuming training from it.")
                checkpoint = torch.load(complete_ckpt_path, map_location=device)
                # Use saved hyperparameters when available
                saved_hparams = checkpoint.get('hyperparameters', {})
                best_params = {
                    'd_token': saved_hparams.get('d_token', 192),
                    'n_layers': saved_hparams.get('n_layers', 3),
                    'n_heads': saved_hparams.get('n_heads', 8),
                    'dropout': saved_hparams.get('dropout', 0.15),
                    'lr': saved_hparams.get('lr', 1e-4),
                    'batch_size': saved_hparams.get('batch_size', 128),
                }
                loaded_from_checkpoint = True
            else:
                # Fallback defaults if no complete checkpoint
                best_params = {
                    'd_token': 192,
                    'n_layers': 3,
                    'n_heads': 8,
                    'dropout': 0.15,
                    'lr': 1e-4,
                    'batch_size': 128
                }

            # Create model (new or from checkpoint)
            # Use temporal transformer (sequence model) to ensure only prior weeks are used as input
            model = TemporalTransformer(
                num_continuous=data_dict['num_continuous'],
                cat_cardinalities_padded=data_dict['cat_cardinalities_padded'],
                d_model=best_params['d_token'],
                n_layers=best_params['n_layers'],
                n_heads=best_params['n_heads'],
                dropout=best_params['dropout'],
                max_len=64,
            )

            if loaded_from_checkpoint and checkpoint is not None:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model weights from complete checkpoint.")
                except Exception as e:
                    print(f"Warning: Failed to load complete checkpoint weights ({e}).")
                    if os.path.exists(best_only_path):
                        try:
                            model.load_state_dict(torch.load(best_only_path, map_location=device))
                            print("Loaded weights from best-only checkpoint as fallback.")
                        except Exception as e2:
                            print(f"Warning: Failed to load best-only checkpoint as well ({e2}). Starting from scratch.")
            elif os.path.exists(best_only_path):
                try:
                    model.load_state_dict(torch.load(best_only_path, map_location=device))
                    print(f"Found best model weights at {best_only_path}. Resuming training from them.")
                except Exception as e:
                    print(f"Warning: Failed to load best-only checkpoint ({e}). Starting from scratch.")
            
            print(f"\nModel Parameters:")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Create sequence dataloaders with padding collate_fn
            train_dataset = SequenceFantasyDataset(*data_dict['train_seq'])
            val_dataset = SequenceFantasyDataset(*data_dict['val_seq'])
            test_dataset = SequenceFantasyDataset(*data_dict['test_seq'])

            pad_idx = data_dict['cat_pad_indices'] if len(data_dict['cat_pad_indices']) > 0 else np.array([])
            collate = (lambda batch: sequence_collate_fn(batch, pad_idx))

            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, collate_fn=collate)
            val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], collate_fn=collate)
            test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], collate_fn=collate)
            
            # Train
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader,
                position=position, epochs=50, lr=best_params['lr'], device=device, is_sequence=True
            )
            
            # Load best model
            if os.path.exists(f'models/best_model_{position}.pt'):
                model.load_state_dict(torch.load(f'models/best_model_{position}.pt', map_location=device))
            
            # Evaluate
            eval_results = evaluate_model(model, test_loader, device=device, is_sequence=True)
            results_summary[position] = eval_results
            
            # Save artifacts
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': data_dict['scaler'],
                'label_encoders': data_dict['label_encoders'],
                'feature_names': data_dict['feature_names'],
                'num_continuous': data_dict['num_continuous'],
                'cat_cardinalities_padded': data_dict['cat_cardinalities_padded'],
                'hyperparameters': best_params,
                'eval_results': eval_results
            }, f'models/{position}_transformer_complete.pt')
            
            print(f"\n{position} model training complete!")
            
        except Exception as e:
            print(f"\nError training {position} model:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    for pos, results in results_summary.items():
        print(f"\n{pos}:")
        print(f"  MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}, R²: {results['r2']:.3f}")
