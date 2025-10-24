# models/train_position.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import optuna
from ft_transformer import FTTransformer


class FantasyDataset(Dataset):
    """PyTorch Dataset for fantasy football data"""
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.FloatTensor(X_num)
        self.X_cat = torch.LongTensor(X_cat)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]



def prepare_data_for_position(position):
    """
    Load and prepare data for specific position.
    Aligns with preprocessing/engineering pipeline.
    """
    # Load featured data
    data = pd.read_parquet(f'data/processed/{position}_featured.parquet')
    
    # Remove rows with missing target (fantasy_points_ppr, not fantasy_points)
    data = data[data['fantasy_points_ppr'].notna()].copy()
    
    # Define continuous features (based on actual preprocessing output)
    continuous_features = [
        # NGS Receiving metrics (WR/TE primarily, but also tracked for others)
        'avg_cushion', 'avg_separation', 'avg_intended_air_yards',
        'percent_share_of_intended_air_yards', 'avg_yac', 
        'avg_yac_above_expectation', 'catch_percentage',
        
        # NGS Rushing metrics (RB primarily, but QB rushing also tracked)
        'efficiency', 'percent_attempts_gte_eight_defenders', 
        'avg_time_to_los', 'avg_rush_yards', 'rush_yards_over_expected',
        'rush_yards_over_expected_per_att', 'rush_pct_over_expected',
        
        # NGS Passing metrics (QB only)
        'avg_time_to_throw', 'avg_completed_air_yards', 
        'avg_intended_air_yards_pass', 'avg_air_yards_differential', 
        'aggressiveness', 'completion_percentage_above_expectation',
        
        # Workload & opportunity
        'offense_snaps', 'offense_pct', 'snap_pct', 'snap_pct_rolling_3',
        
        # Game context & Vegas lines
        'spread_magnitude', 'game_total', 'implied_team_total',
        'temp', 'wind',
        
        # Engineered features
        'fantasy_trend', 'opp_def_rank_vs_pos', 'injury_severity',
        'games_played', 'days_rest', 'week_of_season',
        
        # Position-specific rolling features (created in engineer_features.py)
        'redzone_touches_rolling_3',  # RB
        'air_yards_share_rolling_3',  # WR/TE
        'target_share', 'carry_share'  # Position-dependent
    ]
    
    # Define categorical features
    categorical_features = [
        'recent_team', 'opponent_team', 'position',
        'is_home', 'is_division_game', 'is_favored', 'is_dome',
        'is_early_season', 'is_late_season',
        'is_cold', 'is_hot',
        'report_status', 'practice_status', 'report_primary_injury'
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
    
    # Split by time (prevents data leakage)
    train_data = data[data['season'].isin([2019, 2020, 2021, 2022])].copy()
    val_data = data[data['season'] == 2023].copy()
    test_data = data[data['season'] == 2024].copy()
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)} samples (seasons 2019-2022)")
    print(f"  Val: {len(val_data)} samples (season 2023)")
    print(f"  Test: {len(test_data)} samples (season 2024)")
    
    # Prepare arrays
    X_num_train = train_data[continuous_features].values
    X_cat_train = train_data[categorical_features].values
    y_train = train_data['fantasy_points_ppr'].values.reshape(-1, 1)
    
    X_num_val = val_data[continuous_features].values
    X_cat_val = val_data[categorical_features].values
    y_val = val_data['fantasy_points_ppr'].values.reshape(-1, 1)
    
    X_num_test = test_data[continuous_features].values
    X_cat_test = test_data[categorical_features].values
    y_test = test_data['fantasy_points_ppr'].values.reshape(-1, 1)
    
    # Get cardinalities for categorical features
    cat_cardinalities = [
        len(label_encoders[feat].classes_) 
        for feat in categorical_features
    ]
    
    print(f"\nTarget variable (fantasy_points_ppr):")
    print(f"  Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"  Val: mean={y_val.mean():.2f}, std={y_val.std():.2f}")
    print(f"  Test: mean={y_test.mean():.2f}, std={y_test.std():.2f}")
    
    return {
        'train': (X_num_train, X_cat_train, y_train),
        'val': (X_num_val, X_cat_val, y_val),
        'test': (X_num_test, X_cat_test, y_test),
        'num_continuous': len(continuous_features),
        'cat_cardinalities': cat_cardinalities,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': {
            'continuous': continuous_features,
            'categorical': categorical_features
        }
    }



def train_model(model, train_loader, val_loader, position='model', epochs=50, lr=1e-4, device='cpu'):
    """
    Train FT-Transformer model with improved monitoring
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
        
        for X_num, X_cat, y in train_loader:
            X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_num, X_cat)
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
            for X_num, X_cat, y in val_loader:
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



def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set with detailed metrics
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_num, X_cat, y in test_loader:
            X_num, X_cat = X_num.to(device), X_cat.to(device)
            pred = model(X_num, X_cat)
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())
    
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
            
            # Optional: Hyperparameter optimization (uncomment to enable)
            # best_params = hyperparameter_optimization(data_dict, position, n_trials=20)
            # Use optimized params or defaults
            best_params = {
                'd_token': 192,
                'n_layers': 3,
                'n_heads': 8,
                'dropout': 0.15,
                'lr': 1e-4,
                'batch_size': 128
            }
            
            # Create model
            model = FTTransformer(
                num_continuous=data_dict['num_continuous'],
                cat_cardinalities=data_dict['cat_cardinalities'],
                d_token=best_params['d_token'],
                n_layers=best_params['n_layers'],
                n_heads=best_params['n_heads'],
                dropout=best_params['dropout']
            )
            
            print(f"\nModel Parameters:")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Create dataloaders
            train_dataset = FantasyDataset(*data_dict['train'])
            val_dataset = FantasyDataset(*data_dict['val'])
            test_dataset = FantasyDataset(*data_dict['test'])
            
            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
            test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
            
            # Train
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, 
                position=position, epochs=50, lr=best_params['lr'], device=device
            )
            
            # Load best model
            model.load_state_dict(torch.load(f'models/best_model_{position}.pt', map_location=device))
            
            # Evaluate
            eval_results = evaluate_model(model, test_loader, device=device)
            results_summary[position] = eval_results
            
            # Save artifacts
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': data_dict['scaler'],
                'label_encoders': data_dict['label_encoders'],
                'feature_names': data_dict['feature_names'],
                'num_continuous': data_dict['num_continuous'],
                'cat_cardinalities': data_dict['cat_cardinalities'],
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
