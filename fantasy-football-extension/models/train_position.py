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
    Load and prepare data for specific position
    """
    # Load featured data
    data = pd.read_parquet(f'data/processed/{position}_featured.parquet')
    
    # Remove rows with missing target
    data = data[data['fantasy_points'].notna()].copy()
    
    # Define continuous and categorical features
    continuous_features = [
        # Performance history
        'fantasy_points_rolling_3', 'fantasy_points_rolling_5',
        'yards_rolling_3', 'touchdowns_rolling_3',
        'fantasy_ppg_ytd', 'fantasy_trend',
        
        # Opponent
        'opp_def_rank_vs_pos',
        
        # Game context
        'spread', 'spread_magnitude', 'game_total', 'implied_team_total',
        
        # Usage
        'snap_pct_rolling_3', 'target_share', 'carry_share',
        
        # NGS features (position-specific)
        'avg_separation_rolling_3', 'efficiency_rolling_3',
        'avg_time_to_throw_rolling_3',
        
        # Temporal
        'games_played', 'week_of_season', 'days_rest'
    ]
    
    categorical_features = [
        'team', 'opponent', 'position',
        'is_home', 'is_division_game', 
        'is_favored', 'is_dome',
        'is_early_season', 'is_late_season'
    ]
    
    # Filter to available columns
    continuous_features = [f for f in continuous_features if f in data.columns]
    categorical_features = [f for f in categorical_features if f in data.columns]
    
    # Encode categorical variables
    label_encoders = {}
    for feat in categorical_features:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat].astype(str))
        label_encoders[feat] = le
    
    # Scale continuous variables
    scaler = StandardScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])
    
    # Split by time (no data leakage)
    train_data = data[data['season'].isin([2019, 2020, 2021, 2022])]
    val_data = data[data['season'] == 2023]
    test_data = data[data['season'] == 2024]
    
    # Prepare arrays
    X_num_train = train_data[continuous_features].values
    X_cat_train = train_data[categorical_features].values
    y_train = train_data['fantasy_points'].values.reshape(-1, 1)
    
    X_num_val = val_data[continuous_features].values
    X_cat_val = val_data[categorical_features].values
    y_val = val_data['fantasy_points'].values.reshape(-1, 1)
    
    X_num_test = test_data[continuous_features].values
    X_cat_test = test_data[categorical_features].values
    y_test = test_data['fantasy_points'].values.reshape(-1, 1)
    
    # Get cardinalities for categorical features
    cat_cardinalities = [
        len(label_encoders[feat].classes_) 
        for feat in categorical_features
    ]
    
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


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda'):
    """
    Train FT-Transformer model
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
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_num, X_cat, y in train_loader:
            X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_num, X_cat)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_num, X_cat, y in val_loader:
                X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                predictions = model(X_num, X_cat)
                loss = criterion(predictions, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/best_model_{position}.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    return model


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_num, X_cat, y in test_loader:
            X_num, X_cat = X_num.to(device), X_cat.to(device)
            pred = model(X_num, X_cat)
            predictions.extend(pred.cpu().numpy())
            actuals.extend(y.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - actuals.mean()) ** 2)
    
    print(f"\nTest Results:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.3f}")
    
    return mae, rmse, r2


def hyperparameter_optimization(data_dict, n_trials=50):
    """
    Use Optuna for hyperparameter tuning
    """
    def objective(trial):
        # Suggest hyperparameters
        d_token = trial.suggest_categorical('d_token', [64, 96, 128, 192, 256])
        n_layers = trial.suggest_int('n_layers', 2, 6)
        n_heads = trial.suggest_categorical('n_heads', [4, 8])
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
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
        model = train_model(model, train_loader, val_loader, epochs=30, lr=lr, device=device)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for X_num, X_cat, y in val_loader:
                X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                pred = model(X_num, X_cat)
                val_loss += criterion(pred, y).item()
        
        return val_loss / len(val_loader)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("\nBest hyperparameters:")
    print(study.best_params)
    
    return study.best_params


if __name__ == '__main__':
    # Train model for each position
    positions = ['QB', 'RB', 'WR', 'TE']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    for position in positions:
        print(f"\n{'='*50}")
        print(f"Training {position} Model")
        print(f"{'='*50}\n")
        
        # Prepare data
        data_dict = prepare_data_for_position(position)
        
        # Optional: Hyperparameter optimization
        # best_params = hyperparameter_optimization(data_dict, n_trials=20)
        
        # Use default or optimized hyperparameters
        model = FTTransformer(
            num_continuous=data_dict['num_continuous'],
            cat_cardinalities=data_dict['cat_cardinalities'],
            d_token=192,
            n_layers=3,
            n_heads=8,
            dropout=0.15
        )
        
        # Create dataloaders
        train_dataset = FantasyDataset(*data_dict['train'])
        val_dataset = FantasyDataset(*data_dict['val'])
        test_dataset = FantasyDataset(*data_dict['test'])
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128)
        test_loader = DataLoader(test_dataset, batch_size=128)
        
        # Train
        model = train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device=device)
        
        # Evaluate
        model.load_state_dict(torch.load(f'models/best_model_{position}.pt'))
        evaluate_model(model, test_loader, device=device)
        
        # Save artifacts
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': data_dict['scaler'],
            'label_encoders': data_dict['label_encoders'],
            'feature_names': data_dict['feature_names'],
            'num_continuous': data_dict['num_continuous'],
            'cat_cardinalities': data_dict['cat_cardinalities']
        }, f'models/{position}_transformer_complete.pt')
        
        print(f"\n{position} model training complete!")
