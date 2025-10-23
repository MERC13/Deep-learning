# api/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import sys
sys.path.append('../models')
from ft_transformer import FTTransformer

app = Flask(__name__)
CORS(app)  # Allow Edge extension to call API

# Load models for each position
models = {}
artifacts = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

positions = ['QB', 'RB', 'WR', 'TE']
for pos in positions:
    # Load model artifacts
    checkpoint = torch.load(f'../models/{pos}_transformer_complete.pt', map_location=device)
    
    # Initialize model
    model = FTTransformer(
        num_continuous=checkpoint['num_continuous'],
        cat_cardinalities=checkpoint['cat_cardinalities'],
        d_token=192,
        n_layers=3,
        n_heads=8
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    models[pos] = model
    artifacts[pos] = checkpoint

print(f"Loaded models for {len(models)} positions")


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
        if val in label_encoders[feat].classes_:
            X_cat.append(label_encoders[feat].transform([val])[0])
        else:
            X_cat.append(0)
    X_cat = np.array([X_cat])
    
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
    
    # Load current week data (would fetch from database in production)
    all_predictions = {}
    
    for pos in positions:
        # Load player data for this week
        # In production, this would query your database
        pos_data = load_weekly_player_data(pos, week, season)
        
        # Generate predictions
        model = models[pos]
        artifact = artifacts[pos]
        
        for idx, player_row in pos_data.iterrows():
            # Prepare features
            X_num, X_cat = prepare_features(player_row, artifact)
            
            # Predict
            with torch.no_grad():
                X_num_tensor = torch.FloatTensor([X_num]).to(device)
                X_cat_tensor = torch.LongTensor([X_cat]).to(device)
                prediction = model(X_num_tensor, X_cat_tensor)
                predicted_points = float(prediction.cpu().numpy()[0][0])
            
            all_predictions[player_row['player_name']] = {
                'position': pos,
                'team': player_row['team'],
                'opponent': player_row['opponent'],
                'predicted_points': round(predicted_points, 1)
            }
    
    return jsonify(all_predictions)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': len(models)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
