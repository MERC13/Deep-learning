import os
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request, url_for
from tensorflow.keras.models import load_model

from dataprocessing import add_technical_indicators

app = Flask(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


MODEL_FILE = 'best_stock_model.h5'
SCALERS_FILE = 'scalers.joblib'
METADATA_FILE = 'preprocess_metadata.json'

_model = None
_scalers = None
_metadata = None


def load_artifacts():
    global _model, _scalers, _metadata
    if _model is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}. Please run training first.")
        _model = load_model(MODEL_FILE)
        logging.info("Model loaded.")
    if _scalers is None:
        import joblib
        if not os.path.exists(SCALERS_FILE):
            raise FileNotFoundError(f"Scalers file not found: {SCALERS_FILE}. Please run training first.")
        _scalers = joblib.load(SCALERS_FILE)
        logging.info("Scalers loaded.")
    if _metadata is None:
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}. Please run training first.")
        with open(METADATA_FILE, 'r') as f:
            _metadata = json.load(f)
        logging.info("Metadata loaded.")


def list_available_graphs():
    # Collect PNGs from static with known naming
    items = []
    try:
        for name in os.listdir('static'):
            if name.endswith('.png'):
                title = name.replace('_', ' ').replace('.png', '')
                items.append({
                    'title': title,
                    'filename': name,
                    'url': url_for('static', filename=name)
                })
    except Exception as e:
        logging.warning(f"Could not list static graphs: {e}")
    # keep a stable order
    items.sort(key=lambda x: x['filename'])
    return items


@app.route("/")
def home():
    graphs = list_available_graphs()
    return render_template("index.html", graphs=graphs)


def build_inference_window(symbol: str, start_date: str = None, end_date: str = None):
    # Use last 120 days if not provided
    if end_date:
        end = datetime.fromisoformat(end_date)
    else:
        end = datetime.now()
    if start_date:
        start = datetime.fromisoformat(start_date)
    else:
        start = end - timedelta(days=200)

    df = yf.download(symbol, start, end)
    df = df.sort_index()
    df = add_technical_indicators(df).dropna()
    if df.empty:
        raise ValueError("No data available for given range")

    features = _metadata['features']
    seq_len = _metadata['sequence_length']
    values = df[features].values

    # Use stock-specific scaler if available; fallback to first scaler
    scaler = _scalers.get(symbol) if isinstance(_scalers, dict) else None
    if scaler is None:
        # try normalized symbol variants
        for k in _scalers.keys():
            if k.replace('^', '').upper() == symbol.replace('^', '').upper():
                scaler = _scalers[k]
                break
    if scaler is None:
        # last resort: pick an arbitrary scaler
        scaler = list(_scalers.values())[0]
        logging.warning(f"Scaler for {symbol} not found. Using default scaler.")

    scaled = scaler.transform(values)
    if len(scaled) < seq_len:
        raise ValueError(f"Insufficient data for sequence length {seq_len}")
    window = scaled[-seq_len:]
    x = np.expand_dims(window, axis=0)
    return x, scaler


@app.route("/predict", methods=['POST'])
def predict():
    try:
        load_artifacts()
    except Exception as e:
        logging.exception("Artifacts loading failed")
        return jsonify({"error": str(e)}), 500

    payload = request.get_json(force=True, silent=True) or {}
    symbol = payload.get('symbol', '^DJI')
    start_date = payload.get('startDate')
    end_date = payload.get('endDate')

    try:
        x, scaler = build_inference_window(symbol, start_date, end_date)
        pred_scaled = _model.predict(x)
        pred_scaled = float(pred_scaled.squeeze())
        # inverse transform using Close position index 3
        n_features = _metadata['shape'][1]
        tmp = np.zeros((1, n_features))
        tmp[0, 3] = pred_scaled
        pred_price = float(scaler.inverse_transform(tmp)[0, 3])

        # Build a simple chart data using available recent close values
        # Note: This is for display; a richer series could be returned by fetching a period.
        labels = []
        actual = []
        # Use last 30 closes for display
        end = datetime.fromisoformat(end_date) if end_date else datetime.now()
        start = (datetime.fromisoformat(start_date) if start_date else end - timedelta(days=200))
        df = yf.download(symbol, start, end)
        df = df.dropna()
        last = df['Close'].tail(30)
        labels = [d.strftime('%Y-%m-%d') for d in last.index]
        actual = last.tolist()
        predicted = actual[:-1] + [pred_price]

        chart_data = {"labels": labels, "actual": actual, "predicted": predicted}
        return jsonify({"prediction": pred_price, "chartData": chart_data})
    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    setup_logging()
    # Do not enable debug in production
    app.run(debug=False, port=5001)