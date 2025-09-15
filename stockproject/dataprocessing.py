from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Constants for preprocessing
SEQUENCE_LENGTH = 30
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'Returns']

def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Returns'] = df['Close'].pct_change()
    
    return df

def dataprocessing(tech_list):
    end = datetime.now()
    start = end - timedelta(days=365*15)

    stock_data = {}
    for stock in tech_list:
        df = yf.download(stock, start, end)
        df = df.sort_index()
        df = add_technical_indicators(df)
        stock_data[stock] = df.dropna()

    scalers = {}
    x_train = {}
    y_train = {}
    x_test = {}
    y_test = {}
    shape = None

    for stock in tech_list:
        df = stock_data[stock]
        if df.empty:
            raise ValueError(f"No data downloaded for {stock}")

        data_values = df[FEATURES].values

        # Chronological split index (80% train, 20% test)
        split_idx = int(len(data_values) * 0.8)

        # Fit scaler ONLY on training portion to avoid leakage
        scaler = StandardScaler()
        scaler.fit(data_values[:split_idx])
        scalers[stock] = scaler

        # Transform the entire series with train-fitted scaler
        scaled_data = scaler.transform(data_values)

        # Build sequences over scaled data
        x_all, y_all = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_data)):
            x_all.append(scaled_data[i - SEQUENCE_LENGTH:i])
            # target is scaled Close (index 3)
            y_all.append(scaled_data[i, 3])
        x_all = np.array(x_all)
        y_all = np.array(y_all)

        # Align split for sequences
        split_seq_idx = max(0, split_idx - SEQUENCE_LENGTH)
        x_train[stock] = x_all[:split_seq_idx]
        y_train[stock] = y_all[:split_seq_idx]
        x_test[stock] = x_all[split_seq_idx:]
        y_test[stock] = y_all[split_seq_idx:]

        # Ensure consistent shape across stocks
        if shape is not None and x_train[stock].shape[1:] != shape:
            raise ValueError(f"Shape mismatch for {stock}: expected {shape}, got {x_train[stock].shape[1:]}")
        shape = x_train[stock].shape[1:]

    metadata = {
        'features': FEATURES,
        'sequence_length': SEQUENCE_LENGTH,
        'shape': shape,
    }

    return x_train, y_train, x_test, y_test, metadata, scalers