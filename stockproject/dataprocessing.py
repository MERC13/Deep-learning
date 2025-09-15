from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to a price dataframe.

    Expects columns: ['Open','High','Low','Close','Volume'].
    """
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Returns'] = df['Close'].pct_change()
    return df


def _build_sequences(scaled_array: np.ndarray, sequence_length: int, target_index: int) -> tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for i in range(sequence_length, len(scaled_array)):
        x_list.append(scaled_array[i - sequence_length:i])
        y_list.append(scaled_array[i, target_index])
    if not x_list:
        return np.empty((0, sequence_length, scaled_array.shape[1])), np.empty((0,))
    return np.asarray(x_list), np.asarray(y_list)


def dataprocessing(tech_list: list[str], sequence_length: int = 30):
    """Prepare train/test sequences per stock without data leakage.

    - Fetch ~15 years of data per symbol.
    - Compute features.
    - Chronologically split into train/test (last 20% as test).
    - Fit a StandardScaler on TRAIN ONLY per stock, then transform train/test.

    Returns:
      x_train, y_train, x_test, y_test: dict[symbol] -> np.ndarray
      shape: tuple[int, int] = (timesteps, features)
      scalers: dict[symbol] -> StandardScaler
      features: list[str]
      sequence_length: int
    """
    end = datetime.now()
    start = end - timedelta(days=365 * 15)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'Returns']
    target_index = features.index('Close')

    x_train: dict[str, np.ndarray] = {}
    y_train: dict[str, np.ndarray] = {}
    x_test: dict[str, np.ndarray] = {}
    y_test: dict[str, np.ndarray] = {}
    scalers: dict[str, StandardScaler] = {}
    shape: tuple[int, int] | None = None

    for stock in tech_list:
        df = yf.download(stock, start, end, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data downloaded for {stock}")

        df = df.sort_index()
        df = add_technical_indicators(df)
        df = df.dropna()
        if df.empty:
            raise ValueError(f"Insufficient data after indicators for {stock}")

        values = df[features].values

        # Chronological split: last 20% as test
        split_idx = int(len(values) * 0.8)
        train_values = values[:split_idx]
        test_values = values[split_idx:]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_values)
        test_scaled = scaler.transform(test_values)

        scalers[stock] = scaler

        # Sequences are created within each split separately to avoid leakage across the boundary
        x_tr, y_tr = _build_sequences(train_scaled, sequence_length, target_index)
        x_te, y_te = _build_sequences(test_scaled, sequence_length, target_index)

        if x_tr.size == 0 or x_te.size == 0:
            raise ValueError(f"Not enough data to build sequences for {stock}")

        x_train[stock], y_train[stock] = x_tr, y_tr
        x_test[stock], y_test[stock] = x_te, y_te

        if shape is not None and x_tr.shape[1:] != shape:
            raise ValueError(f"Shape mismatch for {stock}: expected {shape}, got {x_tr.shape[1:]}")
        shape = x_tr.shape[1:]

    metadata = {
        'shape': shape,
        'features': features,
        'sequence_length': sequence_length,
    }
    return x_train, y_train, x_test, y_test, metadata, scalers