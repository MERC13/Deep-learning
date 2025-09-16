from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import requests
import warnings


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


def fetch_data_yfinance(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        if not df.empty:
            return df
    except Exception as e:
        print(f"yfinance failed for {symbol}: {e}")
    return pd.DataFrame()


def fetch_data_alpha_vantage(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data using Alpha Vantage API (free tier, no API key required for basic usage)."""
    try:
        # Using the free demo endpoint - replace with your API key if you have one
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&datatype=csv"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and 'timestamp' in response.text.lower():
            df = pd.read_csv(url)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Rename columns to match yfinance format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Filter by date range
            df = df[(df.index >= start) & (df.index <= end)]
            
            if not df.empty:
                return df
    except Exception as e:
        print(f"Alpha Vantage failed for {symbol}: {e}")
    return pd.DataFrame()


def generate_sample_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Generate synthetic stock data for testing purposes."""
    print(f"Generating sample data for {symbol} (for testing only)")
    
    # Create date range
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
    
    n_days = len(dates)
    
    # Generate realistic stock price movements
    initial_price = 100 + (hash(symbol) % 200)  # Base price between 100-300
    returns = np.random.normal(0.0008, 0.02, n_days)  # Daily returns ~0.08% mean, 2% std
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = {
        'Open': prices * np.random.uniform(0.995, 1.005, n_days),
        'High': prices * np.random.uniform(1.005, 1.03, n_days),
        'Low': prices * np.random.uniform(0.97, 0.995, n_days),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days)
    }
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    for i in range(n_days):
        data['High'][i] = max(data['High'][i], data['Open'][i], data['Close'][i])
        data['Low'][i] = min(data['Low'][i], data['Open'][i], data['Close'][i])
    
    df = pd.DataFrame(data, index=dates)
    return df


def fetch_stock_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch stock data using multiple sources with fallbacks."""
    print(f"Fetching data for {symbol}...")
    
    # Try yfinance first
    df = fetch_data_yfinance(symbol, start, end)
    if not df.empty:
        print(f"✓ Successfully fetched {symbol} data from yfinance")
        return df
    
    # Try Alpha Vantage
    df = fetch_data_alpha_vantage(symbol, start, end)
    if not df.empty:
        print(f"✓ Successfully fetched {symbol} data from Alpha Vantage")
        return df
    
    # Generate sample data as last resort
    warnings.warn(f"Using synthetic data for {symbol} - real APIs unavailable")
    return generate_sample_data(symbol, start, end)


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
        # Use our multi-source data fetcher
        df = fetch_stock_data(stock, start, end)
            
        if df is None or df.empty:
            raise ValueError(f"No data could be fetched for {stock} from any source")

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