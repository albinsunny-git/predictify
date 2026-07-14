import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# In-memory cache to store predictions and metrics for tickers
# Format: { ticker: { 'timestamp': float, 'data': dict } }
_prediction_cache = {}
CACHE_DURATION_SECONDS = 3600  # Cache duration: 1 hour

def fetch_stock_data(ticker, period="5y"):
    """
    Fetches daily stock history for a given ticker from Yahoo Finance.
    """
    print(f"Fetching data for ticker: {ticker} over period: {period}")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"No stock data found for ticker '{ticker}'. It may be invalid or delisted.")
    
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    return df

def calculate_technical_indicators(df):
    """
    Calculates technical indicators: SMA, EMA, RSI, MACD, and Bollinger Bands.
    Modifies the DataFrame in-place.
    """
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)
    
    return df

def prepare_regression_data(df):
    """
    Prepares lag features and target variable for Linear Regression and Random Forest.
    We'll use closing price lags and some indicators as features to predict next-day Close.
    """
    df_features = df.copy()
    
    # Lags for Close price
    for i in range(1, 6):
        df_features[f'Close_Lag_{i}'] = df_features['Close'].shift(i)
        
    # Drop rows with NaN values arising from indicators and lags
    df_features = df_features.dropna()
    
    features = [
        'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
        'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower'
    ]
    
    X = df_features[features]
    y = df_features['Close']
    
    return X, y, df_features

def train_linear_regression(X_train, y_train, X_test, y_test, X_full):
    """
    Trains and evaluates a Linear Regression model.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    X_full_scaled = scaler.transform(X_full.values)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    preds_test = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    mae = mean_absolute_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)
    
    # Fit values on the entire dataset
    full_preds = model.predict(X_full_scaled)
    
    return model, scaler, full_preds, {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}

def train_random_forest(X_train, y_train, X_test, y_test, X_full):
    """
    Trains and evaluates a Random Forest Regressor.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    X_full_scaled = scaler.transform(X_full.values)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    preds_test = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    mae = mean_absolute_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)
    
    # Fit values on the entire dataset
    full_preds = model.predict(X_full_scaled)
    
    return model, scaler, full_preds, {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}

def train_lstm(df_close, lookback=60, split_ratio=0.8):
    """
    Trains a lightweight LSTM network on closing prices and predicts test + full history.
    """
    data = df_close.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Train-test split boundary
    train_size = int(len(scaled_data) * split_ratio)
    train_data = scaled_data[:train_size]
    
    # Create dataset sequences
    def create_sequences(dataset):
        X, y = [], []
        for i in range(lookback, len(dataset)):
            X.append(dataset[i-lookback:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_data)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Prepare all sequences for full fit and test evaluation
    # To predict from index 'lookback' onwards
    X_all, y_all = [], []
    for i in range(lookback, len(scaled_data)):
        X_all.append(scaled_data[i-lookback:i, 0])
        y_all.append(scaled_data[i, 0])
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    X_all_reshaped = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1))
    
    # Compile a lightweight LSTM model
    model = Sequential([
        LSTM(units=32, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.1),
        LSTM(units=32),
        Dropout(0.1),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Fit model (keep epochs low for rapid execution in local web app)
    model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
    
    # Full history predictions (starts at lookback index)
    scaled_full_preds = model.predict(X_all_reshaped, verbose=0)
    full_preds = scaler.inverse_transform(scaled_full_preds).flatten()
    
    # Evaluate metrics on test split
    # Test items start at train_size and go up to the end of scaled_data.
    # In sequence space, test targets correspond to index: train_size - lookback to the end
    test_start_idx = train_size - lookback
    if test_start_idx < 0:
        test_start_idx = 0
        
    X_test, y_test = create_sequences(scaled_data[test_start_idx:])
    if len(X_test) > 0:
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        scaled_test_preds = model.predict(X_test_reshaped, verbose=0)
        test_preds = scaler.inverse_transform(scaled_test_preds).flatten()
        actual_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(actual_test, test_preds))
        mae = mean_absolute_error(actual_test, test_preds)
        r2 = r2_score(actual_test, test_preds)
    else:
        rmse, mae, r2 = 0.0, 0.0, 0.0
        
    return model, scaler, full_preds, {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}

def forecast_future(df, reg_models, lstm_model, lookback=60, forecast_days=10):
    """
    Forecasts stock prices for the next `forecast_days` trading days using trained models.
    """
    # 1. Forecast with LSTM (simple auto-regressive on single Close price feature)
    lstm_net, lstm_scaler = lstm_model
    lstm_seq = df['Close'].values[-lookback:].reshape(-1, 1)
    lstm_scaled_seq = lstm_scaler.transform(lstm_seq).flatten().tolist()
    
    lstm_forecasts = []
    for _ in range(forecast_days):
        input_seq = np.array(lstm_scaled_seq[-lookback:]).reshape(1, lookback, 1)
        pred_scaled = lstm_net.predict(input_seq, verbose=0)[0][0]
        lstm_forecasts.append(pred_scaled)
        lstm_scaled_seq.append(pred_scaled)
        
    lstm_forecasted_prices = lstm_scaler.inverse_transform(
        np.array(lstm_forecasts).reshape(-1, 1)
    ).flatten().tolist()
    
    # 2. Forecast with Regression Models (Linear Regression & Random Forest)
    # We do a recursive forecasting approach updating lagging values.
    # Note: Since calculating other indicators (RSI, MACD) dynamically during recursive 
    # forecasting is highly complex, we will approximate by holding other indicators 
    # at their last known values and shifting only the closing price lag features.
    lr_model, lr_scaler = reg_models['lr']
    rf_model, rf_scaler = reg_models['rf']
    
    # Get last known features row as starter
    # Features sequence: ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5', SMA_20, SMA_50, EMA_20, RSI, MACD, BB_Upper, BB_Lower]
    last_row = df.copy().dropna().iloc[-1]
    
    # Build list of lag values (indices 0 to 4 correspond to lag 1 to lag 5)
    # At day T+1: lag1 is close(T), lag2 is close(T-1)...
    lags = [
        last_row['Close'], 
        last_row['Close_Lag_1'], 
        last_row['Close_Lag_2'], 
        last_row['Close_Lag_3'], 
        last_row['Close_Lag_4']
    ]
    
    # Retain standard technical indicator values from the last day for simplification
    other_indicators = [
        last_row['SMA_20'], last_row['SMA_50'], last_row['EMA_20'], 
        last_row['RSI'], last_row['MACD'], last_row['BB_Upper'], last_row['BB_Lower']
    ]
    
    lr_forecasts = []
    rf_forecasts = []
    
    current_lags_lr = list(lags)
    current_lags_rf = list(lags)
    
    for _ in range(forecast_days):
        # Linear Regression step
        lr_features = np.array(current_lags_lr + other_indicators).reshape(1, -1)
        lr_features_scaled = lr_scaler.transform(lr_features)
        lr_pred = lr_model.predict(lr_features_scaled)[0]
        lr_forecasts.append(lr_pred)
        
        # Random Forest step
        rf_features = np.array(current_lags_rf + other_indicators).reshape(1, -1)
        rf_features_scaled = rf_scaler.transform(rf_features)
        rf_pred = rf_model.predict(rf_features_scaled)[0]
        rf_forecasts.append(rf_pred)
        
        # Update lags for next prediction step
        current_lags_lr = [lr_pred] + current_lags_lr[:-1]
        current_lags_rf = [rf_pred] + current_lags_rf[:-1]
        
    return {
        'lr': [float(x) for x in lr_forecasts],
        'rf': [float(x) for x in rf_forecasts],
        'lstm': [float(x) for x in lstm_forecasted_prices]
    }

def get_predictions(ticker):
    """
    Main orchestrator that fetches data, computes indicators, trains models, 
    generates 10-day predictions, and returns a compiled payload. Uses cache.
    """
    now = time.time()
    if ticker in _prediction_cache:
        cached_item = _prediction_cache[ticker]
        if now - cached_item['timestamp'] < CACHE_DURATION_SECONDS:
            print(f"Returning cached predictions for ticker: {ticker}")
            return cached_item['data']
            
    # Fetch and compute
    df = fetch_stock_data(ticker, period="5y")
    df = calculate_technical_indicators(df)
    
    # Prepare Regression data
    X_reg, y_reg, df_reg = prepare_regression_data(df)
    
    # Train-test split for regression (sequential split)
    split_idx = int(len(X_reg) * 0.8)
    X_train, X_test = X_reg.iloc[:split_idx], X_reg.iloc[split_idx:]
    y_train, y_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    
    # Train models
    lr_model, lr_scaler, lr_fitted, lr_metrics = train_linear_regression(
        X_train, y_train, X_test, y_test, X_reg
    )
    rf_model, rf_scaler, rf_fitted, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test, X_reg
    )
    
    # Train LSTM (runs on closing price only)
    lstm_lookback = 60
    lstm_net, lstm_scaler, lstm_fitted, lstm_metrics = train_lstm(
        df['Close'], lookback=lstm_lookback, split_ratio=0.8
    )
    
    # Forecast next 10 days
    reg_models = {'lr': (lr_model, lr_scaler), 'rf': (rf_model, rf_scaler)}
    lstm_model = (lstm_net, lstm_scaler)
    forecasts = forecast_future(df_reg, reg_models, lstm_model, lookback=lstm_lookback, forecast_days=10)
    
    # Align fitted series with original DataFrame index
    # Note: df_reg has dropped rows at the start due to indicators and lags.
    # Linear Regression and Random Forest fitted arrays start at index 5 + indicator buffer.
    # LSTM fitted array starts at index `lstm_lookback` (60).
    
    # We will return the historical records as lists of dates and prices
    historical_dates = df.index.strftime('%Y-%m-%d').tolist()
    historical_open = df['Open'].tolist()
    historical_high = df['High'].tolist()
    historical_low = df['Low'].tolist()
    historical_close = df['Close'].tolist()
    
    # Compile technical indicators
    sma_20 = df['SMA_20'].fillna("").tolist()
    sma_50 = df['SMA_50'].fillna("").tolist()
    ema_20 = df['EMA_20'].fillna("").tolist()
    rsi = df['RSI'].fillna("").tolist()
    macd = df['MACD'].fillna("").tolist()
    macd_signal = df['MACD_Signal'].fillna("").tolist()
    bb_upper = df['BB_Upper'].fillna("").tolist()
    bb_lower = df['BB_Lower'].fillna("").tolist()
    bb_middle = df['BB_Middle'].fillna("").tolist()
    
    # Align fitted arrays by padding with None values so they map 1-to-1 with df index
    # df_reg indexes align with df, but start later.
    reg_start_idx = df.index.get_loc(df_reg.index[0])
    lr_full_fitted = [None] * reg_start_idx + [float(x) for x in lr_fitted]
    rf_full_fitted = [None] * reg_start_idx + [float(x) for x in rf_fitted]
    
    # LSTM start index is lstm_lookback
    lstm_full_fitted = [None] * lstm_lookback + [float(x) for x in lstm_fitted]
    
    # Generate future dates (next 10 business days)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=15, freq='B')[:10]
    future_dates_str = future_dates.strftime('%Y-%m-%d').tolist()
    
    payload = {
        'ticker': ticker.upper(),
        'dates': historical_dates,
        'open': historical_open,
        'high': historical_high,
        'low': historical_low,
        'close': historical_close,
        'indicators': {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_20': ema_20,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle
        },
        'fitted': {
            'lr': lr_full_fitted,
            'rf': rf_full_fitted,
            'lstm': lstm_full_fitted
        },
        'forecast': {
            'dates': future_dates_str,
            'lr': forecasts['lr'],
            'rf': forecasts['rf'],
            'lstm': forecasts['lstm']
        },
        'metrics': {
            'lr': lr_metrics,
            'rf': rf_metrics,
            'lstm': lstm_metrics
        }
    }
    
    # Store in cache
    _prediction_cache[ticker] = {
        'timestamp': now,
        'data': payload
    }
    
    return payload
