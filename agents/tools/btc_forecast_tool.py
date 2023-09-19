import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def train_bitcoin_forecaster(n_days=5, save_model=True):
    # Fetch data
    df = yf.download('BTC-USD', start='2020-01-01', end=pd.Timestamp.today())
    
    # Preprocess data
    df = df[['Close']]
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(len(df) - n_days):
        X.append(df[i:i+n_days])
        y.append(df[i+n_days])
    X, y = np.array(X), np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model if specified
    if save_model:
        model.save('bitcoin_forecaster_model.h5')
    
    # Forecast next n days
    last_n_days = df[-n_days:]
    predictions = []
    for _ in range(n_days):
        pred = model.predict(last_n_days.reshape(1, n_days, 1))
        predictions.append(pred[0][0])
        last_n_days = np.append(last_n_days[1:], pred)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions

def load_and_predict(predict_days=5, n_days=5):
    # Load the saved model
    model = tf.keras.models.load_model('bitcoin_forecaster_model.h5')
    
    # Fetch the most recent data
    df = yf.download('BTC-USD', start='2020-01-01', end=pd.Timestamp.today())
    df = df[['Close']]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Get the last n_days dates
    last_dates = df.index[-n_days:]
    
    # Forecast next predict_days
    last_n_days = df_scaled[-n_days:]
    predictions = []
    for _ in range(predict_days):
        pred = model.predict(last_n_days.reshape(1, n_days, 1))
        predictions.append(pred[0][0])
        last_n_days = np.append(last_n_days[1:], pred)
    
    # Generate dates for the predictions
    next_dates = pd.date_range(last_dates[-1] + pd.Timedelta(days=1), periods=predict_days)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Return predictions with dates as the index
    return pd.Series(predictions.flatten(), index=next_dates).to_string()

# Run the training function
#forecasted_prices = train_bitcoin_forecaster(n_days=5)
#print(forecasted_prices)

# To load the model and predict, you can use:
loaded_predictions = load_and_predict(predict_days=3,n_days=5)
print(loaded_predictions)
print(type(loaded_predictions)) 