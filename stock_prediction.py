import sys
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

# Get stock price from argument
stock_price = float(sys.argv[1])

# Fetch last 10 minutes stock data for prediction
def get_live_stock_data(stock_symbol="RELIANCE.NS"):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1d", interval="1m")
    return data[['Close']]

# Preprocess the data
def preprocess_data(data, time_steps=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i, 0])
        y.append(data_scaled[i, 0])
    
    return np.array(X), np.array(y), scaler

# Build LSTM Model
def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 1)),
        LSTM(50),
        LSTM(50),
        Dense(1, activation="linear")
    ])
    model.compile(loss="mse", optimizer="adam")
    return model

# Main Function
def run_prediction():
    data = get_live_stock_data()
    X, y, scaler = preprocess_data(data.values)

    # Reshape for LSTM Model
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_lstm_model()
    model.fit(X, y, epochs=10, batch_size=1, verbose=2)

    # Predict next stock price
    last_10_min_data = X[-1].reshape(1, 10, 1)
    predicted_price = model.predict(last_10_min_data)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

    # Return prediction
    return predicted_price[0][0]

# Run and print prediction
if _name_ == "_main_":
    predicted_price = run_prediction()
    print(predicted_price)  # This will be passed back to Node.js