# Stock Price Forecasting Project


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------
# 1. Load stock data
# -------------------------
# Example: Apple stock (AAPL). You can change ticker symbol.
data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
print(data.head())

# -------------------------
# 2. Preprocess data
# -------------------------
# We use only 'Close' price for forecasting
close_prices = data['Close'].values.reshape(-1, 1)

# Scale values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# -------------------------
# 3. Create sequences
# -------------------------
def create_sequences(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)

# Reshape input for LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# -------------------------
# 4. Train/Test Split
# -------------------------
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------
# 5. Build LSTM model
# -------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# -------------------------
# 6. Train model
# -------------------------
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# -------------------------
# 7. Predictions
# -------------------------
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse scaling to get real prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------------
# 8. Plot results
# -------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Prices')
plt.plot(test_predict, label='Predicted Prices')
plt.title("Stock Price Forecasting (AAPL)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
