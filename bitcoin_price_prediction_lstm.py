import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Step 1: Download BTC data
df = yf.download("BTC-USD", start="2016-01-01", end="2024-12-31")
df = df[['Close', 'Volume']]

# Step 2: Add technical indicators
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd()
df.dropna(inplace=True)

# Step 3: Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Step 4: Dataset builder
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Step 5: Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 6: LSTM Model
model = Sequential()
model.add(LSTM(120, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(60))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Step 7: Train the model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=60, batch_size=64,
                    validation_data=(X_test, y_test), verbose=1)

# Step 8: Predict
predicted = model.predict(X_test)
predicted_full = scaler.inverse_transform(np.hstack([
    predicted, np.zeros((predicted.shape[0], X.shape[2] - 1))
]))[:, 0]
actual_full = scaler.inverse_transform(np.hstack([
    y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X.shape[2] - 1))
]))[:, 0]

# Step 9: Plot
plt.figure(figsize=(14, 6))
plt.plot(actual_full, label="Actual BTC Price")
plt.plot(predicted_full, label="Predicted BTC Price")
plt.title("Bitcoin Price Prediction with LSTM")
plt.xlabel("Time")
plt.ylabel("Price in USD")
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Predict next day's price
last_60_days = scaled_data[-60:]
last_60_days = last_60_days.reshape(1, 60, X.shape[2])
next_day_pred_scaled = model.predict(last_60_days)
next_day_price = scaler.inverse_transform(np.hstack([
    next_day_pred_scaled, np.zeros((1, X.shape[2] - 1))
]))[:, 0]

print(f"\n📅 Predicted Bitcoin Price for Next Day: ${next_day_price[0]:.2f}")
