import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator

st.set_page_config(layout="wide")
st.title("Stock Price Trend Prediction Dashboard")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")

# Load Data
df = yf.download(ticker, start="2015-01-01", end="2023-12-31")
df.dropna(inplace=True)

# Add RSI and Moving Average
rsi = RSIIndicator(close=df['Close'], window=14)
df['RSI'] = rsi.rsi()
df['MA_20'] = df['Close'].rolling(window=20).mean()

# Normalize Close prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Prepare test data (last 100 days)
sequence_length = 100
x_test = []
for i in range(sequence_length, len(scaled_data)):
    x_test.append(scaled_data[i - sequence_length:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Load model (make sure this file exists)
model = load_model("lstm_model.h5")

# Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Create prediction dataframe
prediction_dates = df.index[sequence_length:]
df_pred = pd.DataFrame({
    'Date': prediction_dates,
    'Actual': df['Close'][sequence_length:].values,
    'Predicted': predicted_prices.flatten(),
    'RSI': df['RSI'][sequence_length:].values,
    'MA_20': df['MA_20'][sequence_length:].values
})
df_pred.set_index('Date', inplace=True)

# Plot
st.subheader(f"Predicted vs Actual Closing Prices for {ticker}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_pred['Actual'], label='Actual Price', color='blue')
ax.plot(df_pred['Predicted'], label='Predicted Price', color='orange')
ax.plot(df_pred['MA_20'], label='20-Day Moving Average', color='green')
ax.set_title(f'{ticker} Price Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# RSI Plot
st.subheader("Relative Strength Index (RSI)")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df_pred['RSI'], label='RSI', color='purple')
ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
ax2.set_title(f'{ticker} RSI Chart')
ax2.set_xlabel('Date')
ax2.set_ylabel('RSI')
ax2.legend()
st.pyplot(fig2)
