# Stock Price Trend Prediction with LSTM

## Introduction

This project presents a stock price trend prediction system using an LSTM (Long Short-Term Memory) neural network model. The objective is to forecast future stock prices based on historical data while incorporating technical indicators such as Moving Averages and RSI (Relative Strength Index). The entire workflow is visualized through an interactive dashboard built using Streamlit.

## Objectives

* Fetch historical stock data using the Yahoo Finance API.
* Compute technical indicators such as 20-day Moving Average and 14-day RSI.
* Normalize and prepare time-series data for deep learning.
* Train an LSTM model to predict stock closing prices.
* Visualize predicted vs actual prices using Matplotlib within a Streamlit dashboard.
* Display additional stock insights using RSI and MA plots.
* Deploy the entire system via a local Streamlit web app.

## Technologies Used

* **Python**
* **Pandas** for data handling and transformation
* **Matplotlib** for plotting charts
* **Keras & TensorFlow** for building the LSTM model
* **yfinance** for fetching stock data
* **ta** library for technical indicators (RSI)
* **Streamlit** for dashboard and interactivity
* **scikit-learn** for scaling and metrics

## Project Structure

* `dashboard.py`: Main dashboard script to run the Streamlit web app.
* `lstm_model.h5`: Trained LSTM model saved in HDF5 format.
* `README.md`: Project documentation and instructions.

## How It Works

1. The user inputs a stock ticker (e.g., AAPL) via the sidebar.
2. The system downloads historical data (e.g., from 2015 to 2023).
3. It computes 20-day moving averages and 14-day RSI values for the stock.
4. The closing price data is normalized using MinMaxScaler and prepared for LSTM training.
5. The LSTM model is trained on a portion of the data and saved as `lstm_model.h5`.
6. Upon loading, the model generates predictions on the test dataset.
7. The dashboard displays:

   * Actual vs Predicted stock prices.
   * Moving Average overlay.
   * RSI plot.

## How to Run the Project

### Step 1: Create Virtual Environment

```bash
python -m venv tf-env
```

### Step 2: Activate Environment

```bash
# Windows
.\tf-env\Scripts\activate

# macOS/Linux
source tf-env/bin/activate
```

### Step 3: Install Required Libraries

```bash
pip install streamlit yfinance pandas matplotlib scikit-learn ta keras tensorflow
```

### Step 4: Ensure Required Files Are in Place

* Confirm that `lstm_model.h5` (your trained model) is in the same directory as `dashboard.py`.

### Step 5: Launch the Dashboard

```bash
streamlit run dashboard.py
```

## Future Scope

* Add more technical indicators (MACD, Bollinger Bands).
* Include sentiment analysis using news or tweets.
* Deploy using Streamlit Cloud or Docker.
* Add user authentication for saved portfolios.

## Acknowledgments

* OpenAI for technical guidance
* Yahoo Finance for historical stock data
* TensorFlow/Keras for deep learning tools
* Streamlit for building lightweight dashboards
