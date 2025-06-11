# 🪙 Bitcoin Price Prediction using LSTM

This project predicts future Bitcoin prices using an LSTM (Long Short-Term Memory) neural network. It includes additional features such as Volume, RSI (Relative Strength Index), and MACD (Moving Average Convergence Divergence) for better accuracy.

---

## 📌 Features

- Uses historical BTC-USD data from Yahoo Finance
- Technical indicators included:
  - Volume
  - RSI
  - MACD
- Deep LSTM model with multiple layers
- Dropout layers for regularization
- Live prediction for the next day's BTC price
- Easy to customize and tune hyperparameters

---

## 📁 Files

| File Name                         | Description                                 |
|----------------------------------|---------------------------------------------|
| `bitcoin_price_prediction_lstm.py` | Main Python script for model training and prediction |
| `README.md`                      | Project documentation                       |

---

## 📦 Requirements

Install the dependencies using pip:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn ta tensorflow
````

---

## ▶️ How to Run

1. Clone this repository or save the script locally.
2. Run the Python script:

```bash
python bitcoin_price_prediction_lstm.py
```

3. The script will:

   * Download Bitcoin data
   * Train an LSTM model
   * Plot the prediction results
   * Print the predicted next-day price

---

## 📊 Sample Output

* A graph comparing actual vs predicted BTC prices
* Printed predicted price for the next day

```text
📅 Predicted Bitcoin Price for Next Day: $68492.31
```

---

## 🔧 Customize

* Change time window: `time_step = 60`
* Tune model:

  * LSTM units
  * Epochs
  * Batch size
  * Optimizer (e.g., 'adam', 'rmsprop')

---

## 🚀 Future Improvements

* Add more indicators: Bollinger Bands, EMA
* Deploy as a web app using Streamlit
* Hyperparameter tuning with KerasTuner
* Add real-time live price forecast updates
