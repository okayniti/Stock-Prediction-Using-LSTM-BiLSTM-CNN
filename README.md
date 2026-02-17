# 📈 Stock Price Prediction using Deep Learning

A comprehensive deep learning project that predicts **Apple (AAPL)** stock prices using three neural network architectures — **LSTM**, **CNN (1D)**, and **Bi-LSTM** — and compares their performance through an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Objective

To forecast future stock closing prices by training deep learning models on historical time-series data and evaluate which architecture best captures temporal patterns in financial markets.

---

## 📂 Project Structure

```
stock-prediction-deep-learning/
├── data/                        # Auto-downloaded dataset (AAPL.csv)
│   └── AAPL.csv
├── models/                      # Saved trained model weights
│   ├── LSTM.keras
│   ├── CNN.keras
│   └── Bi-LSTM.keras
├── results/                     # Generated plots & metrics
│   ├── lstm_predictions.png
│   ├── cnn_predictions.png
│   ├── bi_lstm_predictions.png
│   ├── combined_predictions.png
│   ├── loss_curves.png
│   ├── model_comparison.png
│   └── comparison.csv
├── stock_prediction.py          # Main training & evaluation pipeline
├── dashboard.py                 # Interactive dashboard generator
├── dashboard.html               # Self-contained interactive dashboard
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🧠 Models & Architecture

### 1. LSTM (Long Short-Term Memory)

LSTMs are designed to learn long-term dependencies in sequential data using gating mechanisms (forget, input, output gates).

| Layer | Type | Units | Details |
|-------|------|-------|---------|
| 1 | LSTM | 128 | `return_sequences=True` |
| 2 | Dropout | — | 20% dropout |
| 3 | LSTM | 64 | `return_sequences=False` |
| 4 | Dropout | — | 20% dropout |
| 5 | Dense | 32 | ReLU activation |
| 6 | Dense | 1 | Linear output |

### 2. CNN (1D Convolutional Neural Network)

1D CNNs extract local patterns from time-series data using sliding convolutional filters, making them efficient for feature extraction.

| Layer | Type | Filters/Units | Details |
|-------|------|---------------|---------|
| 1 | Conv1D | 64 | Kernel size: 3, ReLU |
| 2 | MaxPooling1D | — | Pool size: 2 |
| 3 | Conv1D | 128 | Kernel size: 3, ReLU |
| 4 | MaxPooling1D | — | Pool size: 2 |
| 5 | Flatten | — | — |
| 6 | Dense | 64 | ReLU activation |
| 7 | Dropout | — | 20% dropout |
| 8 | Dense | 1 | Linear output |

### 3. Bi-LSTM (Bidirectional LSTM)

Bi-LSTMs process sequences in both forward and backward directions, capturing past and future context simultaneously.

| Layer | Type | Units | Details |
|-------|------|-------|---------|
| 1 | Bidirectional LSTM | 128 x 2 | `return_sequences=True` |
| 2 | Dropout | — | 20% dropout |
| 3 | Bidirectional LSTM | 64 x 2 | `return_sequences=False` |
| 4 | Dropout | — | 20% dropout |
| 5 | Dense | 32 | ReLU activation |
| 6 | Dense | 1 | Linear output |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | Yahoo Finance (via `yfinance`) |
| **Ticker** | AAPL (Apple Inc.) |
| **Period** | January 2015 — January 2025 |
| **Records** | ~2,516 trading days |
| **Target Variable** | Closing Price |
| **Normalization** | MinMaxScaler (0–1) |
| **Lookback Window** | 60 days |
| **Train/Test Split** | 80% / 20% |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 50 (max) |
| Batch Size | 32 |
| Early Stopping | Patience = 5, restore best weights |
| Random Seed | 42 (reproducible results) |

---

## 📈 Results & Comparison

| Model | RMSE ↓ | MAE ↓ | R² Score ↑ | Parameters |
|-------|--------|-------|------------|------------|
| **Bi-LSTM** | **5.3722** | **4.2560** | **0.9604** | ~300K |
| LSTM | 5.4197 | 4.2647 | 0.9597 | ~150K |
| CNN | 7.8529 | 6.4187 | 0.9154 | ~120K |

### Key Findings

- **Bi-LSTM achieves the highest R² (0.9604)** — processing sequences bidirectionally captures richer temporal patterns
- **LSTM closely follows** with nearly identical performance at half the parameter count
- **CNN underperforms** on this task — CNNs are better at local feature extraction but miss long-range temporal dependencies that recurrent models capture
- All models achieve R² > 0.91, confirming deep learning's effectiveness for stock price trend forecasting

---

## 🖥️ Interactive Dashboard

The project includes a **self-contained interactive HTML dashboard** powered by Plotly.js with:

- ✅ **Model Metric Cards** — RMSE, MAE, R² for each model with "Best Model" badge
- ✅ **Full Price History** — 10-year AAPL chart with train/test split marker
- ✅ **Combined Predictions** — All models overlaid with toggle buttons
- ✅ **Individual Model View** — Single model with error overlay
- ✅ **Comparison Table** — Side-by-side metrics with best-value highlighting
- ✅ **Metric Bar Charts** — Grouped bar comparison
- ✅ **Error Distribution** — Overlaid histograms of prediction errors
- ✅ **Cumulative Error** — Running error accumulation over time

> Open `dashboard.html` in any browser — no server needed.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation & Running

```bash
# Clone the repository
git clone https://github.com/okayniti/Stock-Prediction-Using-LSTM-BiLSTM-CNN.git
cd Stock-Prediction-Using-LSTM-BiLSTM-CNN

# Install dependencies
pip install -r requirements.txt

# Train all models (downloads data automatically)
python stock_prediction.py

# Generate interactive dashboard
python dashboard.py

# Open dashboard.html in your browser
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.13 |
| **Deep Learning** | TensorFlow 2.20, Keras 3.x |
| **Data** | yfinance, Pandas, NumPy |
| **Preprocessing** | scikit-learn (MinMaxScaler) |
| **Evaluation** | scikit-learn (RMSE, MAE, R²) |
| **Visualization** | Matplotlib, Seaborn, Plotly.js |
| **Dashboard** | Self-contained HTML + Plotly.js |

---

## 🔮 Future Improvements

- [ ] Add **GRU** and **Transformer** architectures for comparison
- [ ] Incorporate **multi-feature input** (Open, High, Low, Volume)
- [ ] Add **attention mechanisms** to LSTM/Bi-LSTM
- [ ] Implement **hyperparameter tuning** with Optuna
- [ ] Add **multi-step forecasting** (predict next N days)
- [ ] Deploy as a **Streamlit web app**

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built with ❤️ using TensorFlow & Keras</b>
</p>
