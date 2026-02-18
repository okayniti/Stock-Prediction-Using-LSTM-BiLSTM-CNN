# Stock Price Prediction Using Deep Learning

**Forecasting Apple (AAPL) stock prices with LSTM, Bi-LSTM, and CNN architectures — achieving 96% prediction accuracy (R² = 0.96) on real-world financial time-series data.**

This project demonstrates a complete end-to-end deep learning pipeline: from raw market data acquisition and preprocessing through model training, evaluation, and interactive visualization. Three distinct neural network architectures are implemented, trained, and rigorously compared to determine which best captures temporal patterns in stock price movements — a challenge at the intersection of finance and artificial intelligence where even marginal accuracy improvements carry significant real-world value.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Demo

![Dashboard Demo](Stock%20Prediction%20Dashboard%20-%20AAPL%20-%20Google%20Chrome%202026-02-17%2019-29-13.gif)

> A fully interactive Plotly-powered dashboard comparing all three model predictions, error distributions, and evaluation metrics — generated entirely from code.

---

## Why This Project Matters

Stock price prediction is one of the most studied — and most difficult — problems in machine learning. Financial time-series data is inherently noisy, non-stationary, and influenced by factors that no model can fully capture. This makes it an excellent benchmark for evaluating how well deep learning architectures handle real-world temporal pattern recognition.

This project goes beyond a single model script. It implements **three fundamentally different architectures**, trains each under identical conditions, and provides both quantitative metrics and visual analysis to draw meaningful conclusions — the kind of rigorous comparison expected in production ML workflows.

---

## Architecture Overview

### LSTM (Long Short-Term Memory)

Designed specifically for sequential data, LSTMs use gating mechanisms to selectively retain or discard information across long time horizons — making them a natural fit for time-series forecasting.

| Layer | Type | Configuration |
|-------|------|---------------|
| 1 | LSTM | 128 units, return sequences |
| 2 | Dropout | 20% |
| 3 | LSTM | 64 units |
| 4 | Dropout | 20% |
| 5 | Dense | 32 units, ReLU |
| 6 | Dense | 1 unit, Linear output |

### Bi-LSTM (Bidirectional LSTM)

Extends LSTM by processing sequences in both forward and reverse directions simultaneously. This dual-pass approach captures dependencies that unidirectional models miss, particularly useful when recent context and historical trends both influence the target.

| Layer | Type | Configuration |
|-------|------|---------------|
| 1 | Bidirectional LSTM | 128 units x 2 directions, return sequences |
| 2 | Dropout | 20% |
| 3 | Bidirectional LSTM | 64 units x 2 directions |
| 4 | Dropout | 20% |
| 5 | Dense | 32 units, ReLU |
| 6 | Dense | 1 unit, Linear output |

### CNN (1D Convolutional Neural Network)

Applies sliding convolutional filters to extract local patterns from time-series windows. While CNNs excel at spatial feature extraction, this project tests whether that strength transfers to temporal data — and the results reveal an important architectural insight.

| Layer | Type | Configuration |
|-------|------|---------------|
| 1 | Conv1D | 64 filters, kernel size 3, ReLU |
| 2 | MaxPooling1D | Pool size 2 |
| 3 | Conv1D | 128 filters, kernel size 3, ReLU |
| 4 | MaxPooling1D | Pool size 2 |
| 5 | Flatten | — |
| 6 | Dense | 64 units, ReLU |
| 7 | Dropout | 20% |
| 8 | Dense | 1 unit, Linear output |

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Yahoo Finance via `yfinance` |
| Ticker | AAPL (Apple Inc.) |
| Time Period | January 2015 — January 2025 |
| Total Records | ~2,516 trading days |
| Target Variable | Daily closing price |
| Normalization | MinMaxScaler (0–1 range) |
| Lookback Window | 60 days |
| Train / Test Split | 80% / 20% (chronological, no data leakage) |

The dataset is downloaded automatically on first run and cached locally. The chronological split ensures no future data leaks into training — a common mistake in financial ML that this project deliberately avoids.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Max Epochs | 50 |
| Batch Size | 32 |
| Early Stopping | Patience = 5, restores best weights |
| Random Seed | 42 (reproducible results) |

All three models are trained under identical hyperparameters to ensure the comparison reflects architectural differences, not tuning advantages.

---

## Results

| Model | RMSE | MAE | R² Score | Parameters |
|-------|------|-----|----------|------------|
| **Bi-LSTM** | **5.37** | **4.26** | **0.9604** | ~300K |
| LSTM | 5.42 | 4.26 | 0.9597 | ~150K |
| CNN | 7.85 | 6.42 | 0.9154 | ~120K |

### Interpretation

**Bi-LSTM delivers the strongest performance**, marginally outperforming standard LSTM. The bidirectional architecture's ability to process sequences from both ends provides a measurable — though small — advantage, suggesting that reverse-context carries meaningful information in stock price patterns.

**LSTM performs nearly identically** at half the parameter count, making it the most parameter-efficient choice. For production environments where inference speed matters, LSTM offers the best accuracy-to-cost ratio.

**CNN underperforms both recurrent architectures** (R² = 0.92 vs 0.96). This confirms an important insight: while CNNs capture local temporal features effectively, they lack the long-range memory that recurrent networks provide — a critical capability when stock movements today depend on trends established weeks or months ago.

All three models achieve R² > 0.91, validating deep learning as a viable approach for stock price trend forecasting.

---

## Interactive Dashboard

The project generates a self-contained HTML dashboard (no server required) with eight interactive panels:

| Panel | Description |
|-------|-------------|
| Model Metric Cards | RMSE, MAE, R² per model with best-model indicator |
| Full Price History | 10-year AAPL chart with train/test boundary |
| Combined Predictions | All models overlaid with toggle controls |
| Individual Model View | Single-model predictions with error overlay |
| Comparison Table | Side-by-side metrics with best-value highlighting |
| Metric Bar Charts | Grouped comparison across RMSE, MAE, R² |
| Error Distribution | Overlaid histograms of prediction errors per model |
| Cumulative Error | Running absolute error accumulation over time |

Built with Plotly.js — fully interactive with hover tooltips, zoom, and pan.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Setup and Execution

```bash
# Clone the repository
git clone https://github.com/okayniti/Stock-Prediction-Using-LSTM-BiLSTM-CNN.git
cd Stock-Prediction-Using-LSTM-BiLSTM-CNN

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Train all models
python stock_prediction.py

# Generate interactive dashboard
python dashboard.py
```

Open `dashboard.html` in any browser to explore the results.

---

## Project Structure

```
├── data/                    # Auto-downloaded dataset
├── models/                  # Saved model weights (.keras)
├── results/                 # Generated plots and comparison CSV
├── stock_prediction.py      # Training and evaluation pipeline
├── dashboard.py             # Dashboard generator
├── dashboard.html           # Interactive dashboard (open in browser)
├── requirements.txt         # Dependencies
└── README.md
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| Deep Learning | TensorFlow 2.20, Keras 3.x |
| Data Acquisition | yfinance |
| Data Processing | Pandas, NumPy |
| Preprocessing | scikit-learn (MinMaxScaler) |
| Evaluation | scikit-learn (RMSE, MAE, R²) |
| Static Visualization | Matplotlib, Seaborn |
| Interactive Dashboard | Plotly.js |

---

## Skills Demonstrated

- End-to-end ML pipeline: data collection, preprocessing, training, evaluation, visualization
- Time-series forecasting with deep learning (LSTM, Bi-LSTM, CNN)
- Sliding window sequence generation for temporal modeling
- Comparative model analysis with consistent experimental controls
- Interactive data visualization and dashboard development
- Model persistence and reproducible training workflows
- Clean, modular Python code following production conventions

---

## Future Scope

- Add GRU and Transformer-based architectures for broader comparison
- Incorporate multi-feature input (Open, High, Low, Volume) for richer signal
- Implement attention mechanisms on recurrent layers
- Automate hyperparameter search using Optuna or Keras Tuner
- Extend to multi-step forecasting (predict next N days)
- Deploy as a live web application using Streamlit or FastAPI

---

## License

This project is open source under the [MIT License](LICENSE).

---

## About the Developer

**Niti** — Computer Science undergraduate focused on applied AI and deep learning.

This project reflects a deliberate effort to move beyond tutorials and build end-to-end intelligent systems that solve real problems. From data pipelines to model architecture decisions to interactive visualization, every component was built from scratch to understand not just *how* deep learning works, but *why* specific design choices lead to measurably better outcomes.

Currently exploring the intersection of machine learning, product thinking, and real-world deployment — with a focus on building things that work, not just things that compile.

[GitHub](https://github.com/okayniti)
