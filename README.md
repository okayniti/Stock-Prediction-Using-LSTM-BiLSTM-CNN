# 📈 Stock Price Prediction Using Deep Learning

**Forecasting Apple (AAPL) stock prices with LSTM, Bi-LSTM, and CNN — achieving 96% prediction accuracy (R² = 0.96) on 10 years of real-world financial data.**

Built a complete deep learning pipeline: data acquisition → preprocessing → model training → evaluation → interactive dashboard. Three architectures are trained under identical conditions and rigorously compared to reveal which best captures temporal patterns in stock markets.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎬 Demo

![Dashboard Demo](Stock%20Prediction%20Dashboard%20-%20AAPL%20-%20Google%20Chrome%202026-02-17%2019-29-13.gif)

---

## 🧠 Models

### LSTM (Long Short-Term Memory)
Uses gating mechanisms to selectively retain information across long time horizons — a natural fit for sequential financial data.

> 2× Stacked LSTM (128 → 64) → Dropout (20%) → Dense (32) → Output

### Bi-LSTM (Bidirectional LSTM)
Processes sequences in both forward and reverse directions, capturing dependencies that unidirectional models miss.

> 2× Bidirectional LSTM (128 → 64) → Dropout (20%) → Dense (32) → Output

### CNN (1D Convolutional)
Applies sliding convolutional filters to extract local temporal patterns from price windows.

> Conv1D (64) → MaxPool → Conv1D (128) → MaxPool → Flatten → Dense (64) → Output

---

## 📊 Results

| Model | RMSE ↓ | MAE ↓ | R² Score ↑ | Params |
|-------|--------|-------|------------|--------|
| 🥇 **Bi-LSTM** | **5.37** | **4.26** | **0.9604** | ~300K |
| 🥈 LSTM | 5.42 | 4.26 | 0.9597 | ~150K |
| 🥉 CNN | 7.85 | 6.42 | 0.9154 | ~120K |

**Key Insight:** Bi-LSTM edges out LSTM by processing sequences bidirectionally. CNN lacks the long-range memory that recurrent networks provide — critical when today's prices depend on trends from weeks ago. LSTM offers the best accuracy-to-cost ratio at half the parameters.

---

## 🖥️ Interactive Dashboard

Self-contained HTML dashboard with **8 interactive Plotly panels:**

- Model metric cards with best-model badge
- Full 10-year price history with train/test split
- Combined predictions with toggle controls
- Individual model view with error overlay
- Comparison table, metric bars, error distribution, cumulative error

> Just open `dashboard.html` in any browser — no server needed.

---

## 🚀 Getting Started

```bash
git clone https://github.com/okayniti/Stock-Prediction-Using-LSTM-BiLSTM-CNN.git
cd Stock-Prediction-Using-LSTM-BiLSTM-CNN

pip install -r requirements.txt

# Train all models (~5 min)
python stock_prediction.py

# Generate dashboard
python dashboard.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow, Keras |
| Data | yfinance, Pandas, NumPy |
| Preprocessing | scikit-learn (MinMaxScaler) |
| Evaluation | RMSE, MAE, R² Score |
| Visualization | Matplotlib, Seaborn, Plotly.js |

---

## 💡 Skills Demonstrated

- End-to-end ML pipeline ownership (data → model → evaluation → visualization)
- Time-series forecasting with 3 deep learning architectures
- Comparative model analysis with controlled experiments
- Interactive dashboard development
- Clean, modular, production-style Python code

---

## 🔮 Future Scope

- GRU and Transformer architectures
- Multi-feature input (Open, High, Low, Volume)
- Attention mechanisms on recurrent layers
- Hyperparameter tuning with Optuna
- Multi-step forecasting (next N days)
- Live deployment with Streamlit

---

## 👤 About the Developer

**Niti** — CS undergraduate focused on applied AI and deep learning. Building end-to-end intelligent systems that go beyond tutorials — from data pipelines to architecture decisions to interactive visualization. Passionate about the intersection of ML, product thinking, and real-world deployment.

→ [GitHub](https://github.com/okayniti)

---

<p align="center"><b>Built with ❤️ using TensorFlow & Keras</b></p>
