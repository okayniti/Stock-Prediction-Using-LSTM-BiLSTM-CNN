"""
Stock Price Prediction using Deep Learning
============================================
Models: LSTM, CNN (1D), Bi-LSTM
Dataset: Apple (AAPL) historical stock data (2015-2025)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
    Flatten, Bidirectional, Input
)
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# ======================== Configuration =========================
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
WINDOW_SIZE = 60          # lookback window (days)
TRAIN_SPLIT = 0.8
EPOCHS = 50
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)


# =================== 1. Data Collection =========================
def download_data():
    """Download historical stock data and save to CSV."""
    csv_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
    if os.path.exists(csv_path):
        print(f"[INFO] Loading cached data from {csv_path}")
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    else:
        print(f"[INFO] Downloading {TICKER} data from Yahoo Finance ...")
        df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(csv_path)
        print(f"[INFO] Saved to {csv_path}  ({len(df)} rows)")
    return df


# =================== 2. Preprocessing ==========================
def preprocess(df):
    """Scale the Close price and create windowed sequences."""
    close_prices = df[["Close"]].values.astype(float)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled)):
        X.append(scaled[i - WINDOW_SIZE : i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for models:  (samples, timesteps, features)
    X_train = X_train.reshape(-1, WINDOW_SIZE, 1)
    X_test  = X_test.reshape(-1, WINDOW_SIZE, 1)

    print(f"[INFO] Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler


# =================== 3. Model Builders =========================
def build_lstm(input_shape):
    """Stacked LSTM model."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_cnn(input_shape):
    """1-D Convolutional Neural Network model."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_bilstm(input_shape):
    """Bidirectional LSTM model."""
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# =================== 4. Training ===============================
def train_model(model, name, X_train, y_train, X_test, y_test):
    """Train a model and return its history."""
    print(f"\n{'='*60}")
    print(f"  Training {name}")
    print(f"{'='*60}")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # Save model weights
    model_path = os.path.join(MODEL_DIR, f"{name}.keras")
    model.save(model_path)
    print(f"[INFO] Model saved -> {model_path}")

    return history


# =================== 5. Evaluation =============================
def evaluate_model(model, name, X_test, y_test, scaler):
    """Predict, inverse-transform, and compute metrics."""
    predictions = model.predict(X_test, verbose=0)

    # Inverse transform to original price scale
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    mae  = mean_absolute_error(y_test_inv, predictions_inv)
    r2   = r2_score(y_test_inv, predictions_inv)

    print(f"\n[{name}]  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  R2: {r2:.4f}")
    return predictions_inv, y_test_inv, {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


# =================== 6. Visualization ==========================
def plot_predictions(y_true, predictions_dict, title="Actual vs Predicted"):
    """Overlay predictions from all models on one chart."""
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label="Actual Price", color="#2c3e50", linewidth=2)

    colors = {"LSTM": "#e74c3c", "CNN": "#3498db", "Bi-LSTM": "#2ecc71"}
    for name, preds in predictions_dict.items():
        plt.plot(preds, label=f"{name} Prediction", color=colors.get(name, "#999"),
                 linewidth=1.3, alpha=0.85)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price (USD)")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "combined_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved -> {path}")


def plot_individual(y_true, preds, name, color):
    """Individual prediction plot for a single model."""
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="Actual", color="#2c3e50", linewidth=2)
    plt.plot(preds, label=f"{name} Prediction", color=color, linewidth=1.5, alpha=0.85)
    plt.title(f"{name} - Actual vs Predicted", fontsize=14, fontweight="bold")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price (USD)")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f"{name.lower().replace('-', '_')}_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_loss_curves(histories):
    """Training & validation loss curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"LSTM": "#e74c3c", "CNN": "#3498db", "Bi-LSTM": "#2ecc71"}

    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history.history["loss"], label="Train Loss", color=colors[name], linewidth=2)
        ax.plot(history.history["val_loss"], label="Val Loss", color=colors[name],
                linewidth=2, linestyle="--")
        ax.set_title(f"{name} Loss", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved -> {path}")


def plot_comparison_bar(metrics_list):
    """Bar chart comparing models on RMSE, MAE, and R2."""
    df = pd.DataFrame(metrics_list)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = ["#e74c3c", "#3498db", "#2ecc71"]

    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        bars = ax.bar(df["Model"], df[metric], color=palette, edgecolor="white", width=0.5)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.4f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Model Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved -> {path}")


# =================== 7. Main Pipeline ==========================
def main():
    print("=" * 60)
    print("  Stock Price Prediction - LSTM / CNN / Bi-LSTM")
    print("=" * 60)

    # Step 1: Data
    df = download_data()
    print(f"[INFO] Data shape: {df.shape}")
    print(df.tail())

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Step 3: Build models
    models = {
        "LSTM":    build_lstm(input_shape),
        "CNN":     build_cnn(input_shape),
        "Bi-LSTM": build_bilstm(input_shape),
    }

    # Print model summaries
    for name, model in models.items():
        print(f"\n{'-'*40} {name} {'-'*40}")
        model.summary()

    # Step 4: Train
    histories = {}
    for name, model in models.items():
        histories[name] = train_model(model, name, X_train, y_train, X_test, y_test)

    # Step 5: Evaluate
    predictions_dict = {}
    metrics_list = []
    model_colors = {"LSTM": "#e74c3c", "CNN": "#3498db", "Bi-LSTM": "#2ecc71"}

    y_true = None
    for name, model in models.items():
        preds, y_test_inv, metrics = evaluate_model(model, name, X_test, y_test, scaler)
        predictions_dict[name] = preds
        metrics_list.append(metrics)
        y_true = y_test_inv
        plot_individual(y_true, preds, name, model_colors[name])

    # Step 6: Visualize
    plot_predictions(y_true, predictions_dict,
                     title=f"{TICKER} Stock Price Prediction - Model Comparison")
    plot_loss_curves(histories)
    plot_comparison_bar(metrics_list)

    # Step 7: Summary table
    results_df = pd.DataFrame(metrics_list)
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULT_DIR, "comparison.csv"), index=False)
    print(f"\n[INFO] Results saved to {RESULT_DIR}")
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
