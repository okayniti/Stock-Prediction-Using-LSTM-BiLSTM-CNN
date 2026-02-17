"""
Interactive Dashboard for Stock Prediction Models
===================================================
Generates a self-contained HTML dashboard with interactive
Plotly charts comparing LSTM, CNN, and Bi-LSTM performance.
"""

import os
import json
import base64
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# ======================== Configuration =========================
TICKER = "AAPL"
WINDOW_SIZE = 60
TRAIN_SPLIT = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")


def load_and_prepare_data():
    """Load cached CSV and prepare data."""
    csv_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

    close_prices = df[["Close"]].values.astype(float)
    dates = df.index

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled)):
        X.append(scaled[i - WINDOW_SIZE: i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * TRAIN_SPLIT)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = X_train.reshape(-1, WINDOW_SIZE, 1)
    X_test = X_test.reshape(-1, WINDOW_SIZE, 1)

    test_dates = dates[WINDOW_SIZE + split_idx:]
    train_dates = dates[WINDOW_SIZE: WINDOW_SIZE + split_idx]

    return df, X_train, X_test, y_train, y_test, scaler, train_dates, test_dates


def get_predictions_and_metrics(X_train, X_test, y_train, y_test, scaler):
    """Load all saved models, compute predictions and metrics."""
    model_names = ["LSTM", "CNN", "Bi-LSTM"]
    results = {}

    for name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{name}.keras")
        print(f"[INFO] Loading {name} from {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Predictions on test set
        test_preds = model.predict(X_test, verbose=0)
        test_preds_inv = scaler.inverse_transform(test_preds).flatten()

        # Predictions on train set
        train_preds = model.predict(X_train, verbose=0)
        train_preds_inv = scaler.inverse_transform(train_preds).flatten()

        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
        mae = mean_absolute_error(y_test_inv, test_preds_inv)
        r2 = r2_score(y_test_inv, test_preds_inv)

        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_preds_inv))
        train_mae = mean_absolute_error(y_train_inv, train_preds_inv)
        train_r2 = r2_score(y_train_inv, train_preds_inv)

        # Get model summary info
        total_params = model.count_params()
        num_layers = len(model.layers)

        results[name] = {
            "test_preds": test_preds_inv.tolist(),
            "train_preds": train_preds_inv.tolist(),
            "metrics": {
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
                "Train_RMSE": round(train_rmse, 4),
                "Train_MAE": round(train_mae, 4),
                "Train_R2": round(train_r2, 4),
            },
            "params": total_params,
            "layers": num_layers,
        }

        print(f"  [{name}]  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

    return results


def embed_image(path):
    """Read image file and return base64 encoded data URI."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def generate_dashboard(df, results, y_train, y_test, scaler, train_dates, test_dates):
    """Generate the full HTML dashboard."""

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

    # Prepare date strings
    test_dates_str = [d.strftime("%Y-%m-%d") for d in test_dates]
    train_dates_str = [d.strftime("%Y-%m-%d") for d in train_dates]
    all_dates_str = [d.strftime("%Y-%m-%d") for d in df.index]
    all_close = df["Close"].values.astype(float).tolist()

    # Embed training loss curve images
    loss_img = ""
    loss_path = os.path.join(RESULT_DIR, "loss_curves.png")
    if os.path.exists(loss_path):
        loss_img = embed_image(loss_path)

    # Build JSON data for JavaScript
    dashboard_data = {
        "ticker": TICKER,
        "allDates": all_dates_str,
        "allClose": all_close,
        "trainDates": train_dates_str,
        "testDates": test_dates_str,
        "yTrainActual": y_train_inv.tolist(),
        "yTestActual": y_test_inv.tolist(),
        "models": {}
    }

    for name, res in results.items():
        dashboard_data["models"][name] = {
            "testPreds": res["test_preds"],
            "trainPreds": res["train_preds"],
            "metrics": res["metrics"],
            "params": res["params"],
            "layers": res["layers"],
        }

    data_json = json.dumps(dashboard_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Prediction Dashboard - {TICKER}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a1f35;
    --bg-card-hover: #212842;
    --border: #2a3050;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-cyan: #06b6d4;
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    --shadow-glow: 0 0 30px rgba(59, 130, 246, 0.15);
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* Subtle animated gradient background */
  body::before {{
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background:
      radial-gradient(ellipse at 20% 0%, rgba(59,130,246,0.08) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 100%, rgba(139,92,246,0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }}

  .dashboard {{ position: relative; z-index: 1; padding: 24px; max-width: 1600px; margin: 0 auto; }}

  /* ── Header ─────────────────── */
  .header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
    padding: 28px 36px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: var(--shadow);
  }}
  .header-left {{ display: flex; align-items: center; gap: 18px; }}
  .header-icon {{
    width: 52px; height: 52px;
    background: var(--gradient-1);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 20px rgba(102,126,234,0.3);
  }}
  .header h1 {{ font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }}
  .header p {{ color: var(--text-secondary); font-size: 14px; margin-top: 2px; }}
  .header-badge {{
    padding: 8px 18px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 30px;
    color: var(--accent-green);
    font-size: 13px;
    font-weight: 600;
  }}

  /* ── Metric Cards Row ─────────── */
  .metrics-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 28px;
  }}
  .model-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }}
  .model-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
  }}
  .model-card:nth-child(1)::before {{ background: var(--accent-red); }}
  .model-card:nth-child(2)::before {{ background: var(--accent-blue); }}
  .model-card:nth-child(3)::before {{ background: var(--accent-green); }}
  .model-card:hover {{
    border-color: var(--accent-blue);
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow);
  }}
  .model-card.best {{
    border-color: var(--accent-green);
    box-shadow: 0 0 25px rgba(16,185,129,0.12);
  }}
  .model-card .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }}
  .model-card .model-name {{ font-size: 20px; font-weight: 700; }}
  .model-card .best-badge {{
    font-size: 11px;
    background: rgba(16,185,129,0.15);
    color: var(--accent-green);
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  .metric-item {{ text-align: center; }}
  .metric-item .label {{
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
  }}
  .metric-item .value {{
    font-size: 22px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .metric-item .sub {{
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
  }}
  .val-rmse {{ color: var(--accent-amber); }}
  .val-mae {{ color: var(--accent-cyan); }}
  .val-r2 {{ color: var(--accent-green); }}

  .model-meta {{
    display: flex; gap: 16px; margin-top: 14px; padding-top: 14px;
    border-top: 1px solid var(--border);
  }}
  .model-meta span {{
    font-size: 12px; color: var(--text-muted);
  }}
  .model-meta span b {{ color: var(--text-secondary); }}

  /* ── Charts ────────────────── */
  .chart-section {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
  }}
  .chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }}
  .chart-title {{
    font-size: 18px;
    font-weight: 600;
  }}
  .chart-controls {{
    display: flex; gap: 8px;
  }}
  .chart-btn {{
    padding: 7px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
    color: var(--text-secondary);
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .chart-btn:hover {{ border-color: var(--accent-blue); color: var(--text-primary); }}
  .chart-btn.active {{
    background: var(--accent-blue);
    border-color: var(--accent-blue);
    color: white;
  }}

  /* ── Two Column Grid ─────────── */
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }}

  /* ── Comparison Table ────────── */
  .comp-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
  }}
  .comp-table th {{
    text-align: left;
    padding: 14px 18px;
    background: rgba(59,130,246,0.08);
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid var(--border);
  }}
  .comp-table th:first-child {{ border-radius: 10px 0 0 0; }}
  .comp-table th:last-child {{ border-radius: 0 10px 0 0; }}
  .comp-table td {{
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
    font-variant-numeric: tabular-nums;
  }}
  .comp-table tr:last-child td {{ border-bottom: none; }}
  .comp-table tr:hover td {{ background: rgba(59,130,246,0.04); }}
  .comp-table .model-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 10px;
  }}
  .highlight-best {{
    color: var(--accent-green);
    font-weight: 700;
  }}

  /* ── Loss curves image ────────── */
  .loss-img {{
    width: 100%;
    border-radius: 12px;
    opacity: 0.95;
  }}

  /* ── Footer ────────────────── */
  .footer {{
    text-align: center;
    padding: 20px;
    color: var(--text-muted);
    font-size: 12px;
  }}

  /* ── Responsive ────────────── */
  @media (max-width: 900px) {{
    .metrics-row {{ grid-template-columns: 1fr; }}
    .two-col {{ grid-template-columns: 1fr; }}
    .header {{ flex-direction: column; gap: 14px; text-align: center; }}
  }}
</style>
</head>
<body>
<div class="dashboard">

  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <div class="header-icon">$</div>
      <div>
        <h1>{TICKER} Stock Prediction Dashboard</h1>
        <p>Deep Learning Model Comparison &mdash; LSTM, CNN, Bi-LSTM</p>
      </div>
    </div>
    <div class="header-badge">All Models Trained</div>
  </div>

  <!-- Model Metric Cards -->
  <div class="metrics-row" id="metricCards"></div>

  <!-- Full Price History -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">Full Price History &amp; Train/Test Split</span>
    </div>
    <div id="chartFullHistory" style="height:380px;"></div>
  </div>

  <!-- Combined Predictions -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">Test Set Predictions &mdash; All Models</span>
      <div class="chart-controls">
        <button class="chart-btn active" onclick="toggleModel(this,'LSTM')">LSTM</button>
        <button class="chart-btn active" onclick="toggleModel(this,'CNN')">CNN</button>
        <button class="chart-btn active" onclick="toggleModel(this,'Bi-LSTM')">Bi-LSTM</button>
      </div>
    </div>
    <div id="chartCombined" style="height:420px;"></div>
  </div>

  <!-- Individual Model Charts -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">Individual Model Predictions</span>
      <div class="chart-controls">
        <button class="chart-btn active" onclick="showIndividual('LSTM', this)">LSTM</button>
        <button class="chart-btn" onclick="showIndividual('CNN', this)">CNN</button>
        <button class="chart-btn" onclick="showIndividual('Bi-LSTM', this)">Bi-LSTM</button>
      </div>
    </div>
    <div id="chartIndividual" style="height:400px;"></div>
  </div>

  <!-- Two-column: Comparison Table + Loss Curves -->
  <div class="two-col">
    <div class="chart-section" style="margin-bottom:0;">
      <div class="chart-header">
        <span class="chart-title">Model Comparison</span>
      </div>
      <table class="comp-table" id="compTable"></table>
    </div>
    <div class="chart-section" style="margin-bottom:0;">
      <div class="chart-header">
        <span class="chart-title">Metric Comparison</span>
      </div>
      <div id="chartMetricBars" style="height:350px;"></div>
    </div>
  </div>

  <!-- Prediction Error Distribution -->
  <div class="two-col">
    <div class="chart-section" style="margin-bottom:0;">
      <div class="chart-header">
        <span class="chart-title">Prediction Error Distribution</span>
      </div>
      <div id="chartErrorDist" style="height:350px;"></div>
    </div>
    <div class="chart-section" style="margin-bottom:0;">
      <div class="chart-header">
        <span class="chart-title">Cumulative Absolute Error</span>
      </div>
      <div id="chartCumError" style="height:350px;"></div>
    </div>
  </div>

  <!-- Loss Curves -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">Training Loss Curves</span>
    </div>
    {"<img class='loss-img' src='" + loss_img + "' alt='Loss Curves'>" if loss_img else "<p style='color:var(--text-muted)'>Loss curve image not found.</p>"}
  </div>

  <div class="footer">
    Stock Prediction Dashboard &bull; Built with TensorFlow &amp; Plotly &bull; Data: Yahoo Finance ({TICKER})
  </div>

</div>

<script>
const D = {data_json};

const COLORS = {{
  "LSTM":    "#ef4444",
  "CNN":     "#3b82f6",
  "Bi-LSTM": "#10b981"
}};

const plotLayout = {{
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: {{ family: 'Inter, sans-serif', color: '#94a3b8', size: 12 }},
  margin: {{ t: 20, b: 50, l: 60, r: 20 }},
  xaxis: {{
    gridcolor: 'rgba(42,48,80,0.5)',
    linecolor: '#2a3050',
    zerolinecolor: '#2a3050',
  }},
  yaxis: {{
    gridcolor: 'rgba(42,48,80,0.5)',
    linecolor: '#2a3050',
    zerolinecolor: '#2a3050',
    title: 'Price (USD)'
  }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 12 }} }},
  hoverlabel: {{ bgcolor: '#1a1f35', bordercolor: '#3b82f6', font: {{ family: 'Inter', size: 13 }} }}
}};

const plotConfig = {{ responsive: true, displayModeBar: false }};

// ── Metric Cards ──
function renderMetricCards() {{
  const container = document.getElementById('metricCards');
  const names = Object.keys(D.models);

  // Find best model by R2
  let bestName = names[0];
  names.forEach(n => {{
    if (D.models[n].metrics.R2 > D.models[bestName].metrics.R2) bestName = n;
  }});

  names.forEach(name => {{
    const m = D.models[name];
    const isBest = name === bestName;
    const card = document.createElement('div');
    card.className = 'model-card' + (isBest ? ' best' : '');
    card.innerHTML = `
      <div class="card-header">
        <span class="model-name" style="color:${{COLORS[name]}}">${{name}}</span>
        ${{isBest ? '<span class="best-badge">Best Model</span>' : ''}}
      </div>
      <div class="metric-grid">
        <div class="metric-item">
          <div class="label">RMSE</div>
          <div class="value val-rmse">${{m.metrics.RMSE.toFixed(4)}}</div>
          <div class="sub">Train: ${{m.metrics.Train_RMSE.toFixed(4)}}</div>
        </div>
        <div class="metric-item">
          <div class="label">MAE</div>
          <div class="value val-mae">${{m.metrics.MAE.toFixed(4)}}</div>
          <div class="sub">Train: ${{m.metrics.Train_MAE.toFixed(4)}}</div>
        </div>
        <div class="metric-item">
          <div class="label">R2</div>
          <div class="value val-r2">${{m.metrics.R2.toFixed(4)}}</div>
          <div class="sub">Train: ${{m.metrics.Train_R2.toFixed(4)}}</div>
        </div>
      </div>
      <div class="model-meta">
        <span>Layers: <b>${{m.layers}}</b></span>
        <span>Parameters: <b>${{m.params.toLocaleString()}}</b></span>
      </div>
    `;
    container.appendChild(card);
  }});
}}

// ── Full History Chart ──
function renderFullHistory() {{
  const trainEnd = D.trainDates[D.trainDates.length - 1];
  const traces = [
    {{
      x: D.allDates, y: D.allClose,
      type: 'scatter', mode: 'lines',
      name: 'Close Price',
      line: {{ color: '#8b5cf6', width: 1.5 }}
    }}
  ];
  const layout = {{
    ...plotLayout,
    shapes: [{{
      type: 'line', x0: trainEnd, x1: trainEnd,
      y0: 0, y1: 1, yref: 'paper',
      line: {{ color: '#f59e0b', width: 2, dash: 'dash' }}
    }}],
    annotations: [{{
      x: trainEnd, y: 1.02, yref: 'paper',
      text: 'Train / Test Split', showarrow: false,
      font: {{ color: '#f59e0b', size: 12 }}
    }}]
  }};
  Plotly.newPlot('chartFullHistory', traces, layout, plotConfig);
}}

// ── Combined Predictions ──
let visibleModels = {{ 'LSTM': true, 'CNN': true, 'Bi-LSTM': true }};

function renderCombined() {{
  const traces = [
    {{
      x: D.testDates, y: D.yTestActual,
      type: 'scatter', mode: 'lines',
      name: 'Actual',
      line: {{ color: '#f1f5f9', width: 2.5 }}
    }}
  ];
  Object.keys(D.models).forEach(name => {{
    traces.push({{
      x: D.testDates, y: D.models[name].testPreds,
      type: 'scatter', mode: 'lines',
      name: name + ' Predicted',
      line: {{ color: COLORS[name], width: 1.8 }},
      opacity: 0.88,
      visible: visibleModels[name] ? true : 'legendonly'
    }});
  }});
  Plotly.newPlot('chartCombined', traces, plotLayout, plotConfig);
}}

function toggleModel(btn, name) {{
  visibleModels[name] = !visibleModels[name];
  btn.classList.toggle('active');
  renderCombined();
}}

// ── Individual Model Chart ──
function showIndividual(name, btn) {{
  // Update buttons
  document.querySelectorAll('#chartIndividual').forEach(() => {{}});
  if (btn) {{
    btn.parentElement.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }}

  const errors = D.models[name].testPreds.map((p, i) => p - D.yTestActual[i]);

  const traces = [
    {{
      x: D.testDates, y: D.yTestActual,
      type: 'scatter', mode: 'lines',
      name: 'Actual', line: {{ color: '#f1f5f9', width: 2 }}
    }},
    {{
      x: D.testDates, y: D.models[name].testPreds,
      type: 'scatter', mode: 'lines',
      name: name + ' Predicted',
      line: {{ color: COLORS[name], width: 2 }},
      opacity: 0.9
    }},
    {{
      x: D.testDates, y: errors,
      type: 'bar', name: 'Error',
      marker: {{ color: errors.map(e => e >= 0 ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)') }},
      yaxis: 'y2'
    }}
  ];

  const layout = {{
    ...plotLayout,
    yaxis2: {{
      overlaying: 'y', side: 'right',
      gridcolor: 'rgba(0,0,0,0)',
      title: 'Error',
      showgrid: false,
      zerolinecolor: '#f59e0b44'
    }}
  }};

  Plotly.newPlot('chartIndividual', traces, layout, plotConfig);
}}

// ── Comparison Table ──
function renderCompTable() {{
  const table = document.getElementById('compTable');
  const names = Object.keys(D.models);

  // Find best for each metric
  let bestRMSE = names[0], bestMAE = names[0], bestR2 = names[0];
  names.forEach(n => {{
    if (D.models[n].metrics.RMSE < D.models[bestRMSE].metrics.RMSE) bestRMSE = n;
    if (D.models[n].metrics.MAE < D.models[bestMAE].metrics.MAE) bestMAE = n;
    if (D.models[n].metrics.R2 > D.models[bestR2].metrics.R2) bestR2 = n;
  }});

  let html = `<thead><tr>
    <th>Model</th><th>RMSE</th><th>MAE</th><th>R2 Score</th><th>Params</th><th>Layers</th>
  </tr></thead><tbody>`;

  names.forEach(name => {{
    const m = D.models[name];
    html += `<tr>
      <td><span class="model-dot" style="background:${{COLORS[name]}}"></span>${{name}}</td>
      <td class="${{name === bestRMSE ? 'highlight-best' : ''}}">${{m.metrics.RMSE.toFixed(4)}}</td>
      <td class="${{name === bestMAE ? 'highlight-best' : ''}}">${{m.metrics.MAE.toFixed(4)}}</td>
      <td class="${{name === bestR2 ? 'highlight-best' : ''}}">${{m.metrics.R2.toFixed(4)}}</td>
      <td>${{m.params.toLocaleString()}}</td>
      <td>${{m.layers}}</td>
    </tr>`;
  }});

  html += '</tbody>';
  table.innerHTML = html;
}}

// ── Metric Bars ──
function renderMetricBars() {{
  const names = Object.keys(D.models);
  const rmseVals = names.map(n => D.models[n].metrics.RMSE);
  const maeVals  = names.map(n => D.models[n].metrics.MAE);
  const r2Vals   = names.map(n => D.models[n].metrics.R2);

  const traces = [
    {{ x: names, y: rmseVals, type: 'bar', name: 'RMSE', marker: {{ color: '#f59e0b' }} }},
    {{ x: names, y: maeVals,  type: 'bar', name: 'MAE',  marker: {{ color: '#06b6d4' }} }},
    {{ x: names, y: r2Vals,   type: 'bar', name: 'R2',   marker: {{ color: '#10b981' }} }}
  ];

  const layout = {{
    ...plotLayout,
    barmode: 'group',
    yaxis: {{ ...plotLayout.yaxis, title: 'Value' }}
  }};

  Plotly.newPlot('chartMetricBars', traces, layout, plotConfig);
}}

// ── Error Distribution ──
function renderErrorDist() {{
  const traces = [];
  Object.keys(D.models).forEach(name => {{
    const errors = D.models[name].testPreds.map((p, i) => p - D.yTestActual[i]);
    traces.push({{
      x: errors, type: 'histogram', name: name,
      marker: {{ color: COLORS[name] }}, opacity: 0.65,
      nbinsx: 40
    }});
  }});
  const layout = {{
    ...plotLayout,
    barmode: 'overlay',
    xaxis: {{ ...plotLayout.xaxis, title: 'Prediction Error (USD)' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Frequency' }}
  }};
  Plotly.newPlot('chartErrorDist', traces, layout, plotConfig);
}}

// ── Cumulative Absolute Error ──
function renderCumError() {{
  const traces = [];
  Object.keys(D.models).forEach(name => {{
    const absErrors = D.models[name].testPreds.map((p, i) => Math.abs(p - D.yTestActual[i]));
    const cumSum = [];
    absErrors.reduce((acc, val, i) => {{ cumSum.push(acc + val); return acc + val; }}, 0);
    traces.push({{
      x: D.testDates, y: cumSum,
      type: 'scatter', mode: 'lines',
      name: name, line: {{ color: COLORS[name], width: 2 }}
    }});
  }});
  const layout = {{
    ...plotLayout,
    yaxis: {{ ...plotLayout.yaxis, title: 'Cumulative |Error| (USD)' }}
  }};
  Plotly.newPlot('chartCumError', traces, layout, plotConfig);
}}

// ── Init ──
renderMetricCards();
renderFullHistory();
renderCombined();
showIndividual('LSTM', null);
renderCompTable();
renderMetricBars();
renderErrorDist();
renderCumError();

// Fix first individual button
document.querySelectorAll('#chartIndividual')[0];
</script>
</body>
</html>"""

    output_path = os.path.join(BASE_DIR, "dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Dashboard saved -> {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("  Generating Interactive Dashboard")
    print("=" * 60)

    df, X_train, X_test, y_train, y_test, scaler, train_dates, test_dates = load_and_prepare_data()
    print(f"[INFO] Data loaded: {len(df)} rows, {len(X_test)} test samples")

    results = get_predictions_and_metrics(X_train, X_test, y_train, y_test, scaler)

    path = generate_dashboard(df, results, y_train, y_test, scaler, train_dates, test_dates)
    print(f"\n[INFO] Done! Open the dashboard in your browser:")
    print(f"       {path}")


if __name__ == "__main__":
    main()
