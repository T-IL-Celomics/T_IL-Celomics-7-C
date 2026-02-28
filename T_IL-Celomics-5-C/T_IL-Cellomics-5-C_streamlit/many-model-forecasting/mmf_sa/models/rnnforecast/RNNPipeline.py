"""
Simple baseline forecasters for the MMF framework:
  - SimpleLSTM / SimpleGRU   (recurrent)
  - SimpleDLinear             (linear decomposition, from "Are Transformers Effective for TSF?")
  - SimpleAutoformer          (auto-correlation transformer, from "Autoformer: Decomposition Transformers")

Each model trains a small per-series network on the fly (no pretrained weights),
producing output in the same format as the Chronos pipeline so results
are directly comparable.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mmf_sa.models.abstract_model import ForecastingRegressor

# Disable cuDNN for RNNs — avoids CUDNN_STATUS_EXECUTION_FAILED on some GPUs
# (the models are tiny so the cuDNN kernel overhead isn't beneficial anyway).
torch.backends.cudnn.enabled = False

# ──────────────────────────────────────────────────────────────
# Pure-numpy metrics (same as ChronosPipeline)
# ──────────────────────────────────────────────────────────────
EPS = 1e-8

def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, np.float32) - np.asarray(y_pred, np.float32)) ** 2)))

def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, np.float32) - np.asarray(y_pred, np.float32))))

def mse_np(y_true, y_pred):
    return float(np.mean((np.asarray(y_true, np.float32) - np.asarray(y_pred, np.float32)) ** 2))

def smape_np(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, np.float32), np.asarray(y_pred, np.float32)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, EPS)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def mape_np(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, np.float32), np.asarray(y_pred, np.float32)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

METRIC_FN = {"rmse": rmse_np, "mse": mse_np, "mae": mae_np, "smape": smape_np, "mape": mape_np}


# ──────────────────────────────────────────────────────────────
# Tiny PyTorch RNN modules
# ──────────────────────────────────────────────────────────────
class _LSTMNet(nn.Module):
    """Single-layer LSTM → linear head."""
    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # last step → (batch, 1)


class _GRUNet(nn.Module):
    """Single-layer GRU → linear head."""
    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# ──────────────────────────────────────────────────────────────
# DLinear  — "Are Transformers Effective for Time Series Forecasting?"
#  Decomposes input into trend (moving average) + remainder,
#  applies a separate linear layer to each, then sums.
# ──────────────────────────────────────────────────────────────
class _DLinearNet(nn.Module):
    """
    DLinear: input (batch, seq_len, 1) → output (batch, 1).
    Uses a simple moving-average kernel for trend decomposition.
    """
    def __init__(self, seq_len: int, kernel_size: int = 3, **_):
        super().__init__()
        self.seq_len = seq_len
        # Moving-average for trend extraction
        pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)
        # Two independent linear projections: trend → 1, remainder → 1
        self.linear_trend = nn.Linear(seq_len, 1)
        self.linear_resid = nn.Linear(seq_len, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x_1d = x.squeeze(-1)               # (batch, seq_len)
        trend = self.avg(x_1d.unsqueeze(1)).squeeze(1)  # (batch, seq_len)
        resid = x_1d - trend
        out = self.linear_trend(trend) + self.linear_resid(resid)  # (batch, 1)
        return out


# ──────────────────────────────────────────────────────────────
# Autoformer (simplified single-layer)
#  Auto-correlation mechanism + series decomposition.
#  Adapted for single-step prediction from a window.
# ──────────────────────────────────────────────────────────────
class _AutoCorrelation(nn.Module):
    """Period-based attention via FFT auto-correlation (simplified)."""
    def __init__(self, d_model: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        B, L, D = Q.shape

        # Auto-correlation via FFT
        q_fft = torch.fft.rfft(Q, dim=1)
        k_fft = torch.fft.rfft(K, dim=1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=1)  # (B, L, D)

        # Top-k lags
        mean_corr = corr.mean(dim=-1)  # (B, L)
        topk_vals, topk_idx = torch.topk(mean_corr, self.top_k, dim=1)
        weights = torch.softmax(topk_vals, dim=-1)  # (B, top_k)

        # Aggregate V by rolling with top-k lags
        agg = torch.zeros_like(V)
        for i in range(self.top_k):
            lag = topk_idx[:, i]  # (B,)
            for b in range(B):
                agg[b] += weights[b, i] * torch.roll(V[b], shifts=int(lag[b].item()), dims=0)
        return agg


class _AutoformerNet(nn.Module):
    """
    Simplified single-layer Autoformer: embedding → auto-correlation → decompose → linear → 1 output.
    Input: (batch, seq_len, 1) → Output: (batch, 1)
    """
    def __init__(self, seq_len: int, d_model: int = 32, kernel_size: int = 3, top_k: int = 2, **_):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)
        self.auto_corr = _AutoCorrelation(d_model, top_k=top_k)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(seq_len * d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        h = self.embed(x)                          # (B, L, d_model)
        # Auto-correlation block
        ac = self.auto_corr(h)                      # (B, L, d_model)
        h = self.norm(h + ac)                       # residual + norm
        # Series decomposition (trend removal)
        h_flat = h.permute(0, 2, 1)                 # (B, d_model, L)
        trend = self.avg(h_flat).permute(0, 2, 1)   # (B, L, d_model)
        seasonal = h - trend
        out = seasonal.reshape(seasonal.size(0), -1) # (B, L*d_model)
        return self.fc(out)                          # (B, 1)


# ──────────────────────────────────────────────────────────────
# Helper: create sliding-window X/y from a 1-D array
# ──────────────────────────────────────────────────────────────
def _make_windows(series: np.ndarray, window_size: int):
    """Returns (X, y) where X.shape=(n, window_size, 1), y.shape=(n, 1)."""
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    if not X:
        return np.empty((0, window_size, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
    X = np.asarray(X, dtype=np.float32).reshape(-1, window_size, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y


# ──────────────────────────────────────────────────────────────
# Base RNN Forecaster (shared logic for LSTM / GRU)
# ──────────────────────────────────────────────────────────────
class RNNForecaster(ForecastingRegressor):
    """
    Trains a small RNN per series at backtest time.
    Uses the same output schema as the Chronos pipeline so
    metrics CSVs are fully comparable.
    """

    # Subclasses override this
    _rnn_class = None  # _LSTMNet or _GRUNet

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = params.get("device", "cpu")
        self.hidden_size = int(params.get("hidden_size", 64))
        self.num_layers = int(params.get("num_layers", 1))
        self.max_epochs = int(params.get("max_epochs", 150))
        self.learning_rate = float(params.get("learning_rate", 1e-3))
        self.window_size = int(params.get("window_size", 5))
        self.patience = int(params.get("patience", 15))  # early-stop patience

    # ---------- helpers ----------
    def _build_net(self):
        """Instantiate the network. Subclasses can override for non-RNN archs."""
        return self._rnn_class(self.hidden_size, self.num_layers).to(self.device)

    def _train_one_series(self, values: np.ndarray):
        """Train on a single 1-D series. Returns the trained model."""
        net = self._build_net()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        X, y = _make_windows(values, self.window_size)
        if len(X) == 0:
            return net  # nothing to train on

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=min(64, len(X)), shuffle=True)

        best_loss, wait = float("inf"), 0
        for epoch in range(self.max_epochs):
            net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                pred = net(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(X)
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.eval()
        return net

    def _forecast_one_series(self, net: nn.Module, context: np.ndarray, horizon: int):
        """Auto-regressively forecast `horizon` steps from `context`."""
        buf = list(context[-self.window_size:])
        preds = []
        with torch.no_grad():
            for _ in range(horizon):
                inp = torch.tensor(
                    np.asarray(buf[-self.window_size:], dtype=np.float32).reshape(1, -1, 1),
                    device=self.device,
                )
                val = net(inp).item()
                preds.append(val)
                buf.append(val)
        return np.asarray(preds, dtype=np.float32)

    # ---------- interface required by MMF ----------
    def prepare_data(self, df):
        return df

    def fit(self, train_df, spark=None):
        """No-op — training happens inside backtest per series."""
        pass

    def predict(self, hist_df, val_df=None, curr_date=None, spark=None):
        """
        For each unique_id: train RNN on hist_df, forecast len(val_df) steps.
        Returns (forecast_df, None)  — same schema as Chronos.
        """
        group_col = self.params["group_id"]
        date_col = self.params["date_col"]
        target_col = self.params["target"]
        horizon = int(self.params["prediction_length"])

        all_uids, all_timestamps, all_forecasts = [], [], []

        uids = hist_df[group_col].unique()
        for uid in uids:
            hist = hist_df[hist_df[group_col] == uid].sort_values(date_col)
            values = hist[target_col].to_numpy(dtype=np.float32)

            # Normalise for stability
            mu, sigma = float(np.nanmean(values)), float(np.nanstd(values))
            if sigma < EPS:
                sigma = 1.0
            normed = (values - mu) / sigma

            net = self._train_one_series(normed)
            raw_pred = self._forecast_one_series(net, normed, horizon)
            forecast = raw_pred * sigma + mu  # de-normalise

            # Align timestamps
            if val_df is not None and not val_df.empty:
                ts = val_df[val_df[group_col] == uid][date_col].values[:horizon]
            else:
                last_ts = hist[date_col].max()
                ts = pd.date_range(start=last_ts, periods=horizon + 1, freq=self.freq)[1:]

            all_uids.append(uid)
            all_timestamps.append(ts[:len(forecast)])
            all_forecasts.append(forecast[:len(ts)].tolist())

        forecast_df = pd.DataFrame({
            group_col: all_uids,
            date_col: all_timestamps,
            target_col: all_forecasts,
        })
        return forecast_df, None

    def forecast(self, df, spark=None):
        hist_df = df[df[self.params["target"]].notnull()]
        if hist_df.empty:
            return None, None
        last_hist = hist_df[self.params["date_col"]].max()
        future_mask = (
            (df[self.params["date_col"]] > np.datetime64(last_hist)) &
            (df[self.params["date_col"]] <= np.datetime64(last_hist + self.prediction_length_offset))
        )
        val_df = df[future_mask]
        return self.predict(hist_df, val_df, last_hist)

    def backtest(self, full_df, start=None, train_df=None, val_df=None, spark=None):
        if train_df is None or val_df is None:
            assert start is not None
            train_df = full_df[full_df[self.params["date_col"]] <= start]
            val_df = full_df[full_df[self.params["date_col"]] > start]

        curr_date = pd.to_datetime(train_df[self.params["date_col"]].max())
        forecast_df, _ = self.predict(train_df, val_df, curr_date)
        metrics = self.calculate_metrics(train_df, val_df, curr_date, forecast_df=forecast_df)

        return pd.DataFrame(metrics, columns=[
            self.params["group_id"], "ds", "metric", "score", "forecast", "actual", "extra",
        ])

    def calculate_metrics(self, hist_df, val_df, curr_date, forecast_df=None, spark=None):
        if forecast_df is None:
            forecast_df, _ = self.predict(hist_df, val_df, curr_date)

        metric_name = self.params.get("metric", "rmse").lower()
        metric_fn = METRIC_FN.get(metric_name)
        if metric_fn is None:
            raise ValueError(f"Unsupported metric: {metric_name}")

        group_col = self.params["group_id"]
        target_col = self.params["target"]
        results = []

        for uid in forecast_df[group_col].unique():
            row = forecast_df[forecast_df[group_col] == uid]
            if row.empty:
                continue

            forecast_array = np.asarray(row[target_col].iloc[0], dtype=np.float32)
            actual_array = np.asarray(
                val_df[val_df[group_col] == uid][target_col].to_numpy(), dtype=np.float32
            )

            if np.isnan(forecast_array).any() or np.isinf(forecast_array).any():
                continue
            if forecast_array.shape[0] != actual_array.shape[0]:
                continue

            try:
                score = metric_fn(actual_array, forecast_array)
            except Exception:
                continue

            results.append((uid, curr_date, metric_name, score,
                            forecast_array.tolist(), actual_array.tolist(), b""))

        return results


# ──────────────────────────────────────────────────────────────
# Concrete model classes (registered in YAML)
# ──────────────────────────────────────────────────────────────
class SimpleLSTM(RNNForecaster):
    """Simple single-layer LSTM forecaster."""
    _rnn_class = _LSTMNet


class SimpleGRU(RNNForecaster):
    """Simple single-layer GRU forecaster."""
    _rnn_class = _GRUNet


class SimpleDLinear(RNNForecaster):
    """DLinear (linear trend/residual decomposition) forecaster."""
    _rnn_class = _DLinearNet  # not really an RNN — just reusing the training loop

    def _build_net(self):
        return _DLinearNet(seq_len=self.window_size).to(self.device)


class SimpleAutoformer(RNNForecaster):
    """Simplified Autoformer (FFT auto-correlation) forecaster."""
    _rnn_class = _AutoformerNet

    def _build_net(self):
        return _AutoformerNet(seq_len=self.window_size).to(self.device)
