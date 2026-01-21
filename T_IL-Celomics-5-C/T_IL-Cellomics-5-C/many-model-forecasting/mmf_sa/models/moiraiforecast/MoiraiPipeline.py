import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from typing import Iterator
from mmf_sa.models.abstract_model import ForecastingRegressor
from utilsforecast.processing import make_future_dataframe
from einops import rearrange
from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
from uni2ts.model.moirai_moe import MoiraiMoEForecast

EPS = 1e-8

def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))

def smape_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, EPS)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def mse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean((y_true - y_pred) ** 2))

def mape_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

class MoiraiForecaster(ForecastingRegressor):
    def __init__(self, params, repo: str = None, model_folder: str = None):
        super().__init__(params)
        self.params = params
        self.device = params.get("device", "cpu")

        base_dir = os.path.dirname(__file__)
        default_folder = "moirai-moe-1.0-R-base"
        chosen_folder = model_folder or default_folder
        self.repo = repo or os.path.join(base_dir, chosen_folder)

        if not os.path.isdir(self.repo):
            raise FileNotFoundError(
                f"Moirai model directory not found at '{self.repo}'"
            )
        self.model = None

    def forecast(self, df: pd.DataFrame, spark=None):
        print("[Moirai] Starting forecast()")
        hist_df = df[df[self.params["target"]].notnull()]
        if hist_df.empty:
            return None

        last_hist = hist_df[self.params["date_col"]].max()
        future_mask = (
            (df[self.params["date_col"]] > np.datetime64(last_hist)) &
            (df[self.params["date_col"]] <= np.datetime64(last_hist + self.prediction_length_offset))
        )
        val_df = df[future_mask]

        return self.predict(hist_df, val_df, last_hist)

    def predict(self, hist_df, val_df=None, curr_date=None, spark=None):
        print(f"[Moirai] Loading model from: {self.repo} on device: {self.device}")
        module = MoiraiModule.from_pretrained(self.repo, local_files_only=True)
        module.to(self.device)
        print(f"[Moirai] Model moved to: {next(module.parameters()).device}")

        is_scoring = val_df is None or val_df.empty
        last_hist_map = {}

        if is_scoring:
            last_hist_pd = (
                hist_df.groupby(self.params["group_id"])[self.params["date_col"]]
                .max()
                .reset_index()
                .rename(columns={self.params["date_col"]: "max_ds"})
            )
            last_hist_map = dict(zip(last_hist_pd[self.params["group_id"]], last_hist_pd["max_ds"]))

        all_forecasts = []
        all_timestamps = []
        unique_ids = hist_df[self.params["group_id"]].unique()

        for uid in tqdm(unique_ids, desc=f"Evaluating ({self.__class__.__name__})", unit="cell"):
            series = hist_df[hist_df[self.params["group_id"]] == uid]
            seq = series[self.params["target"]].tolist()

            if 'moe' in self.repo:
                model = MoiraiMoEForecast(
                    module=module,
                    prediction_length=self.params["prediction_length"],
                    context_length=len(seq),
                    patch_size=self.params.get("patch_size", 16),
                    num_samples=self.params["num_samples"],
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
            else:
                model = MoiraiForecast(
                    module=module,
                    prediction_length=self.params["prediction_length"],
                    context_length=len(seq),
                    patch_size=self.params.get("patch_size", 16),
                    num_samples=self.params["num_samples"],
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )

            past_target = rearrange(torch.tensor(seq, dtype=torch.float32), "t -> 1 t 1").to(self.device)
            past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

            forecast = model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
            )
            median_forecast = np.median(forecast[0].cpu().numpy(), axis=0)

            if not is_scoring:
                sub_val = val_df[val_df[self.params["group_id"]] == uid]
                val_ts = sub_val[self.params["date_col"]].sort_values().tolist()
                n_val = len(val_ts)
                if median_forecast.shape[0] >= n_val:
                    arr = median_forecast[:n_val]
                else:
                    pad_len = n_val - median_forecast.shape[0]
                    arr = np.concatenate([median_forecast, np.full(pad_len, np.nan)])
                all_forecasts.append(arr.tolist())
                all_timestamps.append([ts.to_pydatetime() for ts in val_ts])
            else:
                last_hist = pd.to_datetime(last_hist_map[uid])
                future_df = make_future_dataframe(
                    uids=[uid],
                    last_times=pd.Series([last_hist]),
                    h=self.params["prediction_length"],
                    freq=self.params["freq"]
                )
                fut_ts = future_df[self.params["date_col"]].tolist()
                all_forecasts.append(median_forecast.tolist())
                all_timestamps.append([ts.to_pydatetime() for ts in fut_ts])

        forecast_df = pd.DataFrame({
            self.params["group_id"]: unique_ids,
            self.params["date_col"]: all_timestamps,
            self.params["target"]: all_forecasts
        })
        return forecast_df, module

    def fit(self, train_df, spark=None):
        pass

    def backtest(self, full_df, start=None, train_df=None, val_df=None, spark=None):
        """
        Performs backtesting by forecasting validation portion and computing metrics.
        """
        if train_df is None or val_df is None:
            assert start is not None, "Either (train_df and val_df) or start must be provided"
            train_df = full_df[full_df[self.params["date_col"]] <= start]
            val_df = full_df[full_df[self.params["date_col"]] > start]

        curr_date = pd.to_datetime(train_df[self.params["date_col"]].max())
        forecast_df, _ = self.predict(train_df, val_df, curr_date)
        metrics = self.calculate_metrics(train_df, val_df, curr_date, forecast_df=forecast_df)
        return pd.DataFrame(metrics, columns=[
            self.params["group_id"], "ds", "metric", "score", "forecast", "actual", "extra"])


    def calculate_metrics(self, hist_df, val_df, curr_date, forecast_df=None, spark=None):
        if forecast_df is None:
            forecast_df, _ = self.predict(hist_df, val_df, curr_date)
            
        # ─── metric selection (no sktime, pure numpy) ───
        metric_name = self.params.get("metric", "rmse").lower()
        
        metric_fn = {
            "rmse": rmse_np,
            "mse": mse_np,
            "mae": mae_np,
            "mape": mape_np,
            "smape": smape_np,
        }.get(metric_name)
        
        if metric_fn is None:
            raise ValueError(f"unsupported metric: {metric_name}")

    
        results = []
    
        for uid in forecast_df[self.params["group_id"]].unique():
            row = forecast_df[forecast_df[self.params["group_id"]] == uid]
            if row.empty:
                continue
    
            forecast_array = np.asarray(row[self.params["target"]].iloc[0], dtype=np.float32)
            actual_array = np.asarray(
                val_df[val_df[self.params["group_id"]] == uid][self.params["target"]].to_numpy(),
                dtype=np.float32
            )
    
            if np.isnan(forecast_array).any():
                continue
            if np.isinf(forecast_array).any():
                continue
            if forecast_array.shape[0] != actual_array.shape[0]:
                continue
    
            try:
                score = metric_fn(actual_array, forecast_array)
            except Exception as e:
                continue
    
            results.append((
                uid,
                curr_date,
                metric_name,
                score,
                forecast_array.tolist(),
                actual_array.tolist(),
                b"",
            ))
    
        return results


# ========== Moirai Model Variants ==========
class MoiraiSmall(MoiraiForecaster):
    def __init__(self, params):
        folder = os.path.join(os.path.dirname(__file__), "moirai-1.0-R-small")
        super().__init__(params, repo=folder)

class MoiraiBase(MoiraiForecaster):
    def __init__(self, params):
        folder = os.path.join(os.path.dirname(__file__), "moirai-1.0-R-base")
        super().__init__(params, repo=folder)

class MoiraiLarge(MoiraiForecaster):
    def __init__(self, params):
        folder = os.path.join(os.path.dirname(__file__), "moirai-1.0-R-large")
        super().__init__(params, repo=folder)

class MoiraiMoESmall(MoiraiForecaster):
    def __init__(self, params):
        folder = os.path.join(os.path.dirname(__file__), "moirai-moe-1.0-R-small")
        super().__init__(params, repo=folder)

class MoiraiMoEBase(MoiraiForecaster):
    def __init__(self, params):
        folder = os.path.join(os.path.dirname(__file__), "moirai-moe-1.0-R-base")
        super().__init__(params, repo=folder)
