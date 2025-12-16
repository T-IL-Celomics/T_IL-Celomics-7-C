import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from typing import Iterator
from mmf_sa.models.abstract_model import ForecastingRegressor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoConfig

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

class ChronosForecaster(ForecastingRegressor):
    def __init__(self, params):
        """
        Base class for Chronos foundation models.
        Initializes configuration parameters and device.
        """
        super().__init__(params)
        self.params = params
        self.device = params.get("device", "cpu")  # Device passed via model params
        self.model = None

    def forecast(self, df: pd.DataFrame, spark=None) -> tuple:
        """
        Generates forecasts aligned to future timestamps based on historical data.

        Args:
            df (pd.DataFrame): Full dataframe with both history and future.
            spark (unused): Compatibility placeholder.

        Returns:
            tuple: (forecast DataFrame, model reference)
        """
        print("[Chronos] Starting forecast()")
        hist_df = df[df[self.params["target"]].notnull()]
        if hist_df.empty:
            return None

        last_hist = hist_df[self.params["date_col"]].max()
        future_mask = (
            (df[self.params["date_col"]] > np.datetime64(last_hist)) &
            (df[self.params["date_col"]] <= np.datetime64(last_hist + self.prediction_length_offset))
        )
        val_df = df[future_mask]

        forecast_df, model_inst = self.predict(
            hist_df=hist_df,
            val_df=val_df,
            curr_date=last_hist
        )

        return forecast_df, model_inst

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None, curr_date=None, spark=None):
        """
        Performs prediction using pretrained Chronos model. Aligns output to either:
        - Known timestamps from val_df (evaluation), or
        - Continuous horizon (scoring).

        Returns:
            pd.DataFrame with columns [group_id, date_col, target]
        """
        from chronos import BaseChronosPipeline
        from utilsforecast.processing import make_future_dataframe

        print("[Chronos] Initializing pipeline on device:", self.device)

        # Load config and force all model layers onto a single device
        config = AutoConfig.from_pretrained(self.repo)
        device_map = {"": self.device}

        pipeline = BaseChronosPipeline.from_pretrained(
            self.repo,
            device_map=device_map,
            torch_dtype=torch.float32,
        )

        unique_ids = hist_df[self.params["group_id"]].unique()
        all_forecasts = []
        all_timestamps = []

        is_scoring = val_df is None or val_df.empty

        for uid in tqdm(unique_ids, desc=f"Evaluating ({self.repo})", unit="cell"):
            series = hist_df[hist_df[self.params["group_id"]] == uid]

            context = torch.tensor(
                series[self.params["target"]].tolist(),
                dtype=torch.float32
            ).unsqueeze(0)

            forecast = pipeline.predict(
                context=context,
                prediction_length=self.params["prediction_length"]
            )

            median_forecast = np.median(forecast[0], axis=0)

            if not is_scoring:
                sorted_ts = val_df[val_df[self.params["group_id"]] == uid][self.params["date_col"]].sort_values().tolist()
                aligned_ts = [ts.to_pydatetime() for ts in sorted_ts]
                n_val = len(aligned_ts)

                if median_forecast.shape[0] >= n_val:
                    aligned_y = median_forecast[:n_val]
                else:
                    pad = n_val - median_forecast.shape[0]
                    aligned_y = np.concatenate([median_forecast, np.full(pad, np.nan)])
            else:
                last_hist = series[self.params["date_col"]].max()
                fut_df = make_future_dataframe(
                    uids=[uid],
                    last_times=pd.Series([last_hist]),
                    h=self.params["prediction_length"],
                    freq=self.params["freq"]
                )
                aligned_ts = fut_df[self.params["date_col"]].tolist()
                aligned_y = median_forecast

            all_forecasts.append(aligned_y.tolist())
            all_timestamps.append(aligned_ts)

        forecast_df = pd.DataFrame({
            self.params["group_id"]: unique_ids,
            self.params["date_col"]: all_timestamps,
            self.params["target"]: all_forecasts
        })

        return forecast_df, pipeline

    def fit(self, train_df: pd.DataFrame, spark=None):
        """No-op for pretrained foundation models."""
        pass

    def backtest(self, full_df, start=None, train_df=None, val_df=None, spark=None):
        """
        Performs backtesting by forecasting validation portion and computing metrics.

        Returns:
            pd.DataFrame with per-series metric results
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

# Define each Chronos model variant with proper repo string
class ChronosT5Tiny(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-t5-tiny"

class ChronosT5Mini(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-t5-mini"

class ChronosT5Small(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-t5-small"

class ChronosT5Base(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-t5-base"

class ChronosT5Large(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-t5-large"

class ChronosBoltTiny(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-bolt-tiny"

class ChronosBoltMini(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-bolt-mini"

class ChronosBoltSmall(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-bolt-small"

class ChronosBoltBase(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.repo = "amazon/chronos-bolt-base"
