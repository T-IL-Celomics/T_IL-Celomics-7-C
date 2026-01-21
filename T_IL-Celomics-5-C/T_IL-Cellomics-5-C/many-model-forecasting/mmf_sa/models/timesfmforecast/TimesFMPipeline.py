import pandas as pd
import numpy as np
from tqdm import tqdm
from utilsforecast.processing import make_future_dataframe
from mmf_sa.models.abstract_model import ForecastingRegressor

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

class TimesFMForecaster(ForecastingRegressor):
    def __init__(self, params):
        """
        Base class for TimesFM forecasting models.

        Args:
            params (dict): Configuration including model settings, data column names, etc.
        """
        super().__init__(params)
        self.params = params
        self.device = params.get("device", "cpu")
        self.model = None
        self.repo = None

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> pd.DataFrame:
        """
        Prepare input data for TimesFM.

        Args:
            df (pd.DataFrame): Input dataframe.
            future (bool): Whether preparing future covariates.

        Returns:
            pd.DataFrame: Cleaned and renamed dataframe.
        """
        if not future:
            cols = [self.params.group_id, self.params.date_col, self.params.target]
            _df = df[cols].rename(columns={
                self.params.group_id: "unique_id",
                self.params.date_col:   "ds",
                self.params.target:     "y",
            })
        else:
            cols = [self.params.group_id, self.params.date_col]
            _df = df[cols].rename(columns={
                self.params.group_id: "unique_id",
                self.params.date_col:   "ds",
            })
        return _df.sort_values(["unique_id", "ds"])

    def predict(self, hist_df, val_df=None, curr_date=None, spark=None):
        """
        Run TimesFM forecast.

        Args:
            hist_df (pd.DataFrame): Historical observations.
            val_df (pd.DataFrame): Future dates for alignment (optional).
            curr_date (datetime): Current timestamp.

        Returns:
            Tuple: (forecast_df, model instance)
        """
        df_hist = self.prepare_data(hist_df, future=False)
        if val_df is not None and not val_df.empty:
            df_future_cov = self.prepare_data(val_df, future=True)
        else:
            df_future_cov = pd.DataFrame(columns=["unique_id", "ds"])

        df_union = pd.concat([df_hist, df_future_cov], axis=0, join="inner", ignore_index=True) if not df_future_cov.empty else df_hist.copy()

        grouped = df_hist.groupby("unique_id")
        forecast_input = [grp["y"].values.astype(np.float64) for _, grp in grouped]

        freq_index = 0 if self.params.freq in ("H", "D") else 1

        dynamic_numerical_covariates = {}
        if "dynamic_future_numerical" in self.params:
            for var in self.params.dynamic_future_numerical:
                dynamic_numerical_covariates[var] = [grp[var].values for _, grp in df_union.groupby("unique_id")]

        dynamic_categorical_covariates = {}
        if "dynamic_future_categorical" in self.params:
            for var in self.params.dynamic_future_categorical:
                dynamic_categorical_covariates[var] = [grp[var].values for _, grp in df_union.groupby("unique_id")]

        static_numerical_covariates = {}
        static_categorical_covariates = {}
        if "static_features" in self.params:
            for var in self.params.static_features:
                if pd.api.types.is_numeric_dtype(df_hist[var]):
                    static_numerical_covariates[var] = [grp[var].iloc[0] for _, grp in df_hist.groupby("unique_id")]
                else:
                    static_categorical_covariates[var] = [grp[var].iloc[0] for _, grp in df_hist.groupby("unique_id")]

        if not dynamic_numerical_covariates and not dynamic_categorical_covariates and not static_numerical_covariates and not static_categorical_covariates:
            forecasts, _ = self.model.forecast(
                inputs=forecast_input,
                freq=[freq_index] * len(forecast_input),
            )
        else:
            forecasts, _ = self.model.forecast_with_covariates(
                inputs=forecast_input,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                dynamic_categorical_covariates=dynamic_categorical_covariates,
                static_numerical_covariates=static_numerical_covariates,
                static_categorical_covariates=static_categorical_covariates,
                freq=[freq_index] * len(forecast_input),
                xreg_mode="xreg + timesfm",
                ridge=0.0,
                force_on_cpu=False,
                normalize_xreg_target_per_input=True,
            )

        unique_ids = list(df_hist["unique_id"].unique())
        aligned_timestamps = []
        aligned_forecasts = []

        is_scoring = val_df is None or val_df.empty
        if not is_scoring:
            for i, uid in enumerate(tqdm(unique_ids, desc=f"Evaluating ({self.repo})", unit="cell")):
                raw_forecast = np.array(forecasts[i], dtype=np.float64)
                sub_val = val_df[val_df[self.params.group_id] == uid]
                if sub_val.empty:
                    aligned_timestamps.append([])
                    aligned_forecasts.append([])
                    continue
                sorted_ts = sub_val[self.params.date_col].sort_values().tolist()
                val_timestamps = [ts.to_pydatetime() for ts in sorted_ts]
                n_val = len(val_timestamps)
                arr = raw_forecast[:n_val] if raw_forecast.shape[0] >= n_val else np.concatenate([raw_forecast, np.full((n_val - raw_forecast.shape[0],), np.nan)])
                aligned_timestamps.append(val_timestamps)
                aligned_forecasts.append(arr.tolist())
        else:
            last_hist_map = {uid: ts_array.max() for uid, ts_array in df_hist.groupby("unique_id")["ds"]}
            for i, uid in enumerate(tqdm(unique_ids, desc=f"Evaluating ({self.repo})", unit="cell")):
                raw_forecast = np.array(forecasts[i], dtype=np.float64)
                last_hist = last_hist_map[uid]
                future_df = make_future_dataframe(
                    uids=[uid],
                    last_times=pd.Series([last_hist]),
                    h=self.params["prediction_length"],
                    freq=self.params["freq"],
                )
                fut_ts = future_df[self.params["date_col"]].iloc[0]
                if isinstance(fut_ts, pd.Timestamp):
                    fut_ts = [fut_ts]
                fut_datetimes = [ts.to_pydatetime() for ts in fut_ts]
                aligned_timestamps.append(fut_datetimes)
                aligned_forecasts.append(raw_forecast.tolist())

        forecast_df = pd.DataFrame({
            self.params["group_id"]: unique_ids,
            self.params["date_col"]: aligned_timestamps,
            self.params["target"]: aligned_forecasts,
        })

        forecast_df[self.params["target"]] = forecast_df[self.params["target"]].apply(lambda arr: [float(x) for x in arr])
        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
        """
        Wrapper for predict that splits input into history and future.
        """
        hist_df = df[df[self.params.target].notnull()]
        if hist_df.empty:
            return pd.DataFrame(columns=[self.params.group_id, self.params.date_col, self.params.target]), self.model

        last_hist = hist_df[self.params.date_col].max()
        future_mask = (
            (df[self.params.date_col] > np.datetime64(last_hist)) &
            (df[self.params.date_col] <= np.datetime64(last_hist + self.prediction_length_offset))
        )
        val_df = df[future_mask]

        return self.predict(hist_df, val_df, last_hist)

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


    def backtest(self, full_df, start=None, train_df=None, val_df=None, spark=None):
        """
        Backtest TimesFM model using per-cell train/val split.
        """
        if train_df is None or val_df is None:
            assert start is not None, "Either (train_df and val_df) or start must be provided"
            train_df = full_df[full_df[self.params.date_col] <= start]
            val_df = full_df[full_df[self.params.date_col] > start]

        curr_date = pd.to_datetime(train_df[self.params.date_col].max())
        forecast_df, _ = self.predict(train_df, val_df, curr_date)
        metrics = self.calculate_metrics(train_df, val_df, curr_date, forecast_df=forecast_df)

        return pd.DataFrame(metrics, columns=[
            self.params["group_id"], "ds", "metric", "score", "forecast", "actual", "extra"])

class TimesFM_1_0_200m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        import timesfm
        self.repo = "google/timesfm-1.0-200m-pytorch"
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=self.params.prediction_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.repo
            ),
        )
        # Explicitly move model to device
        device = params.get("device", "cpu")
        print(f"[TimesFM] Model moved to device: {device}")


class TimesFM_2_0_500m(TimesFMForecaster):
    def __init__(self, params):
        """Initialize TimesFM 2.0 500M model."""
        super().__init__(params)
        import timesfm
        self.repo = "google/timesfm-2.0-500m-pytorch"
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=self.params.prediction_length,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.repo
            ),
        )
        # Explicitly move model to device
        device = params.get("device", "cpu")
        print(f"[TimesFM] Model moved to device: {device}")


__all__ = ["TimesFM_1_0_200m", "TimesFM_2_0_500m"]
