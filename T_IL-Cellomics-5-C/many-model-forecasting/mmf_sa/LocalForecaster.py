import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import statsmodels.api as sm
from mmf_sa.models import ModelRegistry


class LocalForecaster:
    def __init__(self, conf, data_conf, device="cpu"):
        """
        Initializes the LocalForecaster with configuration, data, and target device (CPU/GPU).

        Args:
            conf (OmegaConf): Forecasting configuration.
            data_conf (dict): Dictionary containing train_data and/or scoring_data.
            device (str): Target computation device ("cpu" or "cuda:0").
        """
        self.conf = conf
        self.data_conf = data_conf
        self.device = device
        self.model_registry = ModelRegistry(conf)
        self.run_date = datetime.now()
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def prepare_data_for_global_model(self, mode: str = None):
        """
        Prepares training data for the global model.

        Args:
            mode (str, optional): Mode indicator for future use (e.g., 'evaluating').

        Returns:
            tuple: (train DataFrame, empty list placeholder)
        """
        src_df = self.data_conf.get("train_data")
        if src_df is None:
            raise ValueError("train_data not found in data_conf")
        return src_df, []

    def split_df_train_val(self, df: pd.DataFrame):
        """
        Splits a DataFrame into training and validation sets based on timestamp.

        For each unique_id, the last 5 timepoints are used as validation.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: (train_df, val_df)
        """
        date_col = self.conf["date_col"]
        group_id = self.conf["group_id"]

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        df["_rank"] = (
            df.sort_values([group_id, date_col])
              .groupby(group_id)
              .cumcount()
        )

        group_sizes = (
            df.groupby(group_id)["_rank"]
              .max()
              .reset_index()
              .rename(columns={"_rank": "_last_rank"})
        )
        df = df.merge(group_sizes, on=group_id, how="left")
        df["is_val"] = df["_rank"] > (df["_last_rank"] - 5)

        train_df = df[df["is_val"] == False].drop(columns=["_rank", "_last_rank", "is_val"])
        val_df   = df[df["is_val"] == True ].drop(columns=["_rank", "_last_rank", "is_val"])

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def evaluate_foundation_model(self, model_conf):
        """
        Evaluates a foundation model on training/validation split and saves metrics.

        Args:
            model_conf (dict): Configuration dictionary for a specific model.
        """
        model_name = model_conf["name"]
        model = self.model_registry.get_model(model_name, device=self.device)

        print(f"Evaluating model: {model_name}")
        hist_df, _ = self.prepare_data_for_global_model("evaluating")
        train_df, val_df = self.split_df_train_val(hist_df)

        model.fit(train_df)

        date_col = self.conf["date_col"]
        
        res_pdf = model.backtest(
            full_df=pd.concat([train_df, val_df]),
            start=None, 
            train_df=train_df,
            val_df=val_df
        )

        res_pdf["model"] = model_name

        out_path = os.path.join(self.results_dir, f"{model_name}_metrics.csv")
        res_pdf.to_csv(out_path, index=False)
        print(f"Saved evaluation results to {out_path}")

    def score_foundation_model(self, model_conf):
        """
        Produces forecast for a foundation model and saves result to CSV.

        Args:
            model_conf (dict): Configuration dictionary for a specific model.
        """
        model_name = model_conf["name"]
        model = self.model_registry.get_model(model_name, device=self.device)

        print(f"Scoring model: {model_name}")

        df = self.data_conf.get("scoring_data") or self.data_conf.get("train_data")
        if df is None:
            raise ValueError("scoring_data or train_data must be provided")

        forecast_df, _ = model.forecast(df)
        forecast_df["model"] = model_name

        out_path = os.path.join(self.results_dir, f"{model_name}_forecast.csv")
        forecast_df.to_csv(out_path, index=False)
        print(f"Saved forecast to {out_path}")

    def evaluate_models(self):
        """
        Loops through all active foundation models and evaluates each.
        """
        print("Starting evaluate_models")
        for model_name in tqdm(
            self.model_registry.get_active_model_keys(),
            desc="MMF: evaluating",
            unit="model"
        ):
            t0 = time.time()
            print(f"\n→ Started evaluating {model_name} (time: {time.strftime('%H:%M:%S')})")

            try:
                model_conf = self.model_registry.get_model_conf(model_name)
                if model_conf["model_type"] == "foundation":
                    self.evaluate_foundation_model(model_conf)
            except Exception as err:
                print(f"  ⚠ Error while evaluating model {model_name}: {repr(err)}")

            t1 = time.time()
            print(f"✔ Finished evaluating {model_name} in {t1 - t0:.1f}s")
        print("Finished evaluate_models\n")

    def score_models(self):
        """
        Loops through all active foundation models and produces forecasts.
        """
        print("Starting score_models")
        for model_name in tqdm(
            self.model_registry.get_active_model_keys(),
            desc="MMF: scoring",
            unit="model"
        ):
            t0 = time.time()
            print(f"\n→ Started scoring {model_name} (time: {time.strftime('%H:%M:%S')})")

            try:
                model_conf = self.model_registry.get_model_conf(model_name)
                if model_conf["model_type"] == "foundation":
                    self.score_foundation_model(model_conf)
            except Exception as err:
                print(f"  ⚠ Error while scoring model {model_name}: {repr(err)}")

            t1 = time.time()
            print(f"✔ Finished scoring {model_name} in {t1 - t0:.1f}s")
        print("Finished score_models\n")
