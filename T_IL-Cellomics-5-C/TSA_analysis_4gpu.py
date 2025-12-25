import os
import sys
import glob
import ast
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
import json
import time

# ========== 1) Setup Python path to include MMF repo ==========
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "many-model-forecasting"))
os.environ["PYTHONPATH"] = repo_path + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, repo_path)

from mmf_sa.LocalForecaster import LocalForecaster

# ========== 2) Load forecasting configuration ==========
conf = OmegaConf.load("my_models_conf.yaml").forecasting
conf.freq = conf.freq.upper()
conf.train_data   = "train_data"
conf.scoring_data = "scoring_data"
conf.prediction_length = 5
conf.backtest_length   = 5
conf.limit_num_series = -1

# ========== 3) Detect GPU ==========

if not torch.cuda.is_available():
    raise RuntimeError("cuda not available, refusing to run on cpu")
device = "cuda:0"
torch.cuda.set_device(0)
print(f"[Main] using device: {device}")


# ========== 4) Load sample data ==========
sample_file = "cell_data/raw_all_cells.csv"
if not os.path.exists(sample_file):
    raise FileNotFoundError(f"Sample file '{sample_file}' not found.")

# ========== 5) Load selected features ==========
with open("cell_data/selected_features.txt", "r") as f:
    features_list = [line.strip() for line in f if line.strip()]

#  4-gpu sharding by feature 
shard_idx = int(os.environ.get("SHARD_IDX", "0"))
num_shards = int(os.environ.get("NUM_SHARDS", "1"))
features_list = [f for i, f in enumerate(features_list) if (i % num_shards) == shard_idx]
print(f"[Shard] {shard_idx}/{num_shards}: {len(features_list)} features")


# ========== 6) Preprocess input data ==========
df_raw = pd.read_csv(sample_file)
df_raw['ds'] = pd.to_datetime(df_raw['ds'], infer_datetime_format=True, errors='coerce')
if df_raw['ds'].isna().any():
    missing = df_raw['ds'].isna().sum()
    print(f"Warning: {missing} rows of 'ds' could not be parsed and will be dropped.")
    df_raw = df_raw.dropna(subset=['ds'])
# Subsample 500 unique cells
MAX_CELLS = int(os.environ.get("MAX_CELLS", "500"))  # set -1 for all
unique_ids = df_raw["unique_id"].dropna().unique()
if MAX_CELLS != -1 and len(unique_ids) > MAX_CELLS:
    np.random.seed(42)
    selected_ids = np.random.choice(unique_ids, size=MAX_CELLS, replace=False)
    df_raw = df_raw[df_raw["unique_id"].isin(selected_ids)]
print(f"[data] using {df_raw['unique_id'].nunique()} cells (MAX_CELLS={MAX_CELLS})")


# ========== 7) Initialize top models dictionary ==========
top_models = {}

# ========== 8) Iterate over each feature ==========
t_total0 = time.time()
for feature in features_list:
    t0 = time.time()
    print(f"\n=== Processing feature: {feature} ===")
    feature_dir = os.path.join("results", feature)
    os.makedirs(feature_dir, exist_ok=True)

    df_for_mmf = df_raw[["unique_id", "ds", feature]].rename(columns={feature: "y"})

    forecaster = LocalForecaster(
        conf=conf,
        data_conf={"train_data": df_for_mmf, "scoring_data": df_for_mmf},
        device=device  # <<< pass device to ensure GPU usage
    )
    forecaster.results_dir = feature_dir
    print(f"Metrics will save into: {forecaster.results_dir}")

    if torch.cuda.is_available(): torch.cuda.synchronize()
    forecaster.evaluate_models()
    if torch.cuda.is_available(): torch.cuda.synchronize()

    print(f"[time] feature={feature} evaluate_models: {time.time() - t0:.2f}s")
    rmse_list = []
    chronos_list = []

    for mf in glob.glob(os.path.join(feature_dir, "*_metrics.csv")):
        model_name = os.path.basename(mf).replace("_metrics.csv", "")
        dfm = pd.read_csv(mf, converters={"forecast": lambda s: ast.literal_eval(s) if pd.notna(s) else []})
        dfm = dfm[dfm["forecast"].apply(len) >= conf.prediction_length]
        if not dfm.empty:
            avg_rmse = dfm["score"].mean()
            rmse_list.append((model_name, avg_rmse))
            if "chronost5" in model_name.lower():
                chronos_list.append((model_name, avg_rmse))

    if not rmse_list:
        print(f"No valid runs for feature {feature}; skipping.")
        continue

    best_model, best_rmse = min(rmse_list, key=lambda x: x[1])
    top_models[feature] = best_model
    top_models[f"{feature}_best_model_rmse"] = best_rmse
    print(f" → Best model for {feature}: {best_model} (RMSE={best_rmse:.4f})")

    if "chronos" in best_model.lower():
        # If the best model is already a Chronos model, also store it as best_chronos_model
        top_models[f"{feature}_best_chronos"] = best_model
        top_models[f"{feature}_best_chronos_rmse"] = best_rmse
    elif chronos_list:
        # Otherwise, select the best Chronos model separately (if any exist)
        best_chronos_model, best_chronos_rmse = min(chronos_list, key=lambda x: x[1])
        top_models[f"{feature}_best_chronos"] = best_chronos_model
        top_models[f"{feature}_best_chronos_rmse"] = best_chronos_rmse
        print(f" → Best **Chronos** model for {feature}: {best_chronos_model} (RMSE={best_chronos_rmse:.4f})")

# ========== 9) Save final summary ==========
print("\n=== Best model per feature ===")
print(top_models)

results = {}
for k, v in top_models.items():
    if "_best_chronos_rmse" in k:
        feat = k.replace("_best_chronos_rmse", "")
        results.setdefault(feat, {})["best_chronos_rmse"] = v
    elif "_best_chronos" in k:
        feat = k.replace("_best_chronos", "")
        results.setdefault(feat, {})["best_chronos_model"] = v
    elif "_best_model_rmse" in k:
        feat = k.replace("_best_model_rmse", "")
        results.setdefault(feat, {})["best_model_rmse"] = v
    else:
        results.setdefault(k, {})["best_model"] = v

mapping_df = pd.DataFrame([
    {
        "feature": feat,
        "best_model": d.get("best_model"),
        "best_model_rmse": d.get("best_model_rmse"),
        "best_chronos_model": d.get("best_chronos_model"),
        "best_chronos_rmse": d.get("best_chronos_rmse")
    }
    for feat, d in results.items()
])

csv_path = "best_model_per_feature.csv"
mapping_df.to_csv(csv_path, index=False)
print(f"Saved best model mapping to {csv_path}")

# ========== 10) Save as JSON as well ==========

# General best model per feature
feature_model_dict = {
    feat: d.get("best_model")
    for feat, d in results.items()
    if "best_model" in d
}

# Chronos best model per feature
chronos_model_dict = {
    feat: d.get("best_chronos_model")
    for feat, d in results.items()
    if "best_chronos_model" in d
}

with open("best_model_per_feature.json", "w") as f:
    json.dump(feature_model_dict, f, indent=4)

with open("best_chronos_model_per_feature.json", "w") as f:
    json.dump(chronos_model_dict, f, indent=4)

print("Saved best model mappings to JSON files.")
print(f"[time] total runtime: {time.time() - t_total0:.2f}s")

