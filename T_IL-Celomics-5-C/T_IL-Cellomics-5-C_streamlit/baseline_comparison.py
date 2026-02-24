"""
baseline_comparison.py
=====================
Standalone script that runs baseline models (SimpleLSTM, SimpleGRU,
SimpleDLinear, SimpleAutoformer) on the same cell data used by the MMF
pipeline, then produces a comparison table against the existing MMF
results — including percentage-improvement metrics
(e.g. "X% lower MSE & Y% lower MAE vs. baselines").

This script is NOT part of the MMF pipeline — it reads the already-saved
MMF metrics CSVs and adds baseline columns for side-by-side comparison.

Usage:
    python baseline_comparison.py                              # single GPU, all features
    SHARD_IDX=0 NUM_SHARDS=4 python baseline_comparison.py    # shard-aware

Outputs:
    results/<feature>/<ModelName>_metrics.csv   — per-cell metrics per baseline
    baseline_comparison.csv                     — per-feature comparison table
    baseline_comparison.json                    — same, as JSON
    baseline_comparison_summary.txt             — headline % improvement numbers
"""

import os
import sys
import glob
import ast
import json
import time
import numpy as np
import pandas as pd
import torch

# ── Setup MMF import path (only for the RNN model classes) ──
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "many-model-forecasting"))
sys.path.insert(0, repo_path)

from mmf_sa.models.rnnforecast.RNNPipeline import SimpleLSTM, SimpleGRU, SimpleDLinear, SimpleAutoformer
from omegaconf import OmegaConf

# ══════════════════════════════════════════════════════════════
# Metric helpers (recompute MSE/MAE/RMSE from forecast & actual arrays)
# ══════════════════════════════════════════════════════════════
EPS = 1e-8

def _compute_all_metrics_from_csv(csv_path, prediction_length):
    """Read a *_metrics.csv and recompute MSE, MAE, RMSE from the stored
    forecast/actual arrays.  Returns dict {mse, mae, rmse} or None."""
    try:
        dfm = pd.read_csv(csv_path, converters={
            "forecast": lambda s: ast.literal_eval(s) if pd.notna(s) else [],
            "actual":   lambda s: ast.literal_eval(s) if pd.notna(s) else [],
        })
    except Exception:
        return None

    dfm = dfm[dfm["forecast"].apply(lambda x: len(x) if isinstance(x, list) else 0) >= prediction_length]
    if dfm.empty:
        return None

    all_sq, all_abs = [], []
    for _, r in dfm.iterrows():
        f = np.asarray(r["forecast"], dtype=np.float32)
        a = np.asarray(r["actual"],   dtype=np.float32)
        if f.shape != a.shape or np.isnan(f).any() or np.isinf(f).any():
            continue
        all_sq.append(np.mean((a - f) ** 2))
        all_abs.append(np.mean(np.abs(a - f)))

    if not all_sq:
        return None

    mse = float(np.mean(all_sq))
    mae = float(np.mean(all_abs))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}

# ══════════════════════════════════════════════════════════════
# 1) Configuration
# ══════════════════════════════════════════════════════════════
PREDICTION_LENGTH = 5
FREQ = "H"
METRIC = "rmse"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[baseline comparison] device: {device}")

# ── Data paths (same defaults as TSA_analysis_4gpu.py) ──
sample_file = os.environ.get("PIPELINE_RAW_CSV", "cell_data/raw_all_cells.csv")
if not os.path.exists(sample_file):
    raise FileNotFoundError(f"Sample file '{sample_file}' not found.")

features_file = os.environ.get("PIPELINE_FEATURES_FILE", "cell_data/selected_features.txt")
with open(features_file, "r") as f:
    features_list = [line.strip() for line in f if line.strip()]

# ── Shard-awareness (matches TSA_analysis_4gpu.py) ──
shard_idx = int(os.environ.get("SHARD_IDX", "0"))
num_shards = int(os.environ.get("NUM_SHARDS", "1"))
features_list = [feat for i, feat in enumerate(features_list) if (i % num_shards) == shard_idx]
print(f"[Shard] {shard_idx}/{num_shards}: {len(features_list)} features")

# ── Load raw data ──
df_raw = pd.read_csv(sample_file)
df_raw["ds"] = pd.to_datetime(df_raw["ds"], errors="coerce")
df_raw = df_raw.dropna(subset=["ds"])

# MAX_CELLS is only used as a fallback when no MMF results exist yet.
MAX_CELLS = int(os.environ.get("PIPELINE_MAX_CELLS", os.environ.get("MAX_CELLS", "500")))
all_unique_ids = df_raw["unique_id"].dropna().unique()
print(f"[data] loaded {len(all_unique_ids)} cells total (MAX_CELLS fallback={MAX_CELLS})")


def _get_cells_from_csv(csv_path: str) -> set:
    """Return the set of unique_ids present in a single metrics CSV."""
    try:
        col = pd.read_csv(csv_path, usecols=["unique_id"], dtype=str)["unique_id"]
        return set(col.dropna().unique())
    except Exception:
        return set()


def _subsample_cells(df: pd.DataFrame, cell_ids: np.ndarray) -> pd.DataFrame:
    """Fallback: subsample to MAX_CELLS using the same seed=42 as MMF."""
    if MAX_CELLS > 0 and len(cell_ids) > MAX_CELLS:
        np.random.seed(42)
        cell_ids = np.random.choice(cell_ids, size=MAX_CELLS, replace=False)
    return df[df["unique_id"].isin(cell_ids)]


# ══════════════════════════════════════════════════════════════
# 2) Helper: train/val split (same as LocalForecaster.split_df_train_val)
# ══════════════════════════════════════════════════════════════
def split_train_val(df, date_col="ds", group_col="unique_id", n_val=5):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_rank"] = df.sort_values([group_col, date_col]).groupby(group_col).cumcount()
    last = df.groupby(group_col)["_rank"].max().reset_index().rename(columns={"_rank": "_last"})
    df = df.merge(last, on=group_col, how="left")
    df["is_val"] = df["_rank"] > (df["_last"] - n_val)
    train = df[~df["is_val"]].drop(columns=["_rank", "_last", "is_val"]).reset_index(drop=True)
    val = df[df["is_val"]].drop(columns=["_rank", "_last", "is_val"]).reset_index(drop=True)
    return train, val


# ══════════════════════════════════════════════════════════════
# 3) Helper: build model params OmegaConf (not tied to MMF YAML)
# ══════════════════════════════════════════════════════════════
def _make_baseline_params(model_class_name: str) -> OmegaConf:
    return OmegaConf.create({
        "freq": FREQ,
        "prediction_length": PREDICTION_LENGTH,
        "backtest_length": PREDICTION_LENGTH,
        "stride": 1,
        "date_col": "ds",
        "group_id": "unique_id",
        "target": "y",
        "metric": METRIC,
        "device": device,
        # RNN / baseline hyperparameters
        "hidden_size": 64,
        "num_layers": 1,
        "window_size": 5,
        "max_epochs": 150,
        "learning_rate": 0.001,
        "patience": 15,
        # MMF bookkeeping (not used, but keeps base class happy)
        "module": "mmf_sa.models.rnnforecast.RNNPipeline",
        "model_class": model_class_name,
        "model_type": "foundation",
        "framework": "PyTorchRNN",
        "name": model_class_name,
    })


# ══════════════════════════════════════════════════════════════
# 4) Run all baseline models on every feature
# ══════════════════════════════════════════════════════════════
BASELINE_MODELS = [
    ("SimpleLSTM", SimpleLSTM),
    ("SimpleGRU", SimpleGRU),
    ("SimpleDLinear", SimpleDLinear),
    ("SimpleAutoformer", SimpleAutoformer),
]

comparison_rows = []  # will become the final comparison DataFrame
t_total = time.time()

for feature in features_list:
    t0 = time.time()
    print(f"\n=== Feature: {feature} ===")
    feature_dir = os.path.join("results", feature)
    os.makedirs(feature_dir, exist_ok=True)

    row = {"feature": feature}

    # ── Collect existing MMF metrics (from already-saved metrics CSVs) ──
    mmf_metrics = {}   # model_name → {mse, mae, rmse}
    mmf_csv_paths = {}  # model_name → csv path  (for cell discovery)
    for mf in glob.glob(os.path.join(feature_dir, "*_metrics.csv")):
        model_name = os.path.basename(mf).replace("_metrics.csv", "")
        # Skip our own baseline metrics from any previous run
        _our_models = {n for n, _ in BASELINE_MODELS}
        if model_name in _our_models:
            continue
        m = _compute_all_metrics_from_csv(mf, PREDICTION_LENGTH)
        if m is not None:
            mmf_metrics[model_name] = m
            mmf_csv_paths[model_name] = mf

    # ── Determine which cells to use ──
    # Use the cells from the *best* MMF model so the comparison is
    # apples-to-apples even if different models ran with different
    # MAX_CELLS settings.
    best_mmf = min(mmf_metrics, key=lambda k: mmf_metrics[k]["rmse"]) if mmf_metrics else None
    if best_mmf is not None:
        mmf_cells = _get_cells_from_csv(mmf_csv_paths[best_mmf])
        df_feat = df_raw[df_raw["unique_id"].isin(mmf_cells)][["unique_id", "ds", feature]]
        print(f"  [cells] matched {df_feat['unique_id'].nunique()} cells from best MMF model ({best_mmf})")
    else:
        df_feat = _subsample_cells(df_raw, all_unique_ids)[["unique_id", "ds", feature]]
        print(f"  [cells] no MMF metrics found — fallback to {df_feat['unique_id'].nunique()} cells (MAX_CELLS={MAX_CELLS})")
    df_feat = df_feat.rename(columns={feature: "y"})
    row["n_cells"] = int(df_feat["unique_id"].nunique())
    train_df, val_df = split_train_val(df_feat)

    if best_mmf is not None:
        row["mmf_best_model"] = best_mmf
        row["mmf_best_mse"]   = mmf_metrics[best_mmf]["mse"]
        row["mmf_best_mae"]   = mmf_metrics[best_mmf]["mae"]
        row["mmf_best_rmse"]  = mmf_metrics[best_mmf]["rmse"]
    else:
        row["mmf_best_model"] = None
        row["mmf_best_mse"] = row["mmf_best_mae"] = row["mmf_best_rmse"] = None

    # ── Run each baseline model ──
    for model_name, model_cls in BASELINE_MODELS:
        params = _make_baseline_params(model_name)
        model = model_cls(params)
        model.fit(train_df)  # no-op; training is per-series inside backtest

        res_df = model.backtest(
            full_df=pd.concat([train_df, val_df]),
            train_df=train_df,
            val_df=val_df,
        )

        # Save per-cell metrics (same format as MMF)
        out_path = os.path.join(feature_dir, f"{model_name}_metrics.csv")
        res_df.to_csv(out_path, index=False)

        # Recompute MSE/MAE/RMSE from the forecast & actual arrays
        rnn_m = _compute_all_metrics_from_csv(out_path, PREDICTION_LENGTH)
        if rnn_m:
            row[f"{model_name}_mse"]  = rnn_m["mse"]
            row[f"{model_name}_mae"]  = rnn_m["mae"]
            row[f"{model_name}_rmse"] = rnn_m["rmse"]
            print(f"  {model_name}: MSE={rnn_m['mse']:.4f}  MAE={rnn_m['mae']:.4f}  RMSE={rnn_m['rmse']:.4f}")
        else:
            row[f"{model_name}_mse"]  = float("nan")
            row[f"{model_name}_mae"]  = float("nan")
            row[f"{model_name}_rmse"] = float("nan")
            print(f"  {model_name}: no valid forecasts")

    comparison_rows.append(row)
    print(f"  [time] {time.time() - t0:.1f}s")


# ══════════════════════════════════════════════════════════════
# 5) Build comparison table with % improvement
# ══════════════════════════════════════════════════════════════
comp_df = pd.DataFrame(comparison_rows)

# Compute % improvement of MMF vs each baseline
# Positive = MMF is better;  Negative = baseline is better
_baseline_names = [n for n, _ in BASELINE_MODELS]
for rnn_name in _baseline_names:
    for metric in ("mse", "mae", "rmse"):
        rnn_col = f"{rnn_name}_{metric}"
        mmf_col = f"mmf_best_{metric}"
        pct_col = f"mmf_vs_{rnn_name}_%lower_{metric}"
        if rnn_col in comp_df.columns and mmf_col in comp_df.columns:
            comp_df[pct_col] = (
                (comp_df[rnn_col] - comp_df[mmf_col]) / comp_df[rnn_col] * 100
            ).round(2)

# Reorder columns for readability
base_cols = ["feature", "n_cells", "mmf_best_model", "mmf_best_mse", "mmf_best_mae", "mmf_best_rmse"]
rnn_cols = []
pct_cols = []
for rnn_name in _baseline_names:
    rnn_cols += [f"{rnn_name}_mse", f"{rnn_name}_mae", f"{rnn_name}_rmse"]
    pct_cols += [f"mmf_vs_{rnn_name}_%lower_mse", f"mmf_vs_{rnn_name}_%lower_mae", f"mmf_vs_{rnn_name}_%lower_rmse"]
col_order = [c for c in base_cols + rnn_cols + pct_cols if c in comp_df.columns]
comp_df = comp_df[col_order]

os.makedirs("baseline", exist_ok=True)
_shard_suffix = f"_shard{shard_idx}" if num_shards > 1 else ""
csv_path = f"baseline/baseline_comparison{_shard_suffix}.csv"
json_path = f"baseline/baseline_comparison{_shard_suffix}.json"
summary_path = f"baseline/baseline_comparison_summary{_shard_suffix}.txt"

comp_df.to_csv(csv_path, index=False)
comp_df.to_json(json_path, orient="records", indent=2)


# ══════════════════════════════════════════════════════════════
# 6) Print headline summary (like the slide)
# ══════════════════════════════════════════════════════════════
summary_lines = []
summary_lines.append("=" * 65)
summary_lines.append("  MMF vs Baselines — Comparison Summary")
summary_lines.append("=" * 65)

for rnn_name in _baseline_names:
    mse_col = f"mmf_vs_{rnn_name}_%lower_mse"
    mae_col = f"mmf_vs_{rnn_name}_%lower_mae"
    rmse_col = f"mmf_vs_{rnn_name}_%lower_rmse"

    if mse_col in comp_df.columns:
        avg_mse_pct  = comp_df[mse_col].mean()
        avg_mae_pct  = comp_df[mae_col].mean()
        avg_rmse_pct = comp_df[rmse_col].mean()

        headline = (
            f"  vs. {rnn_name}:  "
            f"{avg_mse_pct:+.1f}% MSE  |  "
            f"{avg_mae_pct:+.1f}% MAE  |  "
            f"{avg_rmse_pct:+.1f}% RMSE"
        )
        summary_lines.append(headline)
        # Sign guide
        summary_lines.append("    (positive = MMF is better, negative = baseline is better)")

summary_lines.append("")
summary_lines.append("Per-feature breakdown:")
summary_lines.append(comp_df.to_string(index=False))
summary_lines.append(f"\n[time] total: {time.time() - t_total:.1f}s")
summary_text = "\n".join(summary_lines)

with open(summary_path, "w") as f:
    f.write(summary_text)

print(f"\n{summary_text}")
print(f"\nSaved: {csv_path}, {json_path}, {summary_path}")
