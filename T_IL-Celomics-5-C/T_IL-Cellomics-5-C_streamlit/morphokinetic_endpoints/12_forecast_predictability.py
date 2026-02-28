#!/usr/bin/env python3
"""12  Forecast predictability (Ridge regression, lag features)

For each track, trains a simple Ridge regressor to predict the next value
of a chosen feature from the previous *lag* values.  Reports per-track MAE.

CLI
---
    python 12_forecast_predictability.py \
        --input data.csv \
        --exp-col Experiment --parent-col Parent --time-col Time \
        --feature Area --lag 3 --alpha 1.0 --outdir ./out
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def forecast_track(t, y, lag=3, alpha=1.0):
    n = len(y)
    if n < lag + 2:
        return np.nan
    X, Y = [], []
    for i in range(lag, n):
        X.append(y[i-lag:i])
        Y.append(y[i])
    X, Y = np.array(X), np.array(Y)
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    preds = model.predict(X)
    return float(np.mean(np.abs(preds - Y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--exp-col", default="Experiment")
    ap.add_argument("--parent-col", default="Parent")
    ap.add_argument("--time-col", default="Time")
    ap.add_argument("--feature", required=True)
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")

    records = []
    for (e, p), grp in df.groupby([a.exp_col, a.parent_col]):
        grp = grp.sort_values(a.time_col)
        y = grp[a.feature].dropna().values.astype(float)
        mae = forecast_track(grp[a.time_col].values.astype(float), y, a.lag, a.alpha)
        rec = {"Experiment": e, "Parent": p, "MAE": mae, "n_frames": len(y)}
        if a.group_col and a.group_col in grp.columns:
            rec["Group"] = grp[a.group_col].iloc[0]
        records.append(rec)

    result = pd.DataFrame(records)
    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, "forecast_mae.csv")
    result.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"\nMAE summary:\n{result['MAE'].describe()}")

    # boxplot by group if available
    fig, ax = plt.subplots(figsize=(7, 4))
    if "Group" in result.columns:
        groups = sorted(result["Group"].dropna().unique())
        data = [result.loc[result["Group"]==g, "MAE"].dropna().values for g in groups]
        ax.boxplot(data, labels=groups, showfliers=False)
        ax.set_xlabel("Group")
    else:
        ax.boxplot(result["MAE"].dropna().values, showfliers=False)
    ax.set_ylabel("Forecast MAE")
    ax.set_title(f"Forecast MAE ({a.feature}, lag={a.lag})")
    fig.tight_layout()
    out_png = os.path.join(a.outdir, "forecast_mae_boxplot.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
