#!/usr/bin/env python3
"""03  Track-level summary (aggregate per-timepoint → per-cell)

Computes mean, std, min, max, first, last, and slope of each numeric feature
per (Experiment, Parent) track.

CLI
---
    python 03_track_level_summary.py \
        --input data.csv \
        --exp-col Experiment --parent-col Parent --time-col Time \
        --outdir ./out
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import linregress


def track_level_summary(df, exp_col, parent_col, time_col, extra_keep=None):
    meta = {exp_col, parent_col, time_col}
    if extra_keep:
        meta.update(extra_keep)
    numeric = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Numeric features: {len(numeric)}")

    grouped = df.sort_values(time_col).groupby([exp_col, parent_col])

    agg = {}
    for feat in numeric:
        agg[(feat, "mean")] = (feat, "mean")
        agg[(feat, "std")]  = (feat, "std")
        agg[(feat, "min")]  = (feat, "min")
        agg[(feat, "max")]  = (feat, "max")

    summary = grouped.agg(**{f"{f}_{s}": pd.NamedAgg(column=f, aggfunc=s) for (f,s) in agg.values()})
    summary = summary.reset_index()

    # first/last/slope per feature
    first_last = grouped.agg(
        **{f"{f}_first": pd.NamedAgg(column=f, aggfunc="first") for f in numeric},
        **{f"{f}_last":  pd.NamedAgg(column=f, aggfunc="last")  for f in numeric},
        n_frames=pd.NamedAgg(column=time_col, aggfunc="count"),
    ).reset_index()

    summary = summary.merge(first_last, on=[exp_col, parent_col])

    # carry forward non-varying categorical columns
    if extra_keep:
        cats = df.groupby([exp_col, parent_col])[list(extra_keep - {exp_col, parent_col})].first().reset_index()
        summary = summary.merge(cats, on=[exp_col, parent_col], how="left")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--exp-col", default="Experiment")
    ap.add_argument("--parent-col", default="Parent")
    ap.add_argument("--time-col", default="Time")
    ap.add_argument("--keep-cols", nargs="*", default=[])
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_parquet(a.input) if a.input.endswith(".parquet") else pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    extra = set(a.keep_cols) if a.keep_cols else set()
    summary = track_level_summary(df, a.exp_col, a.parent_col, a.time_col, extra or None)
    print(f"Track-level summary: {summary.shape}")
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "track_level_summary.csv")
    summary.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
