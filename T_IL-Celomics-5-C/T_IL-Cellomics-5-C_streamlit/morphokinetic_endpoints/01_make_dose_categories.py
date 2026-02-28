#!/usr/bin/env python3
"""01  Continuous → DoseCategory (Neg / Pos / High)

Reads a table with a continuous normalised dose column and adds a categorical
column using either fixed thresholds or data-driven quantiles.

CLI
---
    python 01_make_dose_categories.py \
        --input data.csv \
        --dose-col METR_Norm \
        --method fixed \
        --low-cut -0.524 --high-cut 0.524 \
        --out-col METR_Category \
        --outdir ./out
"""
import argparse, os, sys
import numpy as np, pandas as pd


def assign_dose_categories(
    df, dose_col, method="fixed",
    low_cut=-0.524, high_cut=0.524,
    low_q=1/3, high_q=2/3,
    out_col=None,
):
    if dose_col not in df.columns:
        raise KeyError(f"'{dose_col}' not in columns: {list(df.columns)}")
    out_col = out_col or f"{dose_col.replace('_Norm','')}_Category"
    df = df.copy()
    vals = df[dose_col].dropna()
    if len(vals) == 0:
        df[out_col] = np.nan
        return df
    if method == "quantile":
        low_cut = float(vals.quantile(low_q))
        high_cut = float(vals.quantile(high_q))
        print(f"  Quantile thresholds: low={low_cut:.4f}, high={high_cut:.4f}")
    elif method != "fixed":
        raise ValueError(f"Unknown method '{method}'")
    print(f"  Cutoffs: <= {low_cut:.4f} → Neg | > {high_cut:.4f} → High | else → Pos")
    df[out_col] = np.where(
        df[dose_col].isna(), np.nan,
        np.where(df[dose_col] <= low_cut, "Neg",
                 np.where(df[dose_col] > high_cut, "High", "Pos")))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--dose-col", required=True)
    ap.add_argument("--method", default="fixed", choices=["fixed","quantile"])
    ap.add_argument("--low-cut", type=float, default=-0.524)
    ap.add_argument("--high-cut", type=float, default=0.524)
    ap.add_argument("--out-col", default=None)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_parquet(a.input) if a.input.endswith(".parquet") else pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    df = assign_dose_categories(df, a.dose_col, a.method, a.low_cut, a.high_cut, out_col=a.out_col)
    out_col = a.out_col or f"{a.dose_col.replace('_Norm','')}_Category"
    print(f"\nCounts ({out_col}):\n{df[out_col].value_counts(dropna=False)}")
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "table_with_dose_categories.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
