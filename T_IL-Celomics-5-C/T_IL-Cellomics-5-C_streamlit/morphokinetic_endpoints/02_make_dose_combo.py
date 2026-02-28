#!/usr/bin/env python3
"""02  DoseCombo label  (e.g. METR:High|GABY:Neg)

Combines per-protein dose category columns into a single pipe-delimited label.

CLI
---
    python 02_make_dose_combo.py \
        --input ./out/table_with_dose_categories.csv \
        --cat-cols METR_Category GABY_Category \
        --outdir ./out
"""
import argparse, os
import pandas as pd, numpy as np


def make_dose_combo(df, cat_cols, combo_col="DoseCombo"):
    df = df.copy()
    missing = [c for c in cat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    def _row(r):
        parts = []
        for c in cat_cols:
            name = c.replace("_Category", "")
            val = r[c]
            if pd.notna(val) and str(val).strip():
                parts.append(f"{name}:{val}")
        return "|".join(parts) if parts else np.nan
    df[combo_col] = df.apply(_row, axis=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cat-cols", nargs="+", required=True)
    ap.add_argument("--combo-col", default="DoseCombo")
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    df = make_dose_combo(df, a.cat_cols, a.combo_col)
    print(f"\nDoseCombo counts:\n{df[a.combo_col].value_counts(dropna=False).head(20)}")
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "table_with_dose_combo.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
