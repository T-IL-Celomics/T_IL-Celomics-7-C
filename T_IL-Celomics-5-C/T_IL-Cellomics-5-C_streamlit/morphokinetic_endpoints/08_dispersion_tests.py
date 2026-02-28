#!/usr/bin/env python3
"""08  Dispersion / heterogeneity tests

Levene's test (or Brown-Forsythe) comparing within-group variance of a
chosen value column across groups.  Also reports per-group variance and IQR.

CLI
---
    python 08_dispersion_tests.py \
        --input ./out/table_with_clusters_and_pcs.csv \
        --group-col DoseCategory --value-col PC1 --outdir ./out
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import levene


def dispersion_tests(df, group_col, value_col):
    groups = sorted(df[group_col].dropna().unique())
    arrays = [df.loc[df[group_col]==g, value_col].dropna().values for g in groups]
    arrays = [a for a in arrays if len(a) >= 2]
    if len(arrays) < 2:
        print("  Not enough groups with ≥2 observations.")
        return None, None

    stat, p = levene(*arrays, center="median")  # Brown-Forsythe variant
    print(f"  Levene (Brown-Forsythe) stat={stat:.4f}, p={p:.4e}")

    per_group = []
    for g in groups:
        vals = df.loc[df[group_col]==g, value_col].dropna()
        per_group.append({
            "Group": g, "n": len(vals),
            "variance": vals.var(), "std": vals.std(),
            "IQR": vals.quantile(0.75) - vals.quantile(0.25),
            "median": vals.median(), "mean": vals.mean(),
        })
    pg = pd.DataFrame(per_group)
    print(f"\n  Per-group dispersion:\n{pg.to_string(index=False)}")

    stats = {"levene_stat": stat, "levene_p": p, "n_groups": len(groups)}
    return pg, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--value-col", default="PC1")
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    pg, stats = dispersion_tests(df, a.group_col, a.value_col)
    if pg is not None:
        os.makedirs(a.outdir, exist_ok=True)
        pg.to_csv(os.path.join(a.outdir, "dispersion_per_group.csv"), index=False)
        pd.DataFrame([stats]).to_csv(os.path.join(a.outdir, "dispersion_stats.csv"), index=False)
        print(f"\n  Saved to {a.outdir}/")


if __name__ == "__main__":
    main()
