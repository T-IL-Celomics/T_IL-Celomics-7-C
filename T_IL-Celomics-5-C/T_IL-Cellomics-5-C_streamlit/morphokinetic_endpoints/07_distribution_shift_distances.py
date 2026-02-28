#!/usr/bin/env python3
"""07  Distribution shift distances (Wasserstein, KS, Energy) between groups

For each pair of groups, computes 1-D distributional distances on a chosen
value column (e.g. PC1).

CLI
---
    python 07_distribution_shift_distances.py \
        --input ./out/table_with_clusters_and_pcs.csv \
        --group-col DoseCategory --value-col PC1 --outdir ./out
"""
import argparse, os, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp, energy_distance


def distribution_distances(df, group_col, value_col):
    groups = sorted(df[group_col].dropna().unique())
    records = []
    for g1, g2 in itertools.combinations(groups, 2):
        a = df.loc[df[group_col]==g1, value_col].dropna().values
        b = df.loc[df[group_col]==g2, value_col].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        w = wasserstein_distance(a, b)
        ks_stat, ks_p = ks_2samp(a, b)
        try:
            ed = energy_distance(a, b)
        except Exception:
            ed = np.nan
        records.append({
            "Group_A": g1, "Group_B": g2,
            "Wasserstein": w, "KS_stat": ks_stat, "KS_p": ks_p, "Energy": ed,
            "n_A": len(a), "n_B": len(b),
        })
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--value-col", default="PC1")
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    dist = distribution_distances(df, a.group_col, a.value_col)
    print(dist.to_string(index=False))

    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, "distribution_shift_distances.csv")
    dist.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # heatmap
    groups = sorted(set(dist["Group_A"]) | set(dist["Group_B"]))
    n = len(groups)
    mat = np.zeros((n, n))
    idx = {g: i for i, g in enumerate(groups)}
    for _, r in dist.iterrows():
        i, j = idx[r["Group_A"]], idx[r["Group_B"]]
        mat[i, j] = mat[j, i] = r["Wasserstein"]

    fig, ax = plt.subplots(figsize=(max(5, n*0.9), max(4, n*0.8)))
    im = ax.imshow(mat, cmap="YlOrRd")
    ax.set_xticks(range(n)); ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(groups, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Wasserstein distance")
    ax.set_title(f"Pairwise Wasserstein on {a.value_col}")
    fig.tight_layout()
    out_png = os.path.join(a.outdir, "distribution_shift_heatmap.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
