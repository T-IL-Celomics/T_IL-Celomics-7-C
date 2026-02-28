#!/usr/bin/env python3
"""09  Mean PC shift + experiment-block bootstrap CI

Computes per-group means of PC1 and PC2, then generates bootstrap 95% CIs
by resampling experiments (block bootstrap).

CLI
---
    python 09_mean_pc_shift_bootstrap.py \
        --input ./out/table_with_clusters_and_pcs.csv \
        --group-col DoseCategory --exp-col Experiment \
        --pc1 PC1 --pc2 PC2 --n-boot 2000 --outdir ./out
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def block_bootstrap_mean(df, value_col, group_col, exp_col, n_boot=2000, ci=0.95):
    rng = np.random.RandomState(42)
    groups = sorted(df[group_col].dropna().unique())
    alpha = (1 - ci) / 2
    records = []
    for g in groups:
        sub = df[df[group_col] == g]
        exps = sub[exp_col].unique()
        n_exp = len(exps)
        obs_mean = sub[value_col].mean()
        boot_means = []
        for _ in range(n_boot):
            sampled_exps = rng.choice(exps, size=n_exp, replace=True)
            vals = pd.concat([sub[sub[exp_col]==e][value_col] for e in sampled_exps])
            boot_means.append(vals.mean())
        boot_means = np.array(boot_means)
        lo, hi = np.quantile(boot_means, [alpha, 1-alpha])
        records.append({
            "Group": g, "mean": obs_mean, "ci_lo": lo, "ci_hi": hi,
            "n_cells": len(sub), "n_experiments": n_exp,
        })
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--exp-col", default="Experiment")
    ap.add_argument("--pc1", default="PC1")
    ap.add_argument("--pc2", default="PC2")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    os.makedirs(a.outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    all_results = []
    for i, pc in enumerate([a.pc1, a.pc2]):
        if pc not in df.columns:
            print(f"  {pc} not found, skipping"); continue
        res = block_bootstrap_mean(df, pc, a.group_col, a.exp_col, a.n_boot)
        res["PC"] = pc
        all_results.append(res)
        print(f"\n  {pc} bootstrap results:\n{res.to_string(index=False)}")

        ax = axes[i]
        x = range(len(res))
        ax.errorbar(x, res["mean"], yerr=[res["mean"]-res["ci_lo"], res["ci_hi"]-res["mean"]],
                     fmt="o", capsize=5, capthick=1.5, markersize=6)
        ax.set_xticks(list(x)); ax.set_xticklabels(res["Group"], rotation=30, ha="right")
        ax.set_ylabel(f"Mean {pc}")
        ax.set_title(f"{pc} — Group means ± 95% CI")
        ax.axhline(0, color="grey", ls="--", lw=0.5)

    fig.tight_layout()
    out_png = os.path.join(a.outdir, "mean_pc_shift_bootstrap.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_png}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(a.outdir, "mean_pc_shift_bootstrap.csv"), index=False)


if __name__ == "__main__":
    main()
