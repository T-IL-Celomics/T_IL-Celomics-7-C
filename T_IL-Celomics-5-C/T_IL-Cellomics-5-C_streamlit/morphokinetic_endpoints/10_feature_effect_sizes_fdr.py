#!/usr/bin/env python3
"""10  Feature effect sizes + MWU + FDR vs a reference group

For each numeric feature, computes Mann-Whitney U vs a reference group,
Cohen's d effect size, and applies Benjamini-Hochberg FDR correction.

CLI
---
    python 10_feature_effect_sizes_fdr.py \
        --input ./out/table_with_clusters_and_pcs.csv \
        --group-col DoseCategory --ref-group Neg --outdir ./out
"""
import argparse, os, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)

EXCLUDE = {"Experiment","Parent","Time","TimeIndex","Cluster","PC1","PC2",
           "Condition","DoseCategory","DoseCombo","Treatment","Treatments",
           "METR_Category","GABY_Category","METR_Norm","GABY_Norm",
           "Dose","Groups","Unnamed: 0","ID","x_Pos","y_Pos","dt","ds",
           "unique_id","n_frames", "DoseLabel"}


def feature_effect_sizes(df, group_col, ref_group, exclude=None):
    exclude = exclude or EXCLUDE
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    ref = df[df[group_col] == ref_group]
    others = sorted([g for g in df[group_col].dropna().unique() if g != ref_group])
    print(f"  Reference: {ref_group} (n={len(ref)})")
    print(f"  Comparisons: {others}")
    print(f"  Features: {len(feats)}")

    records = []
    for g in others:
        grp = df[df[group_col] == g]
        pvals = []
        for feat in feats:
            a = ref[feat].dropna().values
            b = grp[feat].dropna().values
            if len(a) < 2 or len(b) < 2:
                records.append({"group": g, "feature": feat, "U": np.nan, "p": np.nan, "cohen_d": np.nan})
                pvals.append(np.nan)
                continue
            U, p = mannwhitneyu(a, b, alternative="two-sided")
            pooled_std = np.sqrt((a.var() + b.var()) / 2)
            d = (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0
            records.append({"group": g, "feature": feat, "U": U, "p": p, "cohen_d": d})
            pvals.append(p)

        # FDR
        valid = [i for i, v in enumerate(pvals) if not np.isnan(v)]
        if valid:
            raw = np.array([pvals[i] for i in valid])
            _, fdr, _, _ = multipletests(raw, method="fdr_bh")
            j = 0
            for i in valid:
                records[-(len(pvals)-i)]["q_fdr"] = fdr[j]
                j += 1

    result = pd.DataFrame(records)
    if "q_fdr" not in result.columns:
        result["q_fdr"] = np.nan
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--ref-group", required=True)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    result = feature_effect_sizes(df, a.group_col, a.ref_group)

    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, "feature_effect_sizes.csv")
    result.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # volcano plot per group
    groups = result["group"].dropna().unique()
    n_groups = len(groups)
    if n_groups == 0:
        return
    fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5), squeeze=False)
    for i, g in enumerate(groups):
        ax = axes[0, i]
        sub = result[result["group"]==g].dropna(subset=["q_fdr","cohen_d"])
        if len(sub) == 0:
            continue
        neg_log_q = -np.log10(sub["q_fdr"].clip(lower=1e-300))
        colors = ["red" if q < 0.05 else "grey" for q in sub["q_fdr"]]
        ax.scatter(sub["cohen_d"], neg_log_q, c=colors, s=15, alpha=0.7)
        ax.axhline(-np.log10(0.05), color="blue", ls="--", lw=0.5)
        ax.axvline(0, color="grey", ls="--", lw=0.5)
        ax.set_xlabel("Cohen's d")
        ax.set_ylabel("-log10(q FDR)")
        ax.set_title(f"{g} vs {a.ref_group}")
    fig.tight_layout()
    out_png = os.path.join(a.outdir, "volcano_effect_sizes.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # top 20 per group
    for g in groups:
        sub = result[result["group"]==g].sort_values("q_fdr").head(20)
        print(f"\n  Top 20 features for {g}:")
        print(sub[["feature","cohen_d","p","q_fdr"]].to_string(index=False))


if __name__ == "__main__":
    main()
