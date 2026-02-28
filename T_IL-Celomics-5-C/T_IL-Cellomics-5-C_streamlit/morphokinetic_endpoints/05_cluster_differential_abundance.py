#!/usr/bin/env python3
"""05  Differential abundance of clusters across groups

Chi-square test + Cramer's V on the Cluster × Group contingency table.
Optional permutation test blocked by Experiment.

CLI
---
    python 05_cluster_differential_abundance.py \
        --input ./out/table_with_clusters_and_pcs.csv \
        --group-col DoseCategory --cluster-col Cluster \
        --exp-col Experiment --n-perm 500 --outdir ./out
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(ct):
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0


def differential_abundance(df, group_col, cluster_col, exp_col=None, n_perm=500):
    ct = pd.crosstab(df[group_col], df[cluster_col])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = cramers_v(ct)
    print(f"  Contingency table:\n{ct}\n")
    print(f"  Chi2={chi2:.2f}  p={p:.2e}  dof={dof}  Cramer's V={v:.4f}")

    perm_p = np.nan
    if exp_col and exp_col in df.columns and n_perm > 0:
        obs_chi2 = chi2
        count = 0
        rng = np.random.RandomState(42)
        groups = df[group_col].values.copy()
        exps = df[exp_col].values
        unique_exps = np.unique(exps)
        for _ in range(n_perm):
            perm = groups.copy()
            for e in unique_exps:
                mask = exps == e
                perm[mask] = rng.permutation(perm[mask])
            ct_p = pd.crosstab(pd.Series(perm), df[cluster_col])
            chi2_p, _, _, _ = chi2_contingency(ct_p)
            if chi2_p >= obs_chi2:
                count += 1
        perm_p = (count + 1) / (n_perm + 1)
        print(f"  Permutation p (blocked by {exp_col}, {n_perm} perms): {perm_p:.4f}")

    result = {"chi2": chi2, "p_chi2": p, "dof": dof, "cramers_v": v, "perm_p": perm_p}
    return ct, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--cluster-col", default="Cluster")
    ap.add_argument("--exp-col", default=None)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    ct, stats = differential_abundance(df, a.group_col, a.cluster_col, a.exp_col, a.n_perm)
    os.makedirs(a.outdir, exist_ok=True)
    ct.to_csv(os.path.join(a.outdir, "cluster_contingency.csv"))
    pd.DataFrame([stats]).to_csv(os.path.join(a.outdir, "cluster_diff_abundance_stats.csv"), index=False)
    print(f"  Saved to {a.outdir}/")


if __name__ == "__main__":
    main()
