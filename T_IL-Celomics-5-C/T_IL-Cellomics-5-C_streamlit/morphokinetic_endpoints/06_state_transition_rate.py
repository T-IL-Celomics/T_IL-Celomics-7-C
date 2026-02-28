#!/usr/bin/env python3
"""06  State-transition rate per track

For time-resolved data with a Cluster column, compute the fraction of
consecutive timepoints where the cluster label changes within each track.

CLI
---
    python 06_state_transition_rate.py \
        --input data_with_clusters.csv \
        --exp-col Experiment --parent-col Parent --time-col Time \
        --cluster-col Cluster --outdir ./out
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def state_transition_rate(df, exp_col, parent_col, time_col, cluster_col):
    df = df.sort_values([exp_col, parent_col, time_col])
    records = []
    for (e, p), grp in df.groupby([exp_col, parent_col]):
        clusters = grp[cluster_col].values
        n = len(clusters)
        if n < 2:
            records.append({"Experiment": e, "Parent": p, "n_frames": n, "n_transitions": 0, "transition_rate": 0.0})
            continue
        transitions = np.sum(clusters[1:] != clusters[:-1])
        records.append({
            "Experiment": e, "Parent": p,
            "n_frames": n,
            "n_transitions": int(transitions),
            "transition_rate": float(transitions / (n - 1)),
        })
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--exp-col", default="Experiment")
    ap.add_argument("--parent-col", default="Parent")
    ap.add_argument("--time-col", default="Time")
    ap.add_argument("--cluster-col", default="Cluster")
    ap.add_argument("--group-col", default=None, help="Optional group column for coloured histograms")
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    tr = state_transition_rate(df, a.exp_col, a.parent_col, a.time_col, a.cluster_col)
    print(f"  Transition rates computed for {len(tr)} tracks")
    print(f"  Mean rate: {tr['transition_rate'].mean():.4f}")

    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, "transition_rates.csv")
    tr.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    if a.group_col and a.group_col in df.columns:
        grp_map = df.groupby([a.exp_col, a.parent_col])[a.group_col].first().reset_index()
        tr = tr.merge(grp_map, left_on=["Experiment", "Parent"],
                       right_on=[a.exp_col, a.parent_col], how="left")
        groups = sorted(tr[a.group_col].dropna().unique())
        for g in groups:
            ax.hist(tr.loc[tr[a.group_col]==g, "transition_rate"],
                    bins=30, alpha=0.5, label=str(g))
        ax.legend()
    else:
        ax.hist(tr["transition_rate"], bins=30, color="steelblue", edgecolor="black")
    ax.set_xlabel("Transition rate")
    ax.set_ylabel("Count")
    ax.set_title("State-transition rate distribution")
    fig.tight_layout()
    out_png = os.path.join(a.outdir, "transition_rate_hist.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
