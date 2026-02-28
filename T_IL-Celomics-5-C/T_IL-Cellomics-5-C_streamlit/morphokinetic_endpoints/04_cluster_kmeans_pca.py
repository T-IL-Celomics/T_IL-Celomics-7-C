#!/usr/bin/env python3
"""04  KMeans clustering + PCA visualisation

Standardises numeric features, fits KMeans(k), projects onto 2 PCs, saves
table_with_clusters_and_pcs.csv and pca_kmeans.png.

CLI
---
    python 04_cluster_kmeans_pca.py \
        --input ./out/track_level_summary.csv \
        --k 3 --outdir ./out
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


EXCLUDE = {"Experiment","Parent","Time","TimeIndex","Cluster","PC1","PC2",
           "Condition","DoseCategory","DoseCombo","Treatment","Treatments",
           "METR_Category","GABY_Category","METR_Norm","GABY_Norm",
           "Dose","Groups","Unnamed: 0","ID","x_Pos","y_Pos","dt","ds",
           "unique_id","n_frames"}


def cluster_and_pca(df, k, outdir, exclude=None):
    exclude = exclude or EXCLUDE
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Features used: {len(feats)}")
    X = df[feats].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    df = df.copy()
    df["Cluster"] = km.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]
    ev = pca.explained_variance_ratio_

    # save
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "table_with_clusters_and_pcs.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10", k)
    for ci in range(k):
        m = df["Cluster"] == ci
        ax.scatter(df.loc[m, "PC1"], df.loc[m, "PC2"],
                   c=[cmap(ci)], label=f"Cluster {ci}", s=8, alpha=0.5)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title(f"KMeans k={k} on PCA")
    ax.legend(markerscale=3)
    fig.tight_layout()
    out_png = os.path.join(outdir, "pca_kmeans.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    cluster_and_pca(df, a.k, a.outdir)


if __name__ == "__main__":
    main()
