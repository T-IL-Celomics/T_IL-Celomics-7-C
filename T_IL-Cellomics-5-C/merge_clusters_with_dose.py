import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs("dose_dependancy", exist_ok=True)

# ================== CONFIG ==================
# if your clustering file has a different name (e.g.
# "embedding_fitting_Merged_Clusters_PCA.csv"), change this:
CLUSTER_FILE = "clustering/Merged_Clusters_PCA.csv"

DOSE_FILE    = "cell_data/dose_dependency_summary_all_wells.csv"

OUT_MERGED_CELLS   = "dose_dependancy/cells_with_clusters_and_dose.csv"
OUT_CONTINGENCY    = "dose_dependancy/cluster_vs_dose_counts.csv"
OUT_HEATMAP_PNG    = "dose_dependancy/cluster_vs_dose_heatmap.png"
OUT_PCA_PLOT_PNG   = "dose_dependancy/pca_clusters_basic.png"
# ============================================


def main():
    # ---------- 1) load clustering results ----------
    clusters = pd.read_csv(CLUSTER_FILE)

    # we assume these columns exist in your cluster file
    # if any are missing, we stop with a clear error
    needed = {"Experiment", "Parent", "Cluster", "PC1", "PC2"}
    missing = needed - set(clusters.columns)
    if missing:
        raise ValueError(f"Missing columns in {CLUSTER_FILE}: {missing}")

    # one row per cell: experiment, parent, cluster, PCA coords
    cell_level = (
        clusters[list(needed)]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"[INFO] cell-level rows from cluster file: {len(cell_level)}")

    # ---------- 2) load dose summary ----------
    dose = pd.read_csv(DOSE_FILE)

    # find columns like "Cha1_Category", "Cha2_Category", etc.
    category_cols = [c for c in dose.columns if c.endswith("_Category")]
    dose_label_col = category_cols[0] if category_cols else None

    # base columns to keep
    cols_to_keep = ["Experiment", "Parent"]
    if "n_frames" in dose.columns:
        cols_to_keep.append("n_frames")
    if dose_label_col:
        cols_to_keep.append(dose_label_col)

    dose_small = dose[cols_to_keep].copy()

    if dose_label_col:
        dose_small = dose_small.rename(columns={dose_label_col: "DoseLabel"})
        print(f"[INFO] using '{dose_label_col}' as DoseLabel")
    else:
        print("[WARN] no *_Category column found; will use Experiment as dose proxy")

    # ---------- 3) merge per-cell clusters with dose ----------
    merged = cell_level.merge(
        dose_small,
        on=["Experiment", "Parent"],
        how="left",
        validate="m:1"  # each (Experiment, Parent) should map to at most 1 dose row
    )

    print(f"[INFO] merged cells shape: {merged.shape}")
    print(merged.head())

    # save merged per-cell table
    merged.to_csv(OUT_MERGED_CELLS, index=False)
    print(f"[SAVE] {OUT_MERGED_CELLS}")

    # ---------- 4) build cluster vs dose/experiment table ----------
    if "DoseLabel" in merged.columns:
        index_col = "DoseLabel"
    else:
        index_col = "Experiment"  # fallback

    contingency = merged.pivot_table(
        index=index_col,
        columns="Cluster",
        values="Parent",
        aggfunc="nunique",
        fill_value=0,
    )

    print("\n[INFO] Cluster vs dose/experiment counts:")
    print(contingency)

    contingency.to_csv(OUT_CONTINGENCY)
    print(f"[SAVE] {OUT_CONTINGENCY}")

    # ---------- 5) heatmap: cluster vs dose ----------
    plt.figure(figsize=(6, 4))
    plt.imshow(contingency, aspect="auto")
    plt.colorbar(label="Number of cells")

    plt.xticks(range(len(contingency.columns)), contingency.columns)
    plt.yticks(range(len(contingency.index)), contingency.index, fontsize=6)

    plt.xlabel("Cluster")
    plt.ylabel(index_col)
    plt.title("Cell counts per cluster and dose/experiment")
    plt.tight_layout()
    plt.savefig(OUT_HEATMAP_PNG, dpi=300)
    plt.close()
    print(f"[SAVE] {OUT_HEATMAP_PNG}")

    # ---------- 6) basic PCA scatter colored by cluster ----------
    # (doesn't use dose directly, but is nice for slide)
    sub = merged.dropna(subset=["PC1", "PC2", "Cluster"])

    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(
        sub["PC1"],
        sub["PC2"],
        c=sub["Cluster"],
        s=10,
        alpha=0.7,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of embeddings colored by cluster")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(OUT_PCA_PLOT_PNG, dpi=300)
    plt.close()
    print(f"[SAVE] {OUT_PCA_PLOT_PNG}")


if __name__ == "__main__":
    main()
