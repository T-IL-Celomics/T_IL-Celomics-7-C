import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import silhouette_score
import os
import math
# Ensure output directory exists
os.makedirs("clustering", exist_ok=True)

# === 1. Load original experiment data ===
_merged_csv = os.environ.get("PIPELINE_MERGED_CSV", "cell_data/raw_all_cells.csv")
if _merged_csv.lower().endswith((".xlsx", ".xls")):
    df_data = pd.read_excel(_merged_csv)
else:
    try:
        df_data = pd.read_csv(_merged_csv)
    except UnicodeDecodeError:
        print(f"[warn] UTF-8 decode failed for {_merged_csv}, retrying with latin-1")
        df_data = pd.read_csv(_merged_csv, encoding="latin-1")

# === 1b. Load dose information ===
_dose_csv = os.environ.get("PIPELINE_DOSE_CSV", "")
if not _dose_csv:
    # Auto-detect: check cell_data/ first, then project root
    for _candidate in ["cell_data/dose_dependency_summary_all_wells.csv",
                       "dose_dependency_summary_all_wells.csv"]:
        if os.path.isfile(_candidate):
            _dose_csv = _candidate
            break
    else:
        _dose_csv = "cell_data/dose_dependency_summary_all_wells.csv"  # fallback (will trigger FileNotFoundError)
_control_channel = os.environ.get("PIPELINE_CONTROL_CHANNEL", "").strip()


def _parse_channels(experiment_id: str) -> list:
    """Extract the list of 4-char channel codes from an experiment ID.

    Format: <name5><date6><4chars><well3><celltype4><ch1><ch2>...NOCO<rest>
    Everything between position 22 and 'NOCO' is channel codes (4 chars each).
    """
    after = experiment_id[22:]
    noco = after.find("NOCO")
    if noco < 0:
        return []
    return [after[i:i + 4] for i in range(0, noco, 4)]


def _control_cha_col(experiment_id: str, control_name: str) -> str:
    """Return the Cha*_Category column name for the control channel,
    or '' if not found.  E.g. control_name='NNIR' → 'Cha2_Category'
    when NNIR is the 2nd channel in the experiment ID."""
    if not control_name:
        return ""
    channels = _parse_channels(experiment_id)
    for i, ch in enumerate(channels):
        if ch == control_name:
            return f"Cha{i + 1}_Category"
    return ""


def _make_dose_label(row, cols=None):
    """Build pipe-delimited DoseLabel using real channel names from the
    experiment ID (e.g. 'METR:Pos|GABY:Low') instead of generic Cha1/Cha2.

    The control channel is excluded so labels only show treatment channels.
    """
    if cols is None:
        cols = [c for c in row.index if c.endswith("_Category")]
    exp_id = str(row["Experiment"])
    channels = _parse_channels(exp_id)          # e.g. ['METR', 'NNIR']
    skip = _control_cha_col(exp_id, _control_channel)
    parts = []
    for c in cols:
        if c == skip:
            continue
        if pd.notna(row[c]):
            # Map Cha<n>_Category → real channel name
            idx = int(c.replace("Cha", "").replace("_Category", "")) - 1
            name = channels[idx] if idx < len(channels) else c.replace("_Category", "")
            parts.append(f"{name}:{row[c]}")
    return "|".join(parts)


try:
    dose_data = pd.read_csv(_dose_csv)
    # Find all dose label columns (Cha*_Category)
    category_cols = sorted([c for c in dose_data.columns if c.endswith("_Category")])
    
    if category_cols:
        dose_data = dose_data[["Experiment", "Parent"] + category_cols].copy()
        print(f"Loaded dose information with channels: {category_cols}")
        if _control_channel:
            print(f"Control channel to exclude from DoseLabel: {_control_channel}")

        dose_data["DoseLabel"] = dose_data.apply(
            lambda row: _make_dose_label(row, category_cols), axis=1
        )
    else:
        dose_data = None
        print("No dose label column found")
except FileNotFoundError:
    dose_data = None
    print("Dose file not found, skipping dose analysis")

# === 2. Load embedding JSON ===
with open(os.environ.get("PIPELINE_EMBEDDING_JSON", "embeddings/summary_table_Embedding.json"), "r") as f:
    data = json.load(f)

# === 3. Define treatment labels (from env or default) ===
_treatments_env = os.environ.get("PIPELINE_TREATMENTS", "CON0, BRCACON1, BRCACON2, BRCACON3, BRCACON4, BRCACON5")
treatments = [t.strip() for t in _treatments_env.split(",") if t.strip()]
print(f"Treatments: {treatments}")

# === 4. Prepare embedding records ===
excluded = {"Experiment", "Parent"}
feat_keys = [k for k in data[0].keys() if k not in excluded]
feat_keys = [k for k in feat_keys if k != "Unnamed: 0"]  # optional cleanup

records = []
for cell in data:
    embedding_vector = []
    for k in feat_keys:
        v = cell.get(k, [np.nan])
        embedding_vector.extend(v if isinstance(v, list) else [v])

    vec = np.array(embedding_vector, dtype=np.float32)
    if np.isfinite(vec).all():
        treatments_sorted = sorted(treatments, key=len, reverse=True)
        treatment = next((t for t in treatments_sorted if t in cell["Experiment"]), "Unknown")
        records.append({
            "Experiment": str(cell["Experiment"]),
            "Parent": str(cell["Parent"]),
            "embedding": vec,
            "Treatment": treatment
        })

# === 5. Stack embeddings into matrix ===
X = np.stack([r["embedding"] for r in records], axis=0)
print("X shape:", X.shape, "num cells:", len(records))


# === 6. Apply scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

_k_min = int(os.environ.get("PIPELINE_K_MIN", "2"))
_k_max = int(os.environ.get("PIPELINE_K_MAX", "10"))
_n_init = int(os.environ.get("PIPELINE_KMEANS_N_INIT", "10"))
_random_state = int(os.environ.get("PIPELINE_KMEANS_SEED", "42"))
_pca_components = int(os.environ.get("PIPELINE_PCA_COMPONENTS", "2"))
_num_best_k = int(os.environ.get("PIPELINE_NUM_BEST_K", "2"))
k_range = range(_k_min, _k_max + 1)   # usually start from k=2
sil_scores = {}

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=_random_state, n_init=_n_init)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels, metric="euclidean")
    sil_scores[k] = score
    print(f"k={k}, silhouette={score:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(list(k_range), [sil_scores[k] for k in k_range], marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs Number of Clusters")
plt.grid(True)
plt.tight_layout()
plt.savefig("clustering/silhouette_vs_k.png")
plt.show()

# Get best k values
sorted_k = sorted(sil_scores.items(), key=lambda x: x[1], reverse=True)
best_k_values = [sorted_k[i][0] for i in range(min(_num_best_k, len(sorted_k)))]
print(f"\nBest {_num_best_k} k values: {', '.join(f'{k} ({sil_scores[k]:.4f})' for k in best_k_values)}\n")

# === 7. PCA ===
pca = PCA(n_components=_pca_components)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_ * 100
print(f"Explained variance: PC1 = {explained_var[0]:.2f}%, PC2 = {explained_var[1]:.2f}%")

# === 8-14. Process both best k values ===
import math

for best_k in best_k_values:
    print(f"\n{'='*60}")
    print(f"Processing k={best_k}...")
    print(f"{'='*60}\n")
    
    # Clustering
    kmeans_final = KMeans(n_clusters=best_k, random_state=_random_state, n_init=_n_init)
    final_labels = kmeans_final.fit_predict(X_scaled)

    print("Silhouette score per treatment:")
    for treatment in np.unique([r["Treatment"] for r in records]):
        idx = np.array([r["Treatment"] == treatment for r in records])

        # need at least k+1 samples to compute silhouette
        if idx.sum() > best_k:
            score = silhouette_score(
                X_scaled[idx],
                final_labels[idx],
                metric="euclidean"
            )
            print(f"{treatment}: silhouette = {score:.3f}")
        else:
            print(f"{treatment}: not enough samples")

    # Build PCA dataframe
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["Experiment"] = [r["Experiment"] for r in records]
    pca_df["Parent"] = [r["Parent"] for r in records]
    pca_df["Treatment"] = [r["Treatment"] for r in records]
    pca_df["Cluster"] = final_labels

    # Merge with dose information if available
    if dose_data is not None:
        pca_df["Experiment"] = pca_df["Experiment"].astype(str)
        pca_df["Parent"] = pca_df["Parent"].astype(str)
        dose_data["Experiment"] = dose_data["Experiment"].astype(str)
        dose_data["Parent"] = dose_data["Parent"].astype(str)
        pca_df = pca_df.merge(dose_data, on=["Experiment", "Parent"], how="left")

    # Save cluster assignments
    pca_df.to_csv(f"clustering/cluster_assignments_k{best_k}.csv", index=False)
    print(f"Saved cluster_assignments_k{best_k}.csv")

    # Plot full PCA with clusters
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    plt.figure(figsize=(8, 6))
    for i in range(best_k):
        cluster_data = pca_df[pca_df["Cluster"] == i]
        plt.scatter(cluster_data["PC1"], cluster_data["PC2"], color=colors[i],
                    label=f"Cluster {i}", edgecolors='black', s=50, alpha=0.8)
    plt.xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    plt.ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    plt.title(f"PCA with KMeans Clusters (k={best_k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"clustering/pca_kmeans_k{best_k}_clusters.png")
    plt.show()

    # Plot by treatment
    treatments_present = [t for t in treatments if (pca_df["Treatment"] == t).any()]
    n_t = len(treatments_present)

    if n_t > 0:
        # pick a "nice" grid: ~square
        ncols = math.ceil(math.sqrt(n_t))
        nrows = math.ceil(n_t / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)  # flatten safely even if 1 subplot

        for i, treatment in enumerate(treatments_present):
            ax = axes[i]
            subset = pca_df[pca_df["Treatment"] == treatment]

            for j in range(best_k):
                cdata = subset[subset["Cluster"] == j]
                ax.scatter(
                    cdata["PC1"], cdata["PC2"],
                    color=colors[j],
                    edgecolors="black", linewidths=0.3,
                    s=40, alpha=0.7,
                    label=f"Cluster {j}" if i == 0 else None  # legend once
                )

            ax.set_title(treatment)
            ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
            ax.grid(True)

        # delete unused axes (if grid has extra slots)
        for j in range(n_t, len(axes)):
            fig.delaxes(axes[j])

        fig.legend(loc="upper right")
        plt.suptitle(f"PCA Clustering by Treatment (k={best_k})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"clustering/pca_kmeans_k{best_k}_by_treatment.png")
        plt.show()

    # ── Per-treatment dose analysis (only meaningful within each treatment) ──
    if dose_data is not None and n_t > 0:
        print(f"\n{'='*60}")
        print(f"Dose analysis per treatment (k={best_k})")
        print(f"{'='*60}\n")

        for treatment in treatments_present:
            treatment_data = pca_df[pca_df["Treatment"] == treatment].copy()
            treatment_doses = sorted(
                [d for d in treatment_data["DoseLabel"].unique() if pd.notna(d) and d != ""]
            )

            if len(treatment_doses) == 0:
                continue

            # --- PCA scatter per dose combination within this treatment ---
            n_td = len(treatment_doses)
            ncols_td = math.ceil(math.sqrt(n_td))
            nrows_td = math.ceil(n_td / ncols_td)

            fig, axes = plt.subplots(nrows_td, ncols_td,
                                     figsize=(6*ncols_td, 5*nrows_td),
                                     sharex=True, sharey=True)
            axes = np.array(axes).reshape(-1)

            for i, dose in enumerate(treatment_doses):
                ax = axes[i]
                subset = treatment_data[treatment_data["DoseLabel"] == dose]

                for j in range(best_k):
                    cdata = subset[subset["Cluster"] == j]
                    ax.scatter(
                        cdata["PC1"], cdata["PC2"],
                        color=colors[j],
                        edgecolors="black", linewidths=0.3,
                        s=40, alpha=0.7,
                        label=f"Cluster {j}" if i == 0 else None
                    )

                ax.set_title(dose)
                ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
                ax.grid(True)

            for j in range(n_td, len(axes)):
                fig.delaxes(axes[j])

            fig.legend(loc="upper right")
            plt.suptitle(f"{treatment} – PCA by Dose (k={best_k})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"clustering/pca_kmeans_k{best_k}_{treatment}_by_dose.png")
            plt.show()

            # --- Contingency table & heatmap for this treatment ---
            treatment_contingency = treatment_data.pivot_table(
                index="DoseLabel",
                columns="Cluster",
                values="Parent",
                aggfunc="nunique",
                fill_value=0,
            )

            treatment_contingency.to_csv(
                f"clustering/cluster_vs_dose_k{best_k}_{treatment}.csv"
            )
            print(f"{treatment} – Cluster vs Dose counts:")
            print(treatment_contingency)

            # Row-normalised heatmap (% of cells in each dose → cluster)
            row_pct = treatment_contingency.div(
                treatment_contingency.sum(axis=1), axis=0
            ) * 100

            fig, (ax_abs, ax_pct) = plt.subplots(
                1, 2, figsize=(5 + 5, max(3, 0.6 * len(treatment_contingency))),
            )

            # Absolute counts
            im1 = ax_abs.imshow(treatment_contingency.values, aspect="auto", cmap="YlOrRd")
            fig.colorbar(im1, ax=ax_abs, label="# cells")
            ax_abs.set_xticks(range(len(treatment_contingency.columns)))
            ax_abs.set_xticklabels([f"C{c}" for c in treatment_contingency.columns])
            ax_abs.set_yticks(range(len(treatment_contingency.index)))
            ax_abs.set_yticklabels(treatment_contingency.index, fontsize=8)
            ax_abs.set_xlabel("Cluster")
            ax_abs.set_title("Cell count")
            # Annotate cells
            for r in range(treatment_contingency.shape[0]):
                for c_idx in range(treatment_contingency.shape[1]):
                    ax_abs.text(c_idx, r, str(treatment_contingency.values[r, c_idx]),
                                ha="center", va="center", fontsize=8)

            # Percentage
            im2 = ax_pct.imshow(row_pct.values, aspect="auto", cmap="YlOrRd",
                                vmin=0, vmax=100)
            fig.colorbar(im2, ax=ax_pct, label="% of dose group")
            ax_pct.set_xticks(range(len(row_pct.columns)))
            ax_pct.set_xticklabels([f"C{c}" for c in row_pct.columns])
            ax_pct.set_yticks(range(len(row_pct.index)))
            ax_pct.set_yticklabels(row_pct.index, fontsize=8)
            ax_pct.set_xlabel("Cluster")
            ax_pct.set_title("Row %")
            for r in range(row_pct.shape[0]):
                for c_idx in range(row_pct.shape[1]):
                    ax_pct.text(c_idx, r, f"{row_pct.values[r, c_idx]:.0f}%",
                                ha="center", va="center", fontsize=8)

            plt.suptitle(f"{treatment} – Dose vs Cluster (k={best_k})", fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.savefig(
                f"clustering/cluster_vs_dose_heatmap_k{best_k}_{treatment}.png",
                dpi=300,
            )
            plt.close()
            print(f"Saved cluster_vs_dose_heatmap_k{best_k}_{treatment}.png\n")

            # --- Barplot: cell distribution (%) per dose ---
            cluster_cols = [c for c in treatment_contingency.columns]
            row_totals = treatment_contingency.sum(axis=1)
            pct = treatment_contingency.div(row_totals, axis=0) * 100

            x_labels = pct.index.tolist()
            x_pos = np.arange(len(x_labels))
            n_clusters = len(cluster_cols)
            bar_w = 0.8 / n_clusters
            bar_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan'][:n_clusters]

            fig_bar, ax_bar = plt.subplots(figsize=(max(8, len(x_labels)*1.8), 5))
            for ci, cc in enumerate(cluster_cols):
                offset = (ci - n_clusters/2 + 0.5) * bar_w
                ax_bar.bar(
                    x_pos + offset, pct[cc].values,
                    width=bar_w, label=f"Group {cc}",
                    color=bar_colors[ci]
                )

            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(x_labels, fontsize=9)
            ax_bar.set_ylabel("Cell Distribution (%)")
            ax_bar.set_title(f"Cell Distribution Per Dose – {treatment} (k={best_k})")
            ax_bar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=n_clusters, frameon=False)
            ax_bar.grid(axis='y', linestyle='-', alpha=0.3)
            ax_bar.set_axisbelow(True)
            plt.tight_layout()
            bar_path = f"clustering/cluster_barplot_by_dose_k{best_k}_{treatment}.png"
            fig_bar.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {bar_path}")

        # --- Barplot: cell distribution (%) per treatment (no dose) ---
        all_treatment_counts = {}
        for treatment in treatments_present:
            tdata = pca_df[pca_df["Treatment"] == treatment]
            counts_per_cluster = tdata.groupby("Cluster").size()
            all_treatment_counts[treatment] = counts_per_cluster

        if all_treatment_counts:
            df_treat = pd.DataFrame(all_treatment_counts).T.fillna(0).astype(float)
            df_treat = df_treat[sorted(df_treat.columns)]
            cluster_cols_t = list(df_treat.columns)
            row_totals_t = df_treat.sum(axis=1)
            pct_t = df_treat.div(row_totals_t, axis=0) * 100

            x_labels_t = pct_t.index.tolist()
            x_pos_t = np.arange(len(x_labels_t))
            n_cls = len(cluster_cols_t)
            bw = 0.8 / n_cls
            bar_colors_t = ['red', 'blue', 'green', 'orange', 'purple', 'cyan'][:n_cls]

            fig_t, ax_t = plt.subplots(figsize=(max(8, len(x_labels_t)*1.8), 5))
            for ci, cc in enumerate(cluster_cols_t):
                offset = (ci - n_cls/2 + 0.5) * bw
                ax_t.bar(
                    x_pos_t + offset, pct_t[cc].values,
                    width=bw, label=f"Group {cc}",
                    color=bar_colors_t[ci]
                )

            ax_t.set_xticks(x_pos_t)
            ax_t.set_xticklabels(x_labels_t, fontsize=9)
            ax_t.set_ylabel("Cell Distribution (%)")
            ax_t.set_title(f"Cell Distribution Per Treatment (k={best_k})")
            ax_t.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=n_cls, frameon=False)
            ax_t.grid(axis='y', linestyle='-', alpha=0.3)
            ax_t.set_axisbelow(True)
            plt.tight_layout()
            treat_bar_path = f"clustering/cluster_barplot_by_treatment_k{best_k}.png"
            fig_t.savefig(treat_bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {treat_bar_path}")

    # Merge with original full data
    # Convert merge keys to string for safety
    df_data_copy = df_data.copy()
    df_data_copy["Experiment"] = df_data_copy["Experiment"].astype(str)
    df_data_copy["Parent"] = df_data_copy["Parent"].astype(str)
    pca_df["Experiment"] = pca_df["Experiment"].astype(str)
    pca_df["Parent"] = pca_df["Parent"].astype(str)

    merged_df = pd.merge(
        df_data_copy,
        pca_df[['Experiment', 'Parent', 'PC1', 'PC2', 'Cluster', 'Treatment']],
        how='left',
        on=['Experiment', 'Parent']
    )

    # Save merged file
    merged_df.to_csv(f"clustering/Merged_Clusters_PCA_k{best_k}.csv", index=False)
    print(f"Saved as Merged_Clusters_PCA_k{best_k}.csv\n")

    # --- Mean Feature Values by Group (Normalized to Group 0) ---
    _morphological = [
        'Area', 'Ellip_Ax_B_X', 'Ellip_Ax_B_Y', 'Ellip_Ax_C_X', 'Ellip_Ax_C_Y',
        'EllipsoidAxisLengthB', 'EllipsoidAxisLengthC',
        'Ellipticity_oblate', 'Ellipticity_prolate', 'Sphericity', 'Eccentricity',
    ]
    _non_feature = {
        'Unnamed: 0', 'TimeIndex', 'x_Pos', 'y_Pos', 'Parent', 'dt', 'ID',
        'Experiment', 'PC1', 'PC2', 'Cluster', 'Treatment', 'unique_id', 'ds',
        'DoseLabel', 'Coll', 'Coll_CUBE',
    }
    _numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    _all_feats = [c for c in _numeric_cols if c not in _non_feature]
    _kinetic = [f for f in _all_feats if f not in _morphological]
    _morph_present = [f for f in _morphological if f in merged_df.columns]
    _kin_present = [f for f in _kinetic if f in merged_df.columns]
    _mid = len(_kin_present) // 2
    _feat_groups = [
        ('Kinetic Features (Part 1)', _kin_present[:_mid]),
        ('Kinetic Features (Part 2)', _kin_present[_mid:]),
        ('Morphological Features', _morph_present),
    ]
    _feat_groups = [(name, feats) for name, feats in _feat_groups if feats]

    if _feat_groups:
        _mdf = merged_df.dropna(subset=['Cluster']).copy()
        _mdf['Cluster'] = _mdf['Cluster'].astype(int)
        cluster_ids = sorted(_mdf['Cluster'].unique())
        means = _mdf.groupby('Cluster')[_all_feats].mean()
        if 0 in cluster_ids and len(cluster_ids) >= 2:
            # Use absolute values so ratio is always positive
            abs_means = means.abs()
            g0_abs = abs_means.loc[0].replace(0, np.nan)
            normed_means = abs_means.div(g0_abs, axis=1) * 100

            bar_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
            colors = bar_colors[:len(cluster_ids)]

            n_panels = len(_feat_groups)
            fig, axes = plt.subplots(n_panels, 1,
                                      figsize=(max(16, max(len(fg[1]) for fg in _feat_groups) * 0.8), 5.0 * n_panels),
                                      constrained_layout=True)
            if n_panels == 1:
                axes = [axes]

            fig.suptitle(f'Mean Feature Values by Group (Normalized to Group 0, k={best_k})',
                         fontsize=14, fontweight='bold', y=1.02)
            # Legend at top
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(cluster_ids))]
            legend_labels = [f'G{c}/G0' for c in cluster_ids]
            fig.legend(legend_handles, legend_labels, loc='upper center',
                       ncol=len(cluster_ids), frameon=False, fontsize=10,
                       bbox_to_anchor=(0.5, 1.01))

            for ax, (panel_title, feats) in zip(axes, _feat_groups):
                x_pos = np.arange(len(feats))
                nc = len(cluster_ids)
                bw = 0.8 / nc
                panel_max = 0
                for ci, cid in enumerate(cluster_ids):
                    vals = normed_means.loc[cid, feats].values.astype(float)
                    panel_max = max(panel_max, np.nanmax(vals))
                    offset = (ci - nc / 2 + 0.5) * bw
                    bars = ax.bar(x_pos + offset, vals, width=bw,
                                  color=colors[ci])
                    # annotate value on top of each bar
                    for bar_rect, raw_v in zip(bars, vals):
                        ax.text(bar_rect.get_x() + bar_rect.get_width() / 2,
                                raw_v, f'{raw_v:.0f}',
                                ha='center', va='bottom', fontsize=5,
                                color=colors[ci], fontweight='bold')

                # Dynamic y-axis from 0 with padding
                _pad = max(panel_max * 0.15, 20)
                ax.set_ylim(0, panel_max + _pad)
                ax.axhline(y=100, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(feats, rotation=45, ha='right', fontsize=7)
                ax.set_ylabel('Normalized to Group 0 (%)')
                ax.set_title(panel_title, fontsize=11)
                ax.grid(axis='y', linestyle='-', alpha=0.3)
                ax.set_axisbelow(True)

            out_path = f'clustering/mean_features_normalized_k{best_k}.png'
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Saved {out_path}')
