import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import math
import os

# Ensure output directory exists before any writes
os.makedirs("fitting", exist_ok=True)

# === 0. Load dose information if available ===
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
    category_cols = sorted([c for c in dose_data.columns if c.endswith("_Category")])
    if category_cols:
        dose_data = dose_data[["Experiment", "Parent"] + category_cols].copy()
        if _control_channel:
            print(f"Control channel to exclude from DoseLabel: {_control_channel}")

        dose_data["DoseLabel"] = dose_data.apply(
            lambda row: _make_dose_label(row, category_cols), axis=1
        )
        print(f"Loaded dose information with channels: {category_cols}\n")
    else:
        dose_data = None
        print("No dose label column found\n")
except FileNotFoundError:
    dose_data = None
    print("Dose file not found, skipping dose analysis\n")

# === 1. Load files ===
with open(os.environ.get("PIPELINE_EMBEDDING_JSON", "embeddings/summary_table_Embedding.json"), "r") as f:
    embedding_data = json.load(f)

with open(os.environ.get("PIPELINE_FITTING_JSON", "fitting_best_model_log_scaled.json"), "r") as f:
    fitting_data = json.load(f)

_merged_csv = os.environ.get("PIPELINE_MERGED_CSV", "cell_data/MergedAndFilteredExperiment008.csv")
if _merged_csv.lower().endswith((".xlsx", ".xls")):
    df_data = pd.read_excel(_merged_csv)
else:
    try:
        df_data = pd.read_csv(_merged_csv)
    except UnicodeDecodeError:
        print(f"[warn] UTF-8 decode failed for {_merged_csv}, retrying with latin-1")
        df_data = pd.read_csv(_merged_csv, encoding="latin-1")

# === 2. Build fitting dict ===
fitting_dict = {}
for cell in fitting_data:
    key = (str(cell["Experiment"]), str(cell["Parent"]))
    fitting_dict[key] = cell

# === 3. Define treatments (from env or default) ===
_treatments_env = os.environ.get("PIPELINE_TREATMENTS", "CON0, BRCACON1, BRCACON2, BRCACON3, BRCACON4, BRCACON5")
treatments = [t.strip() for t in _treatments_env.split(",") if t.strip()]
print(f"Treatments: {treatments}")
# === 4. Get feature names
all_features = [k for k in embedding_data[0].keys() if k not in ["Experiment", "Parent"]]

# === 5. Scale embedding part only ===
scaler = StandardScaler()
scaled_cells = []

# First build flat embedding per feature for scaling
for feature in all_features:
    feature_matrix = []
    valid_cells = []
    for cell in embedding_data:
        values = cell.get(feature, [])
        if isinstance(values, list) and not any(np.isnan(values)):
            feature_matrix.append(values)
            valid_cells.append(cell)
    if not feature_matrix:
        continue
    scaled = scaler.fit_transform(feature_matrix)
    for i, cell in enumerate(valid_cells):
        if "embedding_scaled" not in cell:
            cell["embedding_scaled"] = {}
        cell["embedding_scaled"][feature] = scaled[i].tolist()

# === 6. Combine embedding_scaled + fitting per feature ===
combined_data = []
for cell in embedding_data:
    experiment = str(cell["Experiment"])
    parent = str(cell["Parent"])
    key = (experiment, parent)
    treatment = next((t for t in treatments if t in experiment), "Unknown")

    if "embedding_scaled" not in cell:
        continue

    combined_cell = {
        "Experiment": experiment,
        "Parent": parent,
        "Treatment": treatment
    }

    fitting_feats = fitting_dict.get(key, {}).get("fitting", {})

    for feature in all_features:
        emb = cell["embedding_scaled"].get(feature)
        fit = fitting_feats.get(feature)
        if emb is None or fit is None or any(np.isnan(emb)) or any(np.isnan(fit)):
            continue
        combined_cell[feature] = emb + fit

    combined_data.append(combined_cell)

# === 7. Save combined JSON ===
with open("fitting/embedding_fitting_combined_by_feature_scaled.json", "w") as f:
    json.dump(combined_data, f, indent=2)
print("Saved: embedding_fitting_combined_by_feature_scaled.json")

# === 8. Flatten vectors for PCA ===
records = []
X = []
for cell in combined_data:
    flat_vec = []
    for feature in all_features:
        values = cell.get(feature)
        if values is not None:
            flat_vec.extend(values)
    if flat_vec and not any(np.isnan(flat_vec)):
        X.append(flat_vec)
        records.append({
            "Experiment": cell["Experiment"],
            "Parent": cell["Parent"],
            "Treatment": cell["Treatment"]
        })

X = np.array(X)

# === 9. PCA ===
_pca_components = int(os.environ.get("PIPELINE_PCA_COMPONENTS", "2"))
pca = PCA(n_components=_pca_components)
X_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_ * 100
print(f"Explained variance: PC1 = {explained[0]:.2f}%, PC2 = {explained[1]:.2f}%")

# === 10. Find best k values using silhouette score ===
_k_min = int(os.environ.get("PIPELINE_K_MIN", "2"))
_k_max = int(os.environ.get("PIPELINE_K_MAX", "10"))
_n_init = int(os.environ.get("PIPELINE_KMEANS_N_INIT", "10"))
_random_state = int(os.environ.get("PIPELINE_KMEANS_SEED", "42"))
_num_best_k = int(os.environ.get("PIPELINE_NUM_BEST_K", "2"))
_ci_z = float(os.environ.get("PIPELINE_CI_ZSCORE", "1.96"))
silhouette_scores = {}
k_range = range(_k_min, _k_max + 1)  # test k from 2 to k_max
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=_random_state, n_init=_n_init)
    clusters_temp = kmeans_temp.fit_predict(X)
    score = silhouette_score(X, clusters_temp)
    silhouette_scores[k] = score
    print(f"k={k}: silhouette_score={score:.4f}")

# Get best k values
sorted_k = sorted(silhouette_scores.items(), key=lambda x: x[1], reverse=True)
best_k_values = [sorted_k[i][0] for i in range(min(_num_best_k, len(sorted_k)))]
print(f"\nBest {_num_best_k} k values: {', '.join(f'{k} ({silhouette_scores[k]:.4f})' for k in best_k_values)}\n")

# === 11-15. Process both k values ===
for best_k in best_k_values:
    print(f"Processing k={best_k}...")
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=best_k, random_state=_random_state, n_init=_n_init)
    clusters = kmeans.fit_predict(X)

    # Create PCA + cluster DataFrame
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["Experiment"] = [r["Experiment"] for r in records]
    pca_df["Parent"] = [r["Parent"] for r in records]
    pca_df["Treatment"] = [r["Treatment"] for r in records]
    pca_df["Cluster"] = clusters

    # Merge with dose information if available
    if dose_data is not None:
        pca_df["Experiment"] = pca_df["Experiment"].astype(str)
        pca_df["Parent"] = pca_df["Parent"].astype(str)
        dose_data["Experiment"] = dose_data["Experiment"].astype(str)
        dose_data["Parent"] = dose_data["Parent"].astype(str)
        pca_df = pca_df.merge(dose_data, on=["Experiment", "Parent"], how="left")

    # Plot PCA with clusters
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    plt.figure(figsize=(8, 6))
    for i in range(best_k):
        subset = pca_df[pca_df["Cluster"] == i]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {i}",
                    color=colors[i], edgecolors='k', alpha=0.7)
    plt.xlabel(f"PC1 ({explained[0]:.1f}%)")
    plt.ylabel(f"PC2 ({explained[1]:.1f}%)")
    plt.title(f"PCA Clusters (k={best_k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fitting/embedding_fitting_PCA_KMeans_k{best_k}.png")
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
            ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
            ax.grid(True)

        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1.1, 1.0))
        plt.suptitle(f"PCA by Treatment (k={best_k})")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"fitting/embedding_fitting_PCA_KMeans_by_Treatment_k{best_k}.png")
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
                ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
                ax.grid(True)

            for j in range(n_td, len(axes)):
                fig.delaxes(axes[j])

            fig.legend(loc="upper right")
            plt.suptitle(f"{treatment} – PCA by Dose (k={best_k})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"fitting/embedding_fitting_PCA_KMeans_k{best_k}_{treatment}_by_dose.png")
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
                f"fitting/cluster_vs_dose_k{best_k}_{treatment}.csv"
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
                f"fitting/cluster_vs_dose_heatmap_k{best_k}_{treatment}.png",
                dpi=300,
            )
            plt.close()
            print(f"Saved cluster_vs_dose_heatmap_k{best_k}_{treatment}.png\n")

    # Merge with original experiment data
    df_data_copy = df_data.copy()
    df_data_copy["Experiment"] = df_data_copy["Experiment"].astype(str)
    df_data_copy["Parent"] = df_data_copy["Parent"].astype(str)
    pca_df["Experiment"] = pca_df["Experiment"].astype(str)
    pca_df["Parent"] = pca_df["Parent"].astype(str)

    merged = pd.merge(df_data_copy, pca_df, how="left", on=["Experiment", "Parent"])
    merged.to_csv(f"fitting/embedding_fitting_Merged_Clusters_PCA_k{best_k}.csv", index=False)
    print(f"Saved: embedding_fitting_Merged_Clusters_PCA_k{best_k}.csv\n")

# === Generate descriptive tables by cluster ===

print("\n" + "="*60)
print("Generating descriptive tables by cluster")
print("="*60 + "\n")

# Load dose information if available (re-use control-channel-aware builder)
try:
    dose_data = pd.read_csv(_dose_csv)
    category_cols = sorted([c for c in dose_data.columns if c.endswith("_Category")])
    if category_cols:
        dose_data = dose_data[["Experiment", "Parent"] + category_cols].copy()
        dose_data["DoseLabel"] = dose_data.apply(
            lambda row: _make_dose_label(row, category_cols), axis=1
        )
        print(f"Loaded dose information\n")
    else:
        dose_data = None
except FileNotFoundError:
    dose_data = None
    print("Dose file not found\n")

# Process each k value
for best_k in best_k_values:
    print(f"{'='*60}")
    print(f"Processing descriptive tables for k={best_k}")
    print(f"{'='*60}\n")
    
    # Load the merged data for this k
    df = pd.read_csv(f"fitting/embedding_fitting_Merged_Clusters_PCA_k{best_k}.csv")
    
    # Columns to exclude
    exclude_cols = [
        'Experiment', 'Parent', 'Treatment', 'PC1', 'PC2', 'Cluster',
        'dt', 'TimeIndex', 'ID', 'DoseLabel'
    ]
    
    # Get feature names
    features = [col for col in df.columns if col not in exclude_cols]
    # Filter to only numeric features
    features = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    print(f"Number of features: {len(features)}")
    
    # Average features per cell within cluster
    grouped_cells = df.groupby(['Cluster', 'Experiment', 'Parent'])[features].mean().reset_index()
    print(f"Unique cells: {grouped_cells.shape[0]}")
    
    # Create descriptive stats by cluster
    summary_list = []
    for cluster, data in grouped_cells.groupby('Cluster'):
        stats = {
            'Cluster': cluster,
            'N_Cells': len(data)
        }
        for feature in features:
            mean = data[feature].mean()
            std = data[feature].std()
            se = std / (len(data) ** 0.5)
            ci_lower = mean - _ci_z * se
            ci_upper = mean + _ci_z * se
            stats[f'{feature}_Mean'] = mean
            stats[f'{feature}_Std'] = std
            stats[f'{feature}_SE'] = se
            stats[f'{feature}_CI_Lower'] = ci_lower
            stats[f'{feature}_CI_Upper'] = ci_upper
        summary_list.append(stats)
    
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_excel(f"fitting/embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells_k{best_k}.xlsx", index=False)
    print(f"Saved: embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells_k{best_k}.xlsx")
    
    # Dose-stratified descriptive tables
    if dose_data is not None:
        # Check if DoseLabel already exists, if not merge it
        if "DoseLabel" not in df.columns:
            df = df.merge(dose_data[["Experiment", "Parent", "DoseLabel"]], 
                          on=["Experiment", "Parent"], how="left")
        
        # Per-treatment dose analysis
        treatments_in_data = [t for t in treatments if (df["Treatment"] == t).any()]
        
        for treatment in treatments_in_data:
            treatment_data = df[df["Treatment"] == treatment].copy()
            treatment_doses = [d for d in treatment_data["DoseLabel"].unique() if pd.notna(d) and d != ""]
            
            if len(treatment_doses) == 0:
                continue
            
            for dose_label in treatment_doses:
                dose_subset = treatment_data[treatment_data["DoseLabel"] == dose_label].copy()
                grouped_cells_dose = dose_subset.groupby(['Cluster', 'Experiment', 'Parent'])[features].mean().reset_index()
                
                summary_list_dose = []
                for cluster, data in grouped_cells_dose.groupby('Cluster'):
                    stats = {
                        'Cluster': cluster,
                        'N_Cells': len(data)
                    }
                    for feature in features:
                        mean = data[feature].mean()
                        std = data[feature].std()
                        se = std / (len(data) ** 0.5)
                        ci_lower = mean - _ci_z * se
                        ci_upper = mean + _ci_z * se
                        stats[f'{feature}_Mean'] = mean
                        stats[f'{feature}_Std'] = std
                        stats[f'{feature}_SE'] = se
                        stats[f'{feature}_CI_Lower'] = ci_lower
                        stats[f'{feature}_CI_Upper'] = ci_upper
                    summary_list_dose.append(stats)
                
                summary_dose_df = pd.DataFrame(summary_list_dose)
                safe_dose_label = dose_label.replace("|", "_").replace(":", "-")
                summary_dose_df.to_excel(f"fitting/embedding_fitting_Descriptive_Table_By_Cluster_Dose_{safe_dose_label}_{treatment}_k{best_k}.xlsx", index=False)
                print(f"Saved: embedding_fitting_Descriptive_Table_By_Cluster_Dose_{safe_dose_label}_{treatment}_k{best_k}.xlsx")
    
    print()
