import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import math
import os

# === 0. Load dose information if available ===
try:
    dose_data = pd.read_csv("cell_data/dose_dependency_summary_all_wells.csv")
    category_cols = sorted([c for c in dose_data.columns if c.endswith("_Category")])
    if category_cols:
        dose_data = dose_data[["Experiment", "Parent"] + category_cols].copy()
        dose_data["DoseLabel"] = dose_data[category_cols].apply(
            lambda row: "|".join([f"{c.replace('_Category', '')}:{row[c]}" for c in category_cols if pd.notna(row[c])]),
            axis=1
        )
        print(f"Loaded dose information with channels: {category_cols}\n")
    else:
        dose_data = None
        print("No dose label column found\n")
except FileNotFoundError:
    dose_data = None
    print("Dose file not found, skipping dose analysis\n")

# === 1. Load files ===
with open("embeddings/summary_table_Embedding.json", "r") as f:
    embedding_data = json.load(f)

with open("fitting/fitting_best_model_log_scaled.json", "r") as f:
    fitting_data = json.load(f)

df_data = pd.read_csv("cell_data/summary_table_filled_no_extrap_FINAL_NO_NAN.csv")

# === 2. Build fitting dict ===
fitting_dict = {}
for cell in fitting_data:
    key = (str(cell["Experiment"]), str(cell["Parent"]))
    fitting_dict[key] = cell

# === 3. Define treatments ===
treatments = ["NNIRMETRGABYNOCO", "METRNNIRNOCO", "GABYNNIRNOCO","NNIRNOCO" ]

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
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_ * 100
print(f"Explained variance: PC1 = {explained[0]:.2f}%, PC2 = {explained[1]:.2f}%")

# === 10. Find best 2 k values using silhouette score ===
silhouette_scores = {}
k_range = range(2, 11)  # test k from 2 to 10
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_temp = kmeans_temp.fit_predict(X)
    score = silhouette_score(X, clusters_temp)
    silhouette_scores[k] = score
    print(f"k={k}: silhouette_score={score:.4f}")

# Get 2 best k values
sorted_k = sorted(silhouette_scores.items(), key=lambda x: x[1], reverse=True)
best_k_values = [sorted_k[0][0], sorted_k[1][0]]
print(f"\nBest 2 k values: {best_k_values[0]} ({silhouette_scores[best_k_values[0]]:.4f}), {best_k_values[1]} ({silhouette_scores[best_k_values[1]]:.4f})\n")

# === 11-15. Process both k values ===
for best_k in best_k_values:
    print(f"Processing k={best_k}...")
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # Create PCA + cluster DataFrame
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
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

    # Plot by dose if available
    from scipy.stats import f_oneway
    
    if dose_data is not None and "DoseLabel" in pca_df.columns:
        doses_present = [d for d in pca_df["DoseLabel"].unique() if pd.notna(d) and d != ""]
        
        if len(doses_present) > 0:
            n_d = len(doses_present)
            ncols_dose = math.ceil(math.sqrt(n_d))
            nrows_dose = math.ceil(n_d / ncols_dose)

            fig, axes = plt.subplots(nrows_dose, ncols_dose, figsize=(6*ncols_dose, 5*nrows_dose), sharex=True, sharey=True)
            axes = np.array(axes).reshape(-1)

            for i, dose in enumerate(doses_present):
                ax = axes[i]
                subset = pca_df[pca_df["DoseLabel"] == dose]

                for j in range(best_k):
                    cdata = subset[subset["Cluster"] == j]
                    ax.scatter(
                        cdata["PC1"], cdata["PC2"],
                        color=colors[j],
                        edgecolors="black", linewidths=0.3,
                        s=40, alpha=0.7,
                        label=f"Cluster {j}" if i == 0 else None
                    )

                ax.set_title(f"Dose: {dose}")
                ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
                ax.grid(True)

            # delete unused axes
            for j in range(n_d, len(axes)):
                fig.delaxes(axes[j])

            fig.legend(loc="upper right")
            plt.suptitle(f"PCA Clustering by Dose (k={best_k})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"fitting/embedding_fitting_PCA_KMeans_by_Dose_k{best_k}.png")
            plt.show()

            # Create contingency table: Dose vs Cluster
            contingency = pca_df.pivot_table(
                index="DoseLabel",
                columns="Cluster",
                values="Parent",
                aggfunc="nunique",
                fill_value=0,
            )
            
            print(f"\nCluster vs Dose contingency table (k={best_k}):")
            print(contingency)
            contingency.to_csv(f"fitting/cluster_vs_dose_k{best_k}.csv")
            print(f"Saved cluster_vs_dose_k{best_k}.csv")

            # Plot heatmap: Dose vs Cluster
            plt.figure(figsize=(6, 4))
            plt.imshow(contingency, aspect="auto", cmap="YlOrRd")
            plt.colorbar(label="Number of cells")
            plt.xticks(range(len(contingency.columns)), contingency.columns)
            plt.yticks(range(len(contingency.index)), contingency.index)
            plt.xlabel("Cluster")
            plt.ylabel("Dose")
            plt.title(f"Cell counts: Dose vs Cluster (k={best_k})")
            plt.tight_layout()
            plt.savefig(f"fitting/cluster_vs_dose_heatmap_k{best_k}.png", dpi=300)
            plt.close()
            print(f"Saved cluster_vs_dose_heatmap_k{best_k}.png")
        
        # Per-treatment dose analysis
        print(f"\n{'='*60}")
        print(f"Dose analysis per treatment (k={best_k})")
        print(f"{'='*60}\n")
        
        for treatment in treatments_present:
            treatment_data = pca_df[pca_df["Treatment"] == treatment].copy()
            treatment_doses = [d for d in treatment_data["DoseLabel"].unique() if pd.notna(d) and d != ""]
            
            if len(treatment_doses) == 0:
                continue
                
            n_td = len(treatment_doses)
            ncols_td = math.ceil(math.sqrt(n_td))
            nrows_td = math.ceil(n_td / ncols_td)

            fig, axes = plt.subplots(nrows_td, ncols_td, figsize=(6*ncols_td, 5*nrows_td), sharex=True, sharey=True)
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

                ax.set_title(f"{dose}")
                ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
                ax.grid(True)

            # delete unused axes
            for j in range(n_td, len(axes)):
                fig.delaxes(axes[j])

            if n_td > 0:
                fig.legend(loc="upper right")
                plt.suptitle(f"{treatment} - PCA by Dose (k={best_k})", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(f"fitting/embedding_fitting_PCA_KMeans_k{best_k}_{treatment}_by_dose.png")
                plt.show()

                # Contingency per treatment
                treatment_contingency = treatment_data.pivot_table(
                    index="DoseLabel",
                    columns="Cluster",
                    values="Parent",
                    aggfunc="nunique",
                    fill_value=0,
                )
                
                treatment_contingency.to_csv(f"fitting/cluster_vs_dose_k{best_k}_{treatment}.csv")
                print(f"{treatment} - Cluster vs Dose counts:")
                print(treatment_contingency)
                print()

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
import os
os.makedirs("fitting", exist_ok=True)

print("\n" + "="*60)
print("Generating descriptive tables by cluster")
print("="*60 + "\n")

# Load dose information if available
try:
    dose_data = pd.read_csv("cell_data/dose_dependency_summary_all_wells.csv")
    category_cols = sorted([c for c in dose_data.columns if c.endswith("_Category")])
    if category_cols:
        dose_data = dose_data[["Experiment", "Parent"] + category_cols].copy()
        dose_data["DoseLabel"] = dose_data[category_cols].apply(
            lambda row: "|".join([f"{c.replace('_Category', '')}:{row[c]}" for c in category_cols if pd.notna(row[c])]),
            axis=1
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
            ci_lower = mean - 1.96 * se
            ci_upper = mean + 1.96 * se
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
                        ci_lower = mean - 1.96 * se
                        ci_upper = mean + 1.96 * se
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
