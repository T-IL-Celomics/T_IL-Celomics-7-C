import json
import numpy as np
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
df_data = pd.read_csv("cell_data/MergedAndFilteredExperiment008.csv")

# === 1b. Load dose information ===
try:
    dose_data = pd.read_csv("cell_data/dose_dependency_summary_all_wells.csv")
    # Find all dose label columns (Cha*_Category)
    category_cols = sorted([c for c in dose_data.columns if c.endswith("_Category")])
    
    if category_cols:
        dose_data = dose_data[["Experiment", "Parent"] + category_cols].copy()
        print(f"Loaded dose information with channels: {category_cols}")
        # Create a combined dose label from all channels
        dose_data["DoseLabel"] = dose_data[category_cols].apply(
            lambda row: "|".join([f"{c.replace('_Category', '')}:{row[c]}" for c in category_cols if pd.notna(row[c])]),
            axis=1
        )
    else:
        dose_data = None
        print("No dose label column found")
except FileNotFoundError:
    dose_data = None
    print("Dose file not found, skipping dose analysis")

# === 2. Load embedding JSON ===
with open("embeddings/summary_table_Embedding.json", "r") as f:
    data = json.load(f)

# === 3. Define treatment labels ===
#treatments = ["NNIRNOCO", "METRNNIRNOCO", "GABYNNIRNOCO", "NNIRMETRGABYNOCO"]
treatments = ["CON0", "BRCACON1", "BRCACON2", "BRCACON3", "BRCACON4", "BRCACON5"]

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

k_range = range(2, 11)   # usually start from k=2
sil_scores = {}

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
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

# Get 2 best k values
sorted_k = sorted(sil_scores.items(), key=lambda x: x[1], reverse=True)
best_k_values = [sorted_k[0][0], sorted_k[1][0]]
print(f"\nBest 2 k values: {best_k_values[0]} ({sil_scores[best_k_values[0]]:.4f}), {best_k_values[1]} ({sil_scores[best_k_values[1]]:.4f})\n")

# === 7. PCA ===
pca = PCA(n_components=2)
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
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
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
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
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

    # Plot by dose if available
    if dose_data is not None:
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
                ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
                ax.grid(True)

            # delete unused axes
            for j in range(n_d, len(axes)):
                fig.delaxes(axes[j])

            fig.legend(loc="upper right")
            plt.suptitle(f"PCA Clustering by Dose (k={best_k})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"clustering/pca_kmeans_k{best_k}_by_dose.png")
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
            contingency.to_csv(f"clustering/cluster_vs_dose_k{best_k}.csv")
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
            plt.savefig(f"clustering/cluster_vs_dose_heatmap_k{best_k}.png", dpi=300)
            plt.close()
            print(f"Saved cluster_vs_dose_heatmap_k{best_k}.png")
        
        # Per-treatment dose analysis
        if n_t > 0:
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
                    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
                    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
                    ax.grid(True)

                # delete unused axes
                for j in range(n_td, len(axes)):
                    fig.delaxes(axes[j])

                if n_td > 0:
                    fig.legend(loc="upper right")
                    plt.suptitle(f"{treatment} - PCA by Dose (k={best_k})", fontsize=16)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(f"clustering/pca_kmeans_k{best_k}_{treatment}_by_dose.png")
                    plt.show()

                    # Contingency per treatment
                    treatment_contingency = treatment_data.pivot_table(
                        index="DoseLabel",
                        columns="Cluster",
                        values="Parent",
                        aggfunc="nunique",
                        fill_value=0,
                    )
                    
                    treatment_contingency.to_csv(f"clustering/cluster_vs_dose_k{best_k}_{treatment}.csv")
                    print(f"{treatment} - Cluster vs Dose counts:")
                    print(treatment_contingency)
                    print()

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
