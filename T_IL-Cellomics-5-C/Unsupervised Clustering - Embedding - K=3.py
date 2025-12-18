import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
# Ensure output directory exists
os.makedirs("clustering", exist_ok=True)

# === 1. Load original experiment data ===
df_data = pd.read_excel("cell_data/summary_table.xlsx")

# === 2. Load embedding JSON ===
with open("embeddings/summary_table_Embedding.json", "r") as f:
    data = json.load(f)

# === 3. Define treatment labels ===
treatments = ["NNIRNOCO", "METRNNIRNOCO", "GABYNNIRNOCO", "NNIRMETRGABYNOCO"]

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
        treatment = next((t for t in treatments if t in cell["Experiment"]), "Unknown")
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

# === 7. PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_ * 100
print(f"Explained variance: PC1 = {explained_var[0]:.2f}%, PC2 = {explained_var[1]:.2f}%")

# === 8. Clustering ===
best_k = 3  # Chosen based on highest silhouette score
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
final_labels = kmeans_final.fit_predict(X_scaled)

# === 9. Build PCA dataframe ===
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Experiment"] = [r["Experiment"] for r in records]
pca_df["Parent"] = [r["Parent"] for r in records]
pca_df["Treatment"] = [r["Treatment"] for r in records]
pca_df["Cluster"] = final_labels

# === 10. Save cluster assignments ===
pca_df.to_csv("clustering/cluster_assignments_k3.csv", index=False)
print("Saved cluster_assignments_k3.csv")

# === 11. Plot full PCA with clusters ===
colors = ['red', 'blue', 'green']
plt.figure(figsize=(8, 6))
for i in range(best_k):
    cluster_data = pca_df[pca_df["Cluster"] == i]
    plt.scatter(cluster_data["PC1"], cluster_data["PC2"], color=colors[i % len(colors)],
                label=f"Cluster {i}", edgecolors='black', s=50, alpha=0.8)
plt.xlabel(f"PC1 ({explained_var[0]:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]:.1f}%)")
plt.title(f"PCA with KMeans Clusters (k={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig("clustering/pca_kmeans_k3_clusters.png")
plt.show()

# === 12. Plot by treatment ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()
for i, treatment in enumerate(treatments):
    ax = axes[i]
    subset = pca_df[pca_df["Treatment"] == treatment]
    for j in range(best_k):
        cdata = subset[subset["Cluster"] == j]
        ax.scatter(cdata["PC1"], cdata["PC2"], color=colors[j % len(colors)],
                   label=f"Cluster {j}", edgecolors='black', linewidths=0.3, s=40, alpha=0.7)
    ax.set_title(treatment)
    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    ax.grid(True)

fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1.1, 1.0))
plt.suptitle("PCA Clustering by Treatment (k=3)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("clustering/pca_kmeans_k3_by_treatment.png")
plt.show()

# === 13. Merge with original full data ===
# Convert merge keys to string for safety
df_data["Experiment"] = df_data["Experiment"].astype(str)
df_data["Parent"] = df_data["Parent"].astype(str)
pca_df["Experiment"] = pca_df["Experiment"].astype(str)
pca_df["Parent"] = pca_df["Parent"].astype(str)

merged_df = pd.merge(
    df_data,
    pca_df[['Experiment', 'Parent', 'PC1', 'PC2', 'Cluster', 'Treatment']],
    how='left',
    on=['Experiment', 'Parent']
)

# === 14. Save merged file ===
merged_df.to_csv("clustering/Merged_Clusters_PCA.csv", index=False)
print("Saved as Merged_Clusters_PCA.csv")
