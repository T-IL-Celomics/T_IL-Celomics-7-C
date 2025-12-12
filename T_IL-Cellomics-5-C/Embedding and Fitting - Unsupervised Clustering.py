import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === 1. Load files ===
with open("Embedding008.json", "r") as f:
    embedding_data = json.load(f)

with open("fitting_best_model_log_scaled.json", "r") as f:
    fitting_data = json.load(f)

df_data = pd.read_csv("../MergedAndFilteredExperiment008.csv")

# === 2. Build fitting dict ===
fitting_dict = {}
for cell in fitting_data:
    key = (str(cell["Experiment"]), str(cell["Parent"]))
    fitting_dict[key] = cell

# === 3. Define treatments ===
treatments = ["CON0", "BRCACON1", "BRCACON2", "BRCACON3", "BRCACON4", "BRCACON5"]

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
with open("embedding_fitting_combined_by_feature_scaled.json", "w") as f:
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

# === 10. KMeans clustering ===
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# === 11. Create PCA + cluster DataFrame ===
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Experiment"] = [r["Experiment"] for r in records]
pca_df["Parent"] = [r["Parent"] for r in records]
pca_df["Treatment"] = [r["Treatment"] for r in records]
pca_df["Cluster"] = clusters

# === 12. Plot PCA with clusters ===
colors = ['red', 'blue', 'green']
plt.figure(figsize=(8, 6))
for i in range(3):
    subset = pca_df[pca_df["Cluster"] == i]
    plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {i}",
                color=colors[i], edgecolors='k', alpha=0.7)
plt.xlabel(f"PC1 ({explained[0]:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]:.1f}%)")
plt.title("PCA Clusters (k=3)")
plt.legend()
plt.tight_layout()
plt.savefig("embedding_fitting_PCA_KMeans_k3.png")
plt.show()

# === 13. Plot by treatment ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()
for i, treatment in enumerate(treatments):
    ax = axes[i]
    subset = pca_df[pca_df["Treatment"] == treatment]
    for j in range(3):
        cdata = subset[subset["Cluster"] == j]
        ax.scatter(cdata["PC1"], cdata["PC2"], color=colors[j], label=f"Cluster {j}",
                   edgecolors='black', linewidths=0.3, s=40, alpha=0.7)
    ax.set_title(treatment)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.grid(True)

fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1.1, 1.0))
plt.suptitle("PCA by Treatment (k=3)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("embedding_fitting_PCA_KMeans_by_Treatment_k3.png")
plt.show()

# === 14. Merge with original experiment data ===
df_data["Experiment"] = df_data["Experiment"].astype(str)
df_data["Parent"] = df_data["Parent"].astype(str)
pca_df["Experiment"] = pca_df["Experiment"].astype(str)
pca_df["Parent"] = pca_df["Parent"].astype(str)

merged = pd.merge(df_data, pca_df, how="left", on=["Experiment", "Parent"])
merged.to_csv("embedding_fitting_Merged_Clusters_PCA.csv", index=False)
print("Saved: embedding_fitting_Merged_Clusters_PCA.csv")
