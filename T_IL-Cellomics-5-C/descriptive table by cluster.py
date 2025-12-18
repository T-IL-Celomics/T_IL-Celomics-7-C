import pandas as pd

# === 1. Load merged data ===
df = pd.read_csv("clustering/Merged_Clusters_PCA.csv")

# === 2. Columns to exclude ===
exclude_cols = [
    'Experiment', 'Parent', 'Treatment', 'PC1', 'PC2', 'Cluster',
    'dt', 'TimeIndex', 'ID'
]

# === 3. Features list ===
features = [col for col in df.columns if col not in exclude_cols]
print(f"Number of features: {len(features)}")

# === 4. First, get unique cells ===
# Average features per cell (Parent+Experiment) within cluster
grouped_cells = df.groupby(['Cluster', 'Experiment', 'Parent'])[features].mean().reset_index()

print(f"Unique cells: {grouped_cells.shape[0]}")

# === 5. Then group by cluster to get descriptive stats ===
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

# === 6. Final table ===
summary_df = pd.DataFrame(summary_list)

print(summary_df.head())

# === 7. Save ===
summary_df.to_excel("clustering/Descriptive_Table_By_Cluster_UniqueCells.xlsx", index=False)
print("Saved: clustering/Descriptive_Table_By_Cluster_UniqueCells.xlsx")
