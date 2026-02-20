import pandas as pd
import os

# === 1. Load merged data ===
df = pd.read_csv(
    os.environ.get("PIPELINE_CLUSTERS_CSV", "Merged_Clusters_PCA.csv"),
    low_memory=False,
)

# === 2. Columns to exclude ===
exclude_cols = [
    'Experiment', 'Parent', 'Treatment', 'PC1', 'PC2', 'Cluster',
    'dt', 'TimeIndex', 'ID'
]

# === 3. Features list — only keep numeric columns ===
features = [
    col for col in df.columns
    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
]
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
        _ci_z = float(os.environ.get("PIPELINE_CI_ZSCORE", "1.96"))
        ci_lower = mean - _ci_z * se
        ci_upper = mean + _ci_z * se
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
summary_df.to_excel("Descriptive_Table_By_Cluster_UniqueCells.xlsx", index=False)
print("Saved: Descriptive_Table_By_Cluster_UniqueCells.xlsx")
