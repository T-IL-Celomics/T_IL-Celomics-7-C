import os
import glob
import pandas as pd


def run_descriptive(csv_path: str, output_path: str):
    """Build descriptive statistics per cluster and save to Excel."""
    df = pd.read_csv(csv_path, low_memory=False)

    exclude_cols = [
        'Experiment', 'Parent', 'Treatment', 'PC1', 'PC2', 'Cluster',
        'dt', 'TimeIndex', 'ID'
    ]
    features = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    print(f"  Number of features: {len(features)}")

    grouped_cells = df.groupby(['Cluster', 'Experiment', 'Parent'])[features].mean().reset_index()
    print(f"  Unique cells: {grouped_cells.shape[0]}")

    summary_list = []
    for cluster, data in grouped_cells.groupby('Cluster'):
        stats = {'Cluster': cluster, 'N_Cells': len(data)}
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

    summary_df = pd.DataFrame(summary_list)
    summary_df.to_excel(output_path, index=False)
    print(f"  ✅ Saved '{output_path}'")


# === Auto-discover all k-suffixed embedding_fitting Merged_Clusters_PCA files ===
_single_csv = os.environ.get("PIPELINE_EMBFIT_CLUSTERS_CSV", "")
_base_dir = "fitting"

files_to_process = []

for f in sorted(glob.glob(os.path.join(_base_dir, "embedding_fitting_Merged_Clusters_PCA_k*.csv"))):
    basename = os.path.basename(f)
    k_part = basename.replace("embedding_fitting_Merged_Clusters_PCA_k", "").replace(".csv", "")
    try:
        k_val = int(k_part)
        out_path = os.path.join(_base_dir, f"embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells_k{k_val}.xlsx")
        files_to_process.append((f, out_path, k_val))
    except ValueError:
        pass

if not files_to_process:
    csv_path = _single_csv or os.path.join(_base_dir, "embedding_fitting_Merged_Clusters_PCA.csv")
    if os.path.exists(csv_path):
        files_to_process.append((csv_path, os.path.join(_base_dir, "embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells.xlsx"), None))

for csv_path, out_path, k_val in files_to_process:
    label = f"k={k_val}" if k_val else "default"
    print(f"\nDescriptive table for {label}: {csv_path}")
    run_descriptive(csv_path, out_path)
