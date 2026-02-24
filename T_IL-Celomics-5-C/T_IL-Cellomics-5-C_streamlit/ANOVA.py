import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import f_oneway


def run_anova(csv_path: str, output_path: str):
    """Run one-way ANOVA on the given merged-clusters CSV and save to Excel."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Average per unique cell
    df_unique = df.groupby(['Experiment', 'Parent', 'Cluster']).mean(numeric_only=True).reset_index()
    print(f"  Unique cells: {df_unique.shape[0]}")

    # Define features — only keep numeric columns
    exclude_cols = ['Experiment', 'Parent', 'Cluster', 'PC1', 'PC2', 'Treatment', 'dt', 'TimeIndex', 'ID']
    features = [
        col for col in df_unique.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_unique[col])
    ]
    print(f"  Number of features: {len(features)}")

    # Compute ANOVA
    results = []
    for feature in features:
        groups = [g[feature].dropna().values for _, g in df_unique.groupby('Cluster')]
        # Skip features where any group has <2 observations
        if any(len(g) < 2 for g in groups):
            print(f"  ⚠️ Skipping '{feature}': a cluster has <2 observations after dropna")
            continue
        f_stat, p_value = f_oneway(*groups)
        grand_mean = df_unique[feature].mean()
        n_groups = len(groups)
        n_total = len(df_unique[feature].dropna())

        ss_between = sum(len(g)*(np.mean(g)-grand_mean)**2 for g in groups)
        ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in groups)
        ss_total = ss_between + ss_within

        df_between = n_groups - 1
        df_within = n_total - n_groups
        df_total = n_total - 1

        ms_between = ss_between / df_between if df_between > 0 else np.nan
        ms_within = ss_within / df_within if df_within > 0 else np.nan

        results.append({
            'Feature': feature,
            ('Between Groups', 'Sum of Squares'): ss_between,
            ('Between Groups', 'df'): df_between,
            ('Between Groups', 'Mean Square'): ms_between,
            ('Between Groups', 'F statistic'): f_stat,
            ('Between Groups', 'p-value'): p_value,
            ('Within Groups', 'Sum of Squares'): ss_within,
            ('Within Groups', 'df'): df_within,
            ('Within Groups', 'Mean Square'): ms_within,
            ('Total', 'Sum of Squares'): ss_total,
            ('Total', 'df'): df_total
        })

    anova_df = pd.DataFrame(results).set_index('Feature')
    anova_df.columns = pd.MultiIndex.from_tuples(anova_df.columns)

    _pval_threshold = float(os.environ.get("PIPELINE_ANOVA_PVAL", "0.05"))

    def highlight_pval(val):
        if val < _pval_threshold:
            return 'color: blue;'
        return ''

    styled = anova_df.style.map(highlight_pval, subset=[('Between Groups', 'p-value')])
    styled.to_excel(output_path, engine='openpyxl')
    print(f"  ✅ Saved '{output_path}'")


# === Auto-discover all k-suffixed Merged_Clusters_PCA files ===
_single_csv = os.environ.get("PIPELINE_CLUSTERS_CSV", "")
_base_dir = "clustering"

# Collect all files to process
files_to_process = []

# Find all k-suffixed files
for f in sorted(glob.glob(os.path.join(_base_dir, "Merged_Clusters_PCA_k*.csv"))):
    # Extract k value from filename
    basename = os.path.basename(f)
    k_part = basename.replace("Merged_Clusters_PCA_k", "").replace(".csv", "")
    try:
        k_val = int(k_part)
        out_path = os.path.join(_base_dir, f"ANOVA - OneWay_k{k_val}.xlsx")
        files_to_process.append((f, out_path, k_val))
    except ValueError:
        pass

if not files_to_process:
    # Fallback: use the single CSV from env or default
    csv_path = _single_csv or os.path.join(_base_dir, "Merged_Clusters_PCA.csv")
    if os.path.exists(csv_path):
        files_to_process.append((csv_path, os.path.join(_base_dir, "ANOVA - OneWay.xlsx"), None))

for csv_path, out_path, k_val in files_to_process:
    label = f"k={k_val}" if k_val else "default"
    print(f"\nRunning ANOVA for {label}: {csv_path}")
    run_anova(csv_path, out_path)
