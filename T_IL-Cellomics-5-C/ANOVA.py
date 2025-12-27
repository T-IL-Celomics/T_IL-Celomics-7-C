import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import os
import glob

# === 0. Find available k values ===
cluster_files = glob.glob("clustering/Merged_Clusters_PCA_k*.csv")

if not cluster_files:
    cluster_files = ["clustering/Merged_Clusters_PCA.csv"]
    k_values = [None]
else:
    k_values = []
    for f in cluster_files:
        try:
            k = int(f.split("_k")[-1].split(".")[0])
            k_values.append(k)
        except:
            k_values.append(None)
    k_values = sorted(k_values)

print(f"Found cluster files for k values: {k_values}\n")

# === 0b. Load dose information if available ===
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

# === Process each k value ===
for k, cluster_file in zip(k_values, cluster_files):
    print(f"{'='*60}")
    print(f"Processing k={k}")
    print(f"{'='*60}\n")
    
    # Create output directory
    k_suffix = f"_k{k}" if k is not None else ""
    os.makedirs(f"clustering/ANOVA{k_suffix}", exist_ok=True)

    # === 1. Load your data ===
    df = pd.read_csv(cluster_file)
    
    # Merge dose data if available
    if dose_data is not None and "DoseLabel" not in df.columns:
        df = df.merge(dose_data[["Experiment", "Parent", "DoseLabel"]], 
                      on=["Experiment", "Parent"], how="left")

    # === 2. Average per unique cell ===
    df_unique = df.groupby(['Experiment', 'Parent', 'Cluster']).mean(numeric_only=True).reset_index()
    print(f"Unique cells: {df_unique.shape[0]}")

    # === 3. Define features ===
    exclude_cols = ['Experiment', 'Parent', 'Cluster', 'PC1', 'PC2', 'Treatment', 'dt', 'TimeIndex', 'ID']
    features = [col for col in df_unique.columns if col not in exclude_cols]
    print(f"Number of features: {len(features)}")

    # === 4. Compute ANOVA ===
    results = []
    for feature in features:
        groups = [g[feature].dropna().values for _, g in df_unique.groupby('Cluster')]
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

    # === 5. Create MultiIndex DataFrame ===
    anova_df = pd.DataFrame(results)
    anova_df = anova_df.set_index('Feature')

    # === 6. Format columns to have nice names ===
    anova_df.columns = pd.MultiIndex.from_tuples(anova_df.columns)

    # === 7. Style p-values < 0.05 in blue ===
    def highlight_pval(val):
        if val < 0.05:
            return 'color: blue;'
        return ''

    styled = anova_df.style.applymap(highlight_pval, subset=[('Between Groups', 'p-value')])

    # === 8. Save to Excel ===
    output_file = f'clustering/ANOVA{k_suffix}/ANOVA - OneWay{k_suffix}.xlsx'
    styled.to_excel(output_file, engine='openpyxl')

    print(f"✅ Saved '{output_file}' with multi-level headers and p-value highlighting!\n")