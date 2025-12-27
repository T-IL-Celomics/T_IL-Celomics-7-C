import pandas as pd
import os

# Ensure output directory exists
os.makedirs("fitting", exist_ok=True)

# === 1. Find available k values ===
import glob
cluster_files = glob.glob("fitting/embedding_fitting_Merged_Clusters_PCA_k*.csv")

if not cluster_files:
    # Fallback to original file if k-specific files don't exist
    cluster_files = ["fitting/embedding_fitting_Merged_Clusters_PCA.csv"]
    k_values = [None]  # Use None to indicate generic processing
else:
    # Extract k values from filenames
    k_values = []
    for f in cluster_files:
        try:
            k = int(f.split("_k")[-1].split(".")[0])
            k_values.append(k)
        except:
            k_values.append(None)
    k_values = sorted(k_values)

print(f"Found cluster files for k values: {k_values}\n")

# === 2. Load dose information if available ===
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

# === 3. Process each k value ===
for k, cluster_file in zip(k_values, cluster_files):
    print(f"{'='*60}")
    print(f"Processing k={k}")
    print(f"{'='*60}\n")
    
    # Load merged data
    df = pd.read_csv(cluster_file)
    
    # === 4. Columns to exclude ===
    exclude_cols = [
        'Experiment', 'Parent', 'Treatment', 'PC1', 'PC2', 'Cluster',
        'dt', 'TimeIndex', 'ID', 'DoseLabel'  # Added DoseLabel to exclusions
    ]

    # === 5. Features list ===
    features = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    print(f"Number of numeric features: {len(features)}")

    # === 6. First, get unique cells ===
    # Average features per cell (Parent+Experiment) within cluster
    grouped_cells = df.groupby(['Cluster', 'Experiment', 'Parent'])[features].mean().reset_index()

    print(f"Unique cells: {grouped_cells.shape[0]}")

    # === 7. Then group by cluster to get descriptive stats ===
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

    # === 8. Final table ===
    summary_df = pd.DataFrame(summary_list)

    print(summary_df.head())

    # === 9. Save descriptive table ===
    k_suffix = f"_k{k}" if k is not None else ""
    summary_df.to_excel(f"fitting/embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells{k_suffix}.xlsx", index=False)
    print(f"Saved: embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells{k_suffix}.xlsx")

    # === 10. Dose-stratified descriptive tables (if dose data available) ===
    if dose_data is not None:
        if "DoseLabel" not in df.columns:
            df = df.merge(dose_data[["Experiment", "Parent", "DoseLabel"]], 
                          on=["Experiment", "Parent"], how="left")
        
        unique_doses = [d for d in df["DoseLabel"].unique() if pd.notna(d) and d != ""]
        
        for dose_label in unique_doses:
            dose_subset = df[df["DoseLabel"] == dose_label].copy()
            
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
            
            # Safe filename
            safe_dose_label = dose_label.replace("|", "_").replace(":", "-")
            summary_dose_df.to_excel(f"fitting/embedding_fitting_Descriptive_Table_By_Cluster_Dose_{safe_dose_label}{k_suffix}.xlsx", index=False)
            print(f"Saved: embedding_fitting_Descriptive_Table_By_Cluster_Dose_{safe_dose_label}{k_suffix}.xlsx")
    
    print()
