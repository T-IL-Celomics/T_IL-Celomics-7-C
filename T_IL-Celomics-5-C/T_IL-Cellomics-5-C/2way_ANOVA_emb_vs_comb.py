import pandas as pd
import os
import glob
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    k_suffix = f"_k{k}" if k is not None else ""
    
    # ========== PARAMETERS ==========
    base_dir = f"fitting/Embedding_Fitting_ANOVA{k_suffix}"
    
    # Find the correct files:
    # embed_csv: from clustering directory (embedding-only clusters from Unsupervised Clustering - Embedding - K=3.py)
    # combined_csv: from fitting directory (embedding+fitting from Embedding and Fitting - Unsupervised Clustering.py)
    if k is not None:
        embed_csv = f"clustering/Merged_Clusters_PCA_k{k}.csv"  # Embedding-only from clustering
        combined_csv = f"fitting/embedding_fitting_Merged_Clusters_PCA_k{k}.csv"  # Embedding+fitting from fitting
    else:
        embed_csv = "clustering/Merged_Clusters_PCA.csv"
        combined_csv = "fitting/embedding_fitting_Merged_Clusters_PCA.csv"

    embed_out = os.path.join(base_dir, f"embedding_means_per_cell{k_suffix}.xlsx")
    combined_out = os.path.join(base_dir, f"embedding_fitting_means_per_cell{k_suffix}.xlsx")

    anova_out = os.path.join(base_dir, f"embedding_fitting_anova_two_way{k_suffix}.csv")
    tukey_out = os.path.join(base_dir, f"embedding_fitting_tukey_two_way{k_suffix}.csv")
    heatmap_png = os.path.join(base_dir, f"embedding_fitting_tukey_pvalue_heatmap{k_suffix}.png")

    # ========== FUNCTION: Compute mean features per cell ==========
    def compute_means(input_csv, output_xlsx):
        df = pd.read_csv(input_csv)

        for col in ['Experiment', 'Parent', 'Cluster']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        non_feature_cols = ['Experiment', 'Parent', 'Cluster', 'TimeIndex', 'dt', 'ID', 'PC1', 'PC2', 'Treatment']
        feature_cols = [
            c for c in df.columns
            if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Group by cell and compute mean of each feature
        grouped = df.groupby(['Experiment', 'Parent', 'Cluster'])[feature_cols].mean().reset_index()
        grouped.to_excel(output_xlsx, index=False)
        print(f"Saved averaged features to: {output_xlsx}")

    # Ensure output directory exists
    os.makedirs(base_dir, exist_ok=True)

    print(f"Looking for source CSV files:")
    print(f"  embed_csv: {embed_csv} (exists: {os.path.exists(embed_csv)})")
    print(f"  combined_csv: {combined_csv} (exists: {os.path.exists(combined_csv)})")

    # ========== STEP 1: Compute means if files don't exist ==========
    if not os.path.exists(embed_out):
        compute_means(embed_csv, embed_out)
    else:
        print(f"File exists, skipping: {embed_out}")

    # Check if combined_csv exists before trying to process it
    if not os.path.exists(combined_csv):
        print(f"Combined clustering file not found: {combined_csv}")
        print(f"Skipping embedding+fitting analysis for k={k}\n")
        continue
    
    if not os.path.exists(combined_out):
        compute_means(combined_csv, combined_out)
    else:
        print(f"File exists, skipping: {combined_out}")

    # ========== STEP 2: Load both summaries and tag Analysis ==========
    df_embed = pd.read_excel(embed_out)
    df_embed["Analysis"] = "Embedding"
    print(f"Loaded embedding data: {df_embed.shape[0]} rows, {df_embed.shape[1]} cols")
    print(f"  Columns: {list(df_embed.columns[:5])}... (showing first 5)")

    df_combined = pd.read_excel(combined_out)
    df_combined["Analysis"] = "Embedding + Fitting"
    print(f"Loaded combined data: {df_combined.shape[0]} rows, {df_combined.shape[1]} cols")
    print(f"  Columns: {list(df_combined.columns[:5])}... (showing first 5)")
    
    # Check if they're actually different
    common_cols = [c for c in df_embed.columns if c in df_combined.columns and c not in ['Analysis']]
    if len(common_cols) > 0:
        # Compare first feature for first row
        try:
            feat_col = [c for c in common_cols if c not in ['Experiment', 'Parent', 'Cluster']][0]
            print(f"  Sample comparison ({feat_col}): embed={df_embed[feat_col].iloc[0]:.4f}, combined={df_combined[feat_col].iloc[0]:.4f}")
        except:
            pass

    # ========== STEP 3: Combine and reshape ==========
    combined = pd.concat([df_embed, df_combined], ignore_index=True)
    features = [c for c in combined.columns if c not in ['Experiment', 'Parent', 'Cluster', 'Analysis']]

    print(f"Combined data shape: {combined.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Analysis groups: {combined['Analysis'].unique()}")
    print(f"Analysis group counts: {combined['Analysis'].value_counts()}")
    print(f"Cluster values: {combined['Cluster'].unique()}")

    long_df = combined.melt(
        id_vars=['Analysis', 'Cluster'],
        value_vars=features,
        var_name='Feature',
        value_name='Value'
    ).dropna(subset=['Value'])
    
    print(f"Long df shape: {long_df.shape}")
    print(f"Long df Analysis counts: {long_df['Analysis'].value_counts()}\n")

    # ========== STEP 4: Two-Way ANOVA ==========
    anova_results = []
    for feat, grp in long_df.groupby('Feature'):
        if grp['Analysis'].nunique() < 2:
            continue
        model = ols('Value ~ C(Analysis) * C(Cluster)', data=grp).fit()
        table = sm.stats.anova_lm(model, typ=2)
        anova_results.append({
            'Feature': feat,
            'p_Analysis': table.loc['C(Analysis)', 'PR(>F)'],
            'p_Cluster': table.loc['C(Cluster)', 'PR(>F)'],
            'p_Interaction': table.loc['C(Analysis):C(Cluster)', 'PR(>F)']
        })

    anova_df = pd.DataFrame(anova_results)
    if len(anova_df) == 0:
        print(f"WARNING: No ANOVA results for k={k}. Check data shape and Cluster/Analysis uniqueness")
    anova_df['p_Analysis_FDR'] = multipletests(anova_df['p_Analysis'], method='fdr_bh')[1]
    anova_df['p_Cluster_FDR'] = multipletests(anova_df['p_Cluster'], method='fdr_bh')[1]
    anova_df['p_Interaction_FDR'] = multipletests(anova_df['p_Interaction'], method='fdr_bh')[1]
    anova_df.to_csv(anova_out, index=False)
    print(f"ANOVA results: {len(anova_df)} features, p-values: min={anova_df['p_Interaction'].min():.4f}, max={anova_df['p_Interaction'].max():.4f}")
    print(f"Saved ANOVA results to: {anova_out}")

    # ========== STEP 5: Tukey HSD ==========
    tukey_results = []
    for feat, grp in long_df.groupby('Feature'):
        for cl in sorted(grp['Cluster'].unique()):
            block = grp[grp['Cluster'] == cl]
            if block['Analysis'].nunique() < 2:
                continue
            tuk = pairwise_tukeyhsd(block['Value'], block['Analysis'])
            for row in tuk.summary().data[1:]:
                g1, g2, md, p_adj, lo, hi, rej = row
                if set([g1, g2]) == {'Embedding', 'Embedding + Fitting'}:
                    tukey_results.append({
                        'Feature': feat,
                        'Cluster': cl,
                        'MeanDiff': md,
                        'p_adj': p_adj,
                        'lower': lo,
                        'upper': hi,
                        'reject': rej
                    })

    tukey_df = pd.DataFrame(tukey_results)
    if len(tukey_df) == 0:
        print(f"WARNING: No Tukey results for k={k}. Check if Analysis groups exist")
        print(f"  Long df shape: {long_df.shape}")
        print(f"  Unique Analysis values: {long_df['Analysis'].unique()}")
        print(f"  Unique Cluster values: {long_df['Cluster'].unique()}")
    else:
        print(f"Tukey results: {len(tukey_df)} comparisons, p-values: min={tukey_df['p_adj'].min():.4f}, max={tukey_df['p_adj'].max():.4f}")
    tukey_df.to_csv(tukey_out, index=False)
    print(f"Saved Tukey results to: {tukey_out}")

    # ========== STEP 6: Tukey heatmap ==========
    if len(tukey_df) > 0:
        heatmap_data = tukey_df.pivot(index='Feature', columns='Cluster', values='p_adj')
        plt.figure(figsize=(10, max(6, len(heatmap_data)/4)))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap='coolwarm_r',
            cbar_kws={'label':'Adjusted p-value'},
            annot_kws={'color':'white', 'fontsize':7.5}
        )
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.title(f'Tukey HSD Adjusted p-values by Feature & Cluster (k={k})')
        plt.tight_layout()
        plt.savefig(heatmap_png, dpi=300)
        print(f"Saved heatmap to: {heatmap_png}\n")
    else:
        print(f"Skipping heatmap - no Tukey results for k={k}\n")

