import pandas as pd
import os
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== PARAMETERS ==========
base_dir = "Embedding - K=3"
embed_csv = os.path.join(base_dir, "Merged_Clusters_PCA.csv")
combined_csv = os.path.join(base_dir, "embedding_fitting_Merged_Clusters_PCA.csv")

embed_out = os.path.join(base_dir, "embedding_means_per_cell.xlsx")
combined_out = os.path.join(base_dir, "embedding_fitting_means_per_cell.xlsx")

anova_out = os.path.join(base_dir, "embedding_fitting_anova_two_way.csv")
tukey_out = os.path.join(base_dir, "embedding_fitting_tukey_two_way.csv")
heatmap_png = os.path.join(base_dir, "embedding_fitting_tukey_pvalue_heatmap.png")

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

# ========== STEP 1: Compute means if files don't exist ==========
if not os.path.exists(embed_out):
    compute_means(embed_csv, embed_out)
else:
    print(f"File exists, skipping: {embed_out}")

if not os.path.exists(combined_out):
    compute_means(combined_csv, combined_out)
else:
    print(f"File exists, skipping: {combined_out}")

# ========== STEP 2: Load both summaries and tag Analysis ==========
df_embed = pd.read_excel(embed_out)
df_embed["Analysis"] = "Embedding"

df_combined = pd.read_excel(combined_out)
df_combined["Analysis"] = "Embedding + Fitting"

# ========== STEP 3: Combine and reshape ==========
combined = pd.concat([df_embed, df_combined], ignore_index=True)
features = [c for c in combined.columns if c not in ['Experiment', 'Parent', 'Cluster', 'Analysis']]

long_df = combined.melt(
    id_vars=['Analysis', 'Cluster'],
    value_vars=features,
    var_name='Feature',
    value_name='Value'
).dropna(subset=['Value'])

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
anova_df['p_Analysis_FDR'] = multipletests(anova_df['p_Analysis'], method='fdr_bh')[1]
anova_df['p_Cluster_FDR'] = multipletests(anova_df['p_Cluster'], method='fdr_bh')[1]
anova_df['p_Interaction_FDR'] = multipletests(anova_df['p_Interaction'], method='fdr_bh')[1]
anova_df.to_csv(anova_out, index=False)
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
tukey_df.to_csv(tukey_out, index=False)
print(f"Saved Tukey results to: {tukey_out}")

# ========== STEP 6: Tukey heatmap ==========
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
plt.title('Tukey HSD Adjusted p-values by Feature & Cluster')
plt.tight_layout()
plt.savefig(heatmap_png, dpi=300)
print(f"Saved heatmap to: {heatmap_png}")

