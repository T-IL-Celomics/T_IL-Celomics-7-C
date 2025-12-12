import pandas as pd
import os
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Script: Aggregate & Perform Two-Way ANOVA + Tukey HSD
# ================================
base_dir       = "Embedding - K=3"
old_feat_csv   = os.path.join(base_dir, "Hala", "rawdatagraph.csv")
old_assign_csv = os.path.join(base_dir, "Hala", "linessplit_element0.csv")
new_feat_csv   = os.path.join(base_dir, "Merged_Clusters_PCA.csv")
old_out        = os.path.join(base_dir, 'old_populations_means.csv')
new_out        = os.path.join(base_dir, 'new_populations_means.csv')
anova_out      = os.path.join(base_dir, 'anova_two_way.csv')
tukey_out      = os.path.join(base_dir, 'tukey_two_way.csv')
heatmap_png    = os.path.join(base_dir, 'tukey_pvalue_heatmap.png')

os.makedirs(base_dir, exist_ok=True)

# 1. Aggregate Old Project Data
if os.path.exists(old_out):
    old_df = pd.read_csv(old_out)
else:
    df_feat = pd.read_csv(old_feat_csv)
    df_assign = pd.read_csv(old_assign_csv)[['Experiment','Treatment','Cluster']]
    if len(df_feat) != len(df_assign):
        raise ValueError(f"Row count mismatch: features={len(df_feat)}, assigns={len(df_assign)}")
    df_old = pd.concat([df_feat, df_assign], axis=1)
    df_old.drop(columns=['TimeIndex','x_Pos','y_Pos','Parent','dt','ID'], errors='ignore', inplace=True)
    old_df = df_old.groupby(['Experiment','Treatment','Cluster'], as_index=False).mean()
    old_df.to_csv(old_out, index=False)

# 2. Aggregate New Project Data
if os.path.exists(new_out):
    new_df = pd.read_csv(new_out)
else:
    new_df = pd.read_csv(new_feat_csv)
    new_df.drop(columns=['TimeIndex','Parent','dt','ID','PC1','PC2'], errors='ignore', inplace=True)
    for col in ['Experiment','Treatment','Cluster']:
        if col not in new_df.columns:
            raise KeyError(f"Missing '{col}' in new data")
    new_df = new_df.groupby(['Experiment','Treatment','Cluster'], as_index=False).mean()
    new_df.to_csv(new_out, index=False)

# 3. Two-Way ANOVA + Tukey HSD
old_df['Analysis'] = 'Old'
new_df['Analysis'] = 'New'
combined = pd.concat([old_df, new_df], ignore_index=True)
features = [c for c in combined.columns if c not in ['Experiment','Treatment','Cluster','Analysis']]
long_df = combined.melt(
    id_vars=['Analysis','Cluster'],
    value_vars=features,
    var_name='Feature', value_name='Value'
).dropna(subset=['Value'])

# ANOVA
anova_results = []
for feat, grp in long_df.groupby('Feature'):
    if grp['Analysis'].nunique() < 2:
        continue
    model = ols('Value ~ C(Analysis) * C(Cluster)', data=grp).fit()
    table = sm.stats.anova_lm(model, typ=2)
    anova_results.append({
        'Feature': feat,
        'p_Analysis': table.loc['C(Analysis)','PR(>F)'],
        'p_Cluster': table.loc['C(Cluster)','PR(>F)'],
        'p_Interaction': table.loc['C(Analysis):C(Cluster)','PR(>F)']
    })
anova_df = pd.DataFrame(anova_results)
anova_df['p_Analysis_FDR'] = multipletests(anova_df['p_Analysis'], method='fdr_bh')[1]
anova_df['p_Cluster_FDR'] = multipletests(anova_df['p_Cluster'], method='fdr_bh')[1]
anova_df['p_Interaction_FDR'] = multipletests(anova_df['p_Interaction'], method='fdr_bh')[1]
anova_df.to_csv(anova_out, index=False)

# Tukey HSD
tukey_results = []
for feat, grp in long_df.groupby('Feature'):
    for cl in sorted(grp['Cluster'].unique()):
        block = grp[grp['Cluster'] == cl]
        if block['Analysis'].nunique() < 2:
            continue
        tuk = pairwise_tukeyhsd(block['Value'], block['Analysis'])
        for row in tuk.summary().data[1:]:
            g1, g2, md, p_adj, lo, hi, rej = row
            if set([g1, g2]) == {'New', 'Old'}:
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

# 4. Heatmap of Tukey p-values with flipped colormap
heatmap_data = tukey_df.pivot(index='Feature', columns='Cluster', values='p_adj')
plt.figure(figsize=(10, max(6, len(heatmap_data)/4)))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap='coolwarm_r',
    cbar_kws={'label':'Adjusted p-value'},
    annot_kws={'color':'white', 'fontsize':7.5},
    mask=heatmap_data.isna()
)

plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.title('Tukey HSD Adjusted p-values by Feature & Cluster')
plt.tight_layout()
plt.savefig(heatmap_png, dpi=300)
print(f"Saved Tukey p-value heatmap to {heatmap_png}")
