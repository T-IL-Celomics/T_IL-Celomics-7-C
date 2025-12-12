import pandas as pd
import numpy as np
from scipy.stats import f_oneway

# === 1. Load your data ===
df = pd.read_csv("Merged_Clusters_PCA.csv")

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
styled.to_excel('ANOVA - OneWay.xlsx', engine='openpyxl')

print("✅ Saved 'ANOVA - OneWay.xlsx' with multi-level headers and p-value highlighting!")
