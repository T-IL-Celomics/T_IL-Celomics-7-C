import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import time


# ───────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────
DATA_FILE = "MergedAndFilteredExperiment008.csv"
RAW_CSV = "raw_all_cells.csv"
NORM_CSV = "normalized_all_cells.csv"
FEATURE_LIST_FILE = "selected_features.txt"
NORMALIZE_BEFORE_VARIANCE = True
MAX_LAG = 25
# Thresholds and constants for filtering
PCA_VARIANCE = 0.95      # Or any float < 1 for explained variance
MIN_FEATURES = 5         # Minimum number of features to keep per step
CLUSTERS = 3             # For KMeans
CORR_THRESHOLD = 0.8     # Spearman correlation threshold for removing redundant features
CONST_CELL_THRESH = 0.99  # % of cells with zero std to define feature as "constant"
STD_KEEP_THRESH = 0.2   # Average std threshold for keeping variable features
# ───────────────────────────────────────────────────────────────────────────────

def drop_irrelevant_columns(df):
    for col in ("x_Pos", "y_Pos", "ID", 'Unnamed: 0'):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df

# ───────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 0: Load or Load-Saved Data ===")
# ───────────────────────────────────────────────────────────────────────────────
if os.path.exists(NORM_CSV) and NORMALIZE_BEFORE_VARIANCE:
    print(f"Loading normalized data from {NORM_CSV}")
    df = pd.read_csv(NORM_CSV)
else:
    # always start from raw_all_cells
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {RAW_CSV}")

    # if needed, normalize and save
    if NORMALIZE_BEFORE_VARIANCE:
        non_feat = {"unique_id", "Parent", "Experiment", "TimeIndex", "ds", "dt"}
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in non_feat]
        scaler = StandardScaler()
        df[feat_cols] = scaler.fit_transform(df[feat_cols])
        df.to_csv(NORM_CSV, index=False)
        print("Saved normalized data → normalized_all_cells.csv")

# Always define feat_cols
non_feat = {"unique_id", "Parent", "Experiment", "TimeIndex", "ds","dt"}
feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in non_feat]

# ───────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 1: PCA-Based Feature Selection ===")
# ───────────────────────────────────────────────────────────────────────────────
X = df[feat_cols].values
print(f"PCA input shape: {X.shape}")
print("Running PCA...")
start = time.time()
pca = PCA(n_components=PCA_VARIANCE, random_state=42)
X_pca = pca.fit_transform(X)
print(f"PCA completed in {time.time() - start:.2f} seconds.")

loadings = np.abs(pca.components_.T)
loading_sum = loadings.sum(axis=1)
FEATURE_STEPS = list(range(MIN_FEATURES, len(feat_cols)+1, 2))
eval_scores = []
for k in FEATURE_STEPS:
    print(f"→ Trying k = {k}")
    idxs = np.argsort(loading_sum)[-k:]
    subset = df[feat_cols].iloc[:, idxs].values
    km = KMeans(n_clusters=CLUSTERS, random_state=42, n_init=10).fit(subset)
    score = silhouette_score(subset, km.labels_, sample_size=50000, random_state=42)
    print(f"Silhouette score = {score}")
    eval_scores.append((k, score))

ks, scores = zip(*eval_scores)

# ───────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 2: Spearman Correlation Filter ===")
# ───────────────────────────────────────────────────────────────────────────────
best_k = ks[np.argmax(scores)]
best_idxs = np.argsort(loading_sum)[-best_k:]
best_features = [feat_cols[i] for i in best_idxs]
print(f"Best feature count (PCA step): {best_k}")
print(best_features)
corr_mat, _ = spearmanr(df[best_features], axis=0)
corr_mat = np.nan_to_num(corr_mat)
to_drop = set()
mean_abs = np.mean(np.abs(corr_mat), axis=1)
for i in range(len(best_features)):
    for j in range(i+1, len(best_features)):
        if abs(corr_mat[i, j]) > CORR_THRESHOLD:
            to_drop.add(i if mean_abs[i] > mean_abs[j] else j)
            break
final_features = [f for idx, f in enumerate(best_features) if idx not in to_drop]
print(f"After correlation filter: {len(final_features)} features")
print(final_features)

# ───────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 3: Filter Nearly Constant Features Across Cells ===")
# ───────────────────────────────────────────────────────────────────────────────
const_drop = []
total_cells = df['unique_id'].nunique()
for f in final_features:
    stds = df.groupby('unique_id')[f].std()
    frac_zero = (stds == 0).mean()
    mean_std = stds[stds > 0].mean() if (stds > 0).any() else 0
    if frac_zero >= CONST_CELL_THRESH and mean_std <= STD_KEEP_THRESH:
        const_drop.append(f)
final_features = [f for f in final_features if f not in const_drop]
print(f"After constant-cell filter: {len(final_features)} features")
print(final_features)

# ───────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 4: Include Mandatory Features ===")
# ───────────────────────────────────────────────────────────────────────────────
requested = [
    "Coll", "Instantaneous_Speed", "Directional_Change", "Displacement2",
    "Current_MSD_1", "Ellipticity_oblate", "Ellipticity_prolate",
    "EllipsoidAxisLengthB", "EllipsoidAxisLengthC", "Sphericity", "Eccentricity"
]
for feat in requested:
    if feat not in final_features and feat in df.columns:
        final_features.append(feat)
print(f"After adding requested features: {len(final_features)}")
print(final_features)

# Save selected features
with open(FEATURE_LIST_FILE, "w") as fh:
    fh.write("\n".join(final_features))
print(f"Saved selected features to {FEATURE_LIST_FILE}")


# ───────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 5: Heatmap of Selected Features ===")
# ───────────────────────────────────────────────────────────────────────────────

# Load selected features
with open(FEATURE_LIST_FILE, "r") as f:
    selected_features = f.read().splitlines()

# Ensure figure output dir exists
os.makedirs("figures", exist_ok=True)

# Sample 100 cells and compute mean of selected features
sampled_cells = np.random.choice(df["unique_id"].unique(), size=min(100, df["unique_id"].nunique()), replace=False)
heat_df = df[df["unique_id"].isin(sampled_cells)].groupby("unique_id")[selected_features].mean()

plt.figure(figsize=(12, 8))
plt.imshow(heat_df.values, aspect='auto', cmap='viridis')
plt.yticks(np.arange(len(heat_df)), heat_df.index, fontsize=6)
plt.xticks(np.arange(len(selected_features)), selected_features, rotation=90, fontsize=8)
plt.colorbar(label='Feature value (mean)')
plt.title('Heatmap of final features')
plt.tight_layout()
plt.savefig("figures/heatmap_final_features.png", dpi=300)
print("Saved heatmap to figures/heatmap_final_features.png")
plt.show()

# ───────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 6: Temporal Autocorrelation for All Features (excluding metadata) ===")
# ───────────────────────────────────────────────────────────────────────────────
metadata_cols = {"unique_id", "Parent", "Experiment", "TimeIndex", "ds","dt"}
all_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in metadata_cols]

tm = df.groupby("TimeIndex")[all_features].mean()
lag_vals = list(range(1, MAX_LAG + 1))
acf = np.zeros((len(all_features), MAX_LAG))

for i, f in enumerate(all_features):
    for lag in lag_vals:
        acf[i, lag - 1] = tm[f].autocorr(lag=lag)

plt.figure(figsize=(12, 8))
plt.imshow(acf, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
plt.yticks(np.arange(len(all_features)), all_features, fontsize=8)
plt.xticks(np.arange(len(lag_vals)), lag_vals, rotation=90, fontsize=10)
plt.xlabel('Lag (frames)', fontsize=14)
plt.colorbar(label='Autocorrelation')
plt.title('Autocorrelation Heatmap (All Features)', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("figures/autocorr_heatmap_all_features.png", dpi=300)
print("📊 Saved autocorrelation heatmap to figures/autocorr_heatmap_all_features.png")
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 6: Temporal Autocorrelation ===")
# ───────────────────────────────────────────────────────────────────────────────
tm = df.groupby("TimeIndex")[selected_features].mean()
lag_vals = list(range(1, MAX_LAG + 1))
acf = np.zeros((len(selected_features), MAX_LAG))

for i, f in enumerate(selected_features):
    for lag in lag_vals:
        acf[i, lag - 1] = tm[f].autocorr(lag=lag)

plt.figure(figsize=(12, 8))
plt.imshow(acf, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
plt.yticks(np.arange(len(selected_features)), selected_features, fontsize=8)
plt.xticks(np.arange(len(lag_vals)), lag_vals, rotation=90, fontsize=10)
plt.xlabel('Lag (frames)', fontsize=14)
plt.colorbar(label='Autocorrelation')
plt.title('Heatmap', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("figures/autocorr_heatmap.png", dpi=300)
print("Saved autocorrelation heatmap to figures/autocorr_heatmap.png")
plt.show()

# ───────────────────────────────────────────────────────────────────────────────
# STEP 7: Silhouette Plot
# ───────────────────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(ks, scores, marker='o')
plt.xlabel('Number of features')
plt.ylabel('Silhouette score')
plt.title('Feature count vs. silhouette')
plt.savefig("figures/silhouette_plot.png", dpi=300)
plt.show()

# ───────────────────────────────────────────────────────────────────────────────
# Save final features
# ───────────────────────────────────────────────────────────────────────────────
with open("cell_data/selected_features.txt", "w") as fh:
    fh.write("\n".join(final_features))
print("\nSaved selected features → selected_features.txt")
