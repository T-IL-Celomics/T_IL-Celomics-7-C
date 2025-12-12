import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---- paths ----
INPUT = "raw_all_cells.csv"
OUT_CLEAN = "raw_all_cells_clean.csv"
OUT_SCALED = "raw_all_cells_scaled_for_embedding.csv"

# ---- load ----
df = pd.read_csv(INPUT)

# 1) drop automatic index columns from excel/csv
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

# 2) make sure ds is proper datetime
#    (1/1/2000 8:00 style -> this will parse correctly)
df["ds"] = pd.to_datetime(df["ds"])

# 3) sort by cell track and time
if "unique_id" in df.columns:
    df = df.sort_values(["unique_id", "ds"])
else:
    raise ValueError("unique_id column is missing – needed for grouping per cell")

# 4) separate metadata vs numeric feature columns
meta_cols = ["Experiment", "Parent", "TimeIndex", "dt", "ds", "unique_id"]
meta_cols = [c for c in meta_cols if c in df.columns]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in meta_cols]

# 5) drop constant features (no variance -> useless + can break some models)
nunique = df[feature_cols].nunique()
keep_feature_cols = nunique[nunique > 1].index.tolist()
removed_const = sorted(set(feature_cols) - set(keep_feature_cols))
print("removed constant features:", removed_const)

# 6) fill NaNs per cell track (grouped by unique_id)
def fill_group(g: pd.DataFrame) -> pd.DataFrame:
    # interpolate along time within each cell
    g[keep_feature_cols] = g[keep_feature_cols].interpolate(
        limit_direction="both"
    )
    # then forward/backward fill remaining holes if any
    g[keep_feature_cols] = (
        g[keep_feature_cols]
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
    return g

df = df.groupby("unique_id", group_keys=False).apply(fill_group)

# if there are still NaNs (super rare), drop those rows
before = len(df)
df = df.dropna(subset=keep_feature_cols)
after = len(df)
print(f"dropped {before - after} rows with remaining NaNs in feature columns")

# 7) create clean (unscaled) version for forecasting
cols_order = meta_cols + keep_feature_cols
df_clean = df[cols_order].copy()
df_clean.to_csv(OUT_CLEAN, index=False)
print(f"wrote {OUT_CLEAN} with shape {df_clean.shape}")

# 8) create scaled version for embeddings (standardize each feature)
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df_clean[keep_feature_cols])

df_scaled = df_clean.copy()
df_scaled[keep_feature_cols] = scaled_values
df_scaled.to_csv(OUT_SCALED, index=False)
print(f"wrote {OUT_SCALED} with shape {df_scaled.shape}")
