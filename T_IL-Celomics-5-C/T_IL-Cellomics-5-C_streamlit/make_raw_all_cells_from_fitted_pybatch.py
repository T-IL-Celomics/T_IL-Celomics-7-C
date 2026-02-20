
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
# IMPORTANT: point this to your FILLED table
INPUT_CSV  = "cell_data/summary_table_filled_no_extrap_FINAL_NO_NAN.csv"
OUTPUT_CSV = "cell_data/raw_all_cells.csv"

# since you already filled inbound, don't do gap filtering anymore
MIN_FRAMES_PER_CELL = 15          # set to 20/25 if you still want to drop very short tracks
DROP_DUP_TIMEINDEX  = True       # drop duplicate TimeIndex within a cell (keeps first)
# ----------------------------

print(f"loading filled table: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# drop old index columns like "Unnamed: 0"
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

# required cols
required = ["Experiment", "Parent", "TimeIndex", "dt"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"missing required columns: {missing}")

# normalize keys
df["Experiment"] = df["Experiment"].astype(str)
df["Parent"] = df["Parent"].astype(str)

# TimeIndex as int (1..48)
df["TimeIndex"] = pd.to_numeric(df["TimeIndex"], errors="coerce").astype("Int64")
df = df.dropna(subset=["TimeIndex"])
df["TimeIndex"] = df["TimeIndex"].astype(int)

# unique cell id
df["unique_id"] = df["Parent"] + "_" + df["Experiment"]

# build ds (fake datetime axis)
start = np.datetime64("2000-01-01T00:00:00")

# dt can vary per row; we'll compute per-row timedelta in seconds
dt_min = pd.to_numeric(df["dt"], errors="coerce").fillna(method="ffill").fillna(method="bfill")
dt_seconds = (dt_min.astype(float) * 60.0).to_numpy()  # assumes dt is minutes

# ds = start + TimeIndex * dt_seconds
# (cast to seconds then to timedelta64[s])
ds_seconds = (df["TimeIndex"].to_numpy(dtype=float) * dt_seconds).round().astype(np.int64)
df["ds"] = start + ds_seconds.astype("timedelta64[s]")

# ---- choose metadata + feature columns ----
meta_cols = ["Experiment", "Parent", "TimeIndex", "dt", "ds", "unique_id"]
drop_as_features = {"ID", "x_Pos", "y_Pos"}

feature_cols = [
    c for c in df.columns
    if c not in meta_cols and c not in drop_as_features and not str(c).startswith("Unnamed")
]

# keep only numeric feature cols (protect from accidental strings)
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# sort
df = df.sort_values(["unique_id", "TimeIndex"]).reset_index(drop=True)

# optionally remove duplicate TimeIndex per cell
if DROP_DUP_TIMEINDEX:
    before = len(df)
    df = df.drop_duplicates(subset=["unique_id", "TimeIndex"], keep="first")
    after = len(df)
    print(f"dropped {before - after} duplicate (unique_id, TimeIndex) rows")

# optional length filter (NO gap filter because data is already filled)
if MIN_FRAMES_PER_CELL and MIN_FRAMES_PER_CELL > 1:
    sizes = df.groupby("unique_id").size()
    keep_ids = sizes[sizes >= MIN_FRAMES_PER_CELL].index
    before = len(df)
    df = df[df["unique_id"].isin(keep_ids)].copy()
    after = len(df)
    print(f"dropped {before - after} rows due to MIN_FRAMES_PER_CELL={MIN_FRAMES_PER_CELL}")

# final sanity prints
print("final rows:", len(df))
print("unique cells:", df["unique_id"].nunique())
sizes2 = df.groupby("unique_id").size()
print("frames per cell (min/median/max):",
      int(sizes2.min()), float(sizes2.median()), int(sizes2.max()))

# write
out = df[meta_cols + feature_cols].copy()
out.to_csv(OUTPUT_CSV, index=False)
print(f"saved {OUTPUT_CSV} with shape {out.shape}")
