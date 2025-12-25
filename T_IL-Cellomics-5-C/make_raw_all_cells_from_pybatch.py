import pandas as pd
import numpy as np

# ---------- CONFIG ----------
INPUT_EXCEL = f"cell_data/summary_table.xlsx"   # pybatch xlsx
OUTPUT_CSV  = f"cell_data/raw_all_cells.csv"    # file used by the pipeline

MIN_FRAMES_PER_CELL = 20
MAX_GAP             = 5   # max allowed gap in TimeIndex
# ----------------------------

print("loading pybatch table ...")
df = pd.read_excel(INPUT_EXCEL)

# drop old index columns like "Unnamed: 0"
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# make sure Experiment + Parent exist and are strings
df["Experiment"] = df["Experiment"].astype(str)
df["Parent"]     = df["Parent"].astype(str)

# unique cell id = Parent (track ID) + Experiment
df["unique_id"]  = df["Parent"] + "_" + df["Experiment"]

# build ds (fake datetime axis)
start = np.datetime64("2000-01-01T00:00:00")

# if dt is in minutes, use * 60; if already in seconds, remove the * 60
dt_seconds = df["dt"].iloc[0] * 60  # <-- change to df["dt"].iloc[0] if dt is already seconds

df["ds"] = start + (df["TimeIndex"] * dt_seconds).astype("timedelta64[s]")

# columns we want as metadata
meta_cols = ["Experiment", "Parent", "TimeIndex", "dt", "ds", "unique_id"]

# everything else (except ID, x_Pos, y_Pos) is a feature
feature_cols = [
    c for c in df.columns
    if c not in meta_cols and c not in ["ID", "x_Pos", "y_Pos"]
]

df = df[meta_cols + feature_cols]

# --- filtering like before ---

def valid_cell(group: pd.DataFrame) -> bool:
    # at least MIN_FRAMES_PER_CELL frames
    if len(group) < MIN_FRAMES_PER_CELL:
        return False
    # no gaps > MAX_GAP in TimeIndex
    t = group["TimeIndex"].sort_values().to_numpy()
    gaps = np.diff(t)
    return (gaps <= MAX_GAP).all()

print("filtering cells ...")
valid_ids = (
    df.groupby("unique_id")
      .filter(valid_cell)["unique_id"]
      .unique()
)

df_filtered = df[df["unique_id"].isin(valid_ids)].copy()

print(f"total rows before filtering: {len(df)}")
print(f"total rows after filtering : {len(df_filtered)}")
print(f"num unique cells kept      : {df_filtered['unique_id'].nunique()}")

df_filtered.to_csv(OUTPUT_CSV, index=False)
print(f"saved {OUTPUT_CSV}")
