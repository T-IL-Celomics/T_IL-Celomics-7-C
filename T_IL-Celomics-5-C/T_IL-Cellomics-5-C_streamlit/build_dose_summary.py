import pandas as pd
from glob import glob
from collections import Counter

# 1) find all relevant Excel files
# if they are in a subfolder like "Gab/", change to "Gab/Gab_Normalized_Combined_*.xlsx"
paths = glob("Gab_Normalized_Combined_*.xlsx")

print("found files:")
for p in paths:
    print("  ", p)

all_rows = []

for path in paths:
    # each file has sheets: Area, ConvexHull, etc.
    # here we start with Area; later we can repeat for other features if you want
    df = pd.read_excel(path, sheet_name="Area")

    # make sure we have Experiment column; if not, create it from filename
    if "Experiment" not in df.columns:
        # example: Gab_Normalized_Combined_c2.xlsx -> "c2"
        exp_name = path.split("Gab_Normalized_Combined_")[-1].split(".xlsx")[0]
        df["Experiment"] = exp_name

    # keep only columns we care about
    base_cols = ["Experiment", "Parent", "Time"]
    channel_cols = [c for c in df.columns if "Cha" in c and ("Norm" in c or "Category" in c)]

    df = df[base_cols + channel_cols]
    all_rows.append(df)

# 2) stack all wells together
expr = pd.concat(all_rows, ignore_index=True)

# 3) helper for majority label
def majority(series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return Counter(s).most_common(1)[0][0]


WELL_TO_FULL = {
    "B2": "AM001100425CHR2B02293TNNIRNOCONNN0NNN0NNN0WH00",
    "B3": "AM001100425CHR2B03293TNNIRNOCONNN0NNN0NNN0WH00",
    "B4": "AM001100425CHR2B04293TNNIRNOCONNN0NNN0NNN0WH00",
    "C2": "AM001100425CHR2C02293TMETRNNIRNOCONNN0NNN0WH00",
    "C3": "AM001100425CHR2C03293TMETRNNIRNOCONNN0NNN0WH00",
    "C4": "AM001100425CHR2C04293TMETRNNIRNOCONNN0NNN0WH00",
    "D2": "AM001100425CHR2D02293TGABYNNIRNOCONNN0NNN0WH00",
    "D3": "AM001100425CHR2D03293TGABYNNIRNOCONNN0NNN0WH00",
    "D4": "AM001100425CHR2D04293TGABYNNIRNOCONNN0NNN0WH00",
    "E2": "AM001100425CHR2E02293TNNIRMETRGABYNOCONNN0WH00",
    "E3": "AM001100425CHR2E03293TNNIRMETRGABYNOCONNN0WH00",
    "E4": "AM001100425CHR2E04293TNNIRMETRGABYNOCONNN0WH00",
}

expr["Experiment"] = expr["Experiment"].replace(WELL_TO_FULL)


# 4) aggregate per (Experiment, Parent)
agg_dict = {"Time": "count"}   # becomes n_frames

for col in expr.columns:
    if col.endswith("_Norm"):
        agg_dict[col] = "mean"
    elif col.endswith("_Category"):
        agg_dict[col] = majority

summary = (
    expr
    .groupby(["Experiment", "Parent"])
    .agg(agg_dict)
    .reset_index()
    .rename(columns={"Time": "n_frames"})
)

summary.to_csv("dose_dependency_summary_all_wells.csv", index=False)
print("saved dose_dependency_summary_all_wells.csv")
