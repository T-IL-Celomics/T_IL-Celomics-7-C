"""
build_dose_summary.py
=====================
Build a per-cell dose-dependency summary from normalised Excel exports.

Pipeline environment variables (all optional — sensible defaults used):
  PIPELINE_DOSE_EXCEL_GLOB   Glob pattern for input Excel files
                             (default: "Gab_Normalized_Combined_*.xlsx")
  PIPELINE_DOSE_SHEET        Sheet name to read in each workbook
                             (default: "Area")
  PIPELINE_DOSE_PREFIX       Filename prefix used to derive Experiment name
                             (default: "Gab_Normalized_Combined_")
  PIPELINE_DOSE_OUTPUT       Output CSV path
                             (default: "dose_dependency_summary_all_wells.csv")
  PIPELINE_DOSE_WELL_MAP     Path to a JSON file mapping short well names to
                             full experiment IDs.  If not set the built-in
                             mapping is used.
  PIPELINE_DOSE_DIR          Base directory in which to search for Excel files.
                             The glob pattern is resolved relative to this dir.
                             (default: current working directory)
"""

import os
import json
import pandas as pd
from glob import glob
from collections import Counter
from pathlib import Path

# ── Resolve parameters from environment or defaults ─────────────────────
DOSE_DIR    = os.environ.get("PIPELINE_DOSE_DIR", "").strip()
EXCEL_GLOB  = os.environ.get("PIPELINE_DOSE_EXCEL_GLOB", "Gab_Normalized_Combined_*.xlsx")
SHEET_NAME  = os.environ.get("PIPELINE_DOSE_SHEET", "Area")
FILE_PREFIX = os.environ.get("PIPELINE_DOSE_PREFIX", "Gab_Normalized_Combined_")
OUTPUT_CSV  = os.environ.get("PIPELINE_DOSE_OUTPUT", "dose_dependency_summary_all_wells.csv")
WELL_MAP_PATH = os.environ.get("PIPELINE_DOSE_WELL_MAP", "")

# ── Built-in well → full experiment-ID mapping (used when no JSON given) ─
_DEFAULT_WELL_MAP = {
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
    "E4": "AM001100425CHR2E04293TNNIRMETRGABYNOCONNN0WH00"
}


def _load_well_map() -> dict:
    """Return the well-to-full-ID mapping, from JSON file or built-in."""
    if WELL_MAP_PATH and Path(WELL_MAP_PATH).exists():
        with open(WELL_MAP_PATH, "r") as f:
            return json.load(f)
    return _DEFAULT_WELL_MAP


def majority(series):
    """Return the most common non-null value in *series*."""
    s = series.dropna()
    if len(s) == 0:
        return None
    return Counter(s).most_common(1)[0][0]


def build_dose_summary():
    """Main entry point — reads Excel files, aggregates, and writes CSV."""

    # 1) find all relevant Excel files
    if DOSE_DIR:
        search_pattern = os.path.join(DOSE_DIR, EXCEL_GLOB)
    else:
        search_pattern = EXCEL_GLOB
    paths = sorted(glob(search_pattern))

    print(f"Search dir   : {DOSE_DIR or '(current working directory)'}")
    print(f"Glob pattern : {EXCEL_GLOB}")
    print(f"Full pattern : {search_pattern}")
    print(f"Sheet        : {SHEET_NAME}")
    print(f"Found {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    if not paths:
        raise FileNotFoundError(
            f"No files matched the glob pattern '{EXCEL_GLOB}'. "
            "Check the pattern and working directory."
        )

    all_rows = []

    for path in paths:
        df = pd.read_excel(path, sheet_name=SHEET_NAME)

        # make sure we have Experiment column; if not, create it from filename
        if "Experiment" not in df.columns:
            exp_name = path.split(FILE_PREFIX)[-1].split(".xlsx")[0]
            df["Experiment"] = exp_name

        # keep only columns we care about
        base_cols = ["Experiment", "Parent", "Time"]
        channel_cols = [
            c for c in df.columns
            if "Cha" in c and ("Norm" in c or "Category" in c)
        ]
        df = df[base_cols + channel_cols]
        all_rows.append(df)

    # 2) stack all wells together
    expr = pd.concat(all_rows, ignore_index=True)

    # 3) map short well names → full experiment IDs
    well_map = _load_well_map()
    expr["Experiment"] = expr["Experiment"].replace(well_map)

    # 4) aggregate per (Experiment, Parent)
    agg_dict = {"Time": "count"}  # becomes n_frames
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

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {OUTPUT_CSV}  ({len(summary)} rows)")

    # Also copy to cell_data/ so downstream scripts find it by default
    _cell_data_copy = os.path.join("cell_data", os.path.basename(OUTPUT_CSV))
    if os.path.isdir("cell_data"):
        import shutil
        shutil.copy2(OUTPUT_CSV, _cell_data_copy)
        print(f"   ↳ Copied to {_cell_data_copy}")

    return summary


if __name__ == "__main__":
    build_dose_summary()
