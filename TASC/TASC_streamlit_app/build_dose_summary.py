"""
build_dose_summary.py
=====================
Build a per-cell dose-dependency summary CSV from the Normalised / Categorised
Excel files produced by the ``filtering_by_protein_expression`` pipeline.

Each input Excel file (``Gab_Normalized_Combined_*.xlsx``) has one sheet per
Imaris-exported parameter (Area, Position, …). Every sheet contains per-cell,
per-time-point rows with columns:
    Experiment, Parent, Time, Cha1_Norm, Cha1_Category, …

This script:
  1. Scans a directory for matching workbooks.
  2. Reads one sheet from each (default: ``Area``).
  3. Aggregates per (Experiment, Parent):
       • ``Cha*_Norm``  → mean
       • ``Cha*_Category`` → majority vote
       • ``Time``  → count  (renamed to ``n_frames``)
  4. Optionally maps short well names → full experiment IDs via a JSON file.
  5. Writes the result to a CSV.

Usage
-----
Standalone::

    python build_dose_summary.py \\
        --dir  D:\\jeries\\output \\
        --glob "Gab_Normalized_Combined_*.xlsx" \\
        --sheet Area \\
        --prefix "Gab_Normalized_Combined_" \\
        --output dose_dependency_summary_all_wells.csv

Or called programmatically from the TASC GUI.

All arguments are optional — sensible defaults are used when omitted.
"""

import os
import sys
import json
import argparse
import pandas as pd
from glob import glob
from collections import Counter
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────

def _majority(series):
    """Return the most common non-null value in *series*."""
    s = series.dropna()
    if len(s) == 0:
        return None
    return Counter(s).most_common(1)[0][0]


def _load_well_map(path: str) -> dict:
    """Load a JSON well-name → full-experiment-ID mapping (optional)."""
    if path and Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ── main entry point ─────────────────────────────────────────────────────

def build_dose_summary(
    dose_dir: str = "",
    excel_glob: str = "Gab_Normalized_Combined_*.xlsx",
    sheet_name: str = "Area",
    file_prefix: str = "Gab_Normalized_Combined_",
    output_csv: str = "dose_dependency_summary_all_wells.csv",
    well_map_path: str = "",
) -> pd.DataFrame:
    """
    Read matching Excel files, aggregate per cell, and write a summary CSV.

    Parameters
    ----------
    dose_dir : str
        Directory to search.  Defaults to cwd.
    excel_glob : str
        Glob pattern for the Normalised-Combined files.
    sheet_name : str
        Which sheet to read inside each workbook.
    file_prefix : str
        Filename prefix stripped to derive the Experiment name when the
        workbook does not already contain an ``Experiment`` column.
    output_csv : str
        Path for the output CSV.
    well_map_path : str
        Optional path to a JSON ``{short_name: full_experiment_id}`` mapping.

    Returns
    -------
    pd.DataFrame  — the summary table that was written to *output_csv*.
    """

    # 1) Find matching Excel files
    if dose_dir:
        search_pattern = os.path.join(dose_dir, excel_glob)
    else:
        search_pattern = excel_glob
    paths = sorted(glob(search_pattern))

    print(f"Search dir   : {dose_dir or '(current working directory)'}")
    print(f"Glob pattern : {excel_glob}")
    print(f"Full pattern : {search_pattern}")
    print(f"Sheet        : {sheet_name}")
    print(f"Found {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    if not paths:
        raise FileNotFoundError(
            f"No files matched the glob pattern '{excel_glob}' "
            f"in '{dose_dir or os.getcwd()}'.\n"
            "Check the pattern and working directory."
        )

    all_rows = []

    for path in paths:
        df = pd.read_excel(path, sheet_name=sheet_name)

        # Make sure we have an Experiment column; if not, derive from filename
        if "Experiment" not in df.columns:
            fname = os.path.basename(path)
            exp_name = fname.replace(file_prefix, "").replace(".xlsx", "")
            df["Experiment"] = exp_name

        # Keep only the columns we care about
        base_cols = [c for c in ["Experiment", "Parent", "Time"] if c in df.columns]
        channel_cols = [
            c for c in df.columns
            if "Cha" in c and ("Norm" in c or "Category" in c)
        ]
        keep = base_cols + channel_cols
        df = df[[c for c in keep if c in df.columns]]
        all_rows.append(df)

    # 2) Stack all wells together
    expr = pd.concat(all_rows, ignore_index=True)

    # 3) Optionally map short well names → full experiment IDs
    well_map = _load_well_map(well_map_path)
    if well_map:
        expr["Experiment"] = expr["Experiment"].replace(well_map)

    # 4) Aggregate per (Experiment, Parent)
    agg_dict = {}
    if "Time" in expr.columns:
        agg_dict["Time"] = "count"  # will become n_frames
    for col in expr.columns:
        if col.endswith("_Norm"):
            agg_dict[col] = "mean"
        elif col.endswith("_Category"):
            agg_dict[col] = _majority

    summary = (
        expr
        .groupby(["Experiment", "Parent"])
        .agg(agg_dict)
        .reset_index()
    )
    if "Time" in summary.columns:
        summary.rename(columns={"Time": "n_frames"}, inplace=True)

    summary.to_csv(output_csv, index=False)
    print(f"\n✅ Saved {output_csv}  ({len(summary)} rows)")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build per-cell dose-dependency summary CSV."
    )
    parser.add_argument("--dir", default="",
                        help="Directory containing Normalised-Combined Excel files")
    parser.add_argument("--glob", default="Gab_Normalized_Combined_*.xlsx",
                        help="Glob pattern for the Excel files")
    parser.add_argument("--sheet", default="Area",
                        help="Sheet name to read inside each workbook")
    parser.add_argument("--prefix", default="Gab_Normalized_Combined_",
                        help="Filename prefix stripped to derive Experiment name")
    parser.add_argument("--output", default="dose_dependency_summary_all_wells.csv",
                        help="Output CSV path")
    parser.add_argument("--well-map", default="",
                        help="Path to a JSON well-name → experiment-ID mapping")
    args = parser.parse_args()

    build_dose_summary(
        dose_dir=args.dir,
        excel_glob=args.glob,
        sheet_name=args.sheet,
        file_prefix=args.prefix,
        output_csv=args.output,
        well_map_path=args.well_map,
    )


if __name__ == "__main__":
    main()
