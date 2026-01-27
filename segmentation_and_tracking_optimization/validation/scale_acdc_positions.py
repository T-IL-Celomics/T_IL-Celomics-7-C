#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scale ACDC X/Y columns and optionally add fixed offsets (dx,dy),
then save a new CSV with the same name + "_corrected" before extension.

Default behavior (your current): multiply X,Y by 1.22.

Now added:
- --dx  : add offset to X AFTER scaling (e.g. +4)
- --dy  : add offset to Y AFTER scaling (e.g. -9)

Example:
  python3 scale_acdc_positions.py "field1_C2_acdc_output_full_try.csv" --scale 1.22 --dx 4 --dy -9 --channel NIR

Output:
  field1_C2_acdc_output_full_try_corrected.csv
"""

import argparse
import os
import re
import pandas as pd


def clean_colname(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def find_col_by_candidates(df: pd.DataFrame, candidates):
    nm = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in nm:
            return nm[key]
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Scale ACDC PositionX/PositionY columns and save *_corrected.csv (optionally add dx/dy)."
    )
    ap.add_argument("csv", help="Input CSV path")
    ap.add_argument("--scale", type=float, default=1.22, help="Scale factor (default 1.22)")
    ap.add_argument("--dx", type=float, default=0.0, help="Offset to add to X AFTER scaling (um). Default 0")
    ap.add_argument("--dy", type=float, default=0.0, help="Offset to add to Y AFTER scaling (um). Default 0")
    ap.add_argument(
        "--channel",
        default=None,
        help="Optional channel prefix (e.g. NIR). If not provided, script auto-detects columns.",
    )
    args = ap.parse_args()

    in_path = args.csv
    scale = float(args.scale)
    dx = float(args.dx)
    dy = float(args.dy)

    df = pd.read_csv(in_path, sep=None, engine="python")
    df.columns = [clean_colname(c) for c in df.columns]

    # Try channel-specific first if provided, else auto-detect common names.
    xcol = ycol = None
    if args.channel:
        xcol = find_col_by_candidates(df, [f"{args.channel}_PositionX_um"])
        ycol = find_col_by_candidates(df, [f"{args.channel}_PositionY_um"])

    if xcol is None:
        xcol = find_col_by_candidates(df, ["PositionX_um", "PosX_um", "X_um", "x_um"])
    if ycol is None:
        ycol = find_col_by_candidates(df, ["PositionY_um", "PosY_um", "Y_um", "y_um"])

    # If still None, try to find any "*PositionX_um" and "*PositionY_um" pair
    if xcol is None or ycol is None:
        cand_x = [c for c in df.columns if _norm(c).endswith("positionxum")]
        cand_y = [c for c in df.columns if _norm(c).endswith("positionyum")]
        if cand_x and cand_y:
            xcol = xcol or cand_x[0]
            ycol = ycol or cand_y[0]

    if xcol is None or ycol is None:
        raise SystemExit(
            "Could not find PositionX/PositionY columns.\n"
            f"Columns found (first 30): {list(df.columns)[:30]} ...\n"
            "Tip: try --channel NIR (or your channel name)."
        )

    # Convert to numeric then apply: scaled + offset
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce") * scale + dx
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce") * scale + dy

    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_corrected{ext if ext else '.csv'}"
    df.to_csv(out_path, index=False)

    print(f"Input:  {in_path}")
    print(f"Updated columns: {xcol}, {ycol}")
    print(f"Applied: X = X*{scale} + ({dx}),  Y = Y*{scale} + ({dy})")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
