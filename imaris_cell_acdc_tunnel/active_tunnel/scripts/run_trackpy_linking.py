#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trackpy_linking.py
======================
Runs TrackPy gap-closing linking on ACDC segmentation output and produces
the TrackPy CSV required by acdc_npz_tif_to_imaris_like.py / export_imaris_like_from_pipeline.py.

WHY THIS SCRIPT EXISTS
──────────────────────
ACDC segments cells frame-by-frame and assigns a per-frame Cell_ID (label).
If a cell's segmentation is missed for 1–2 frames, ACDC treats it as two
separate tracks — a short one before the gap and a new one after.

Imaris bridges these gaps using its own track-linking algorithm, so a cell
that disappears for 2 frames is still counted as ONE long track.

This script re-links ACDC's per-frame detections using TrackPy's nearest-
neighbour linker with gap-closing (the `memory` parameter). It outputs a CSV
mapping every (frame, Cell_ID) pair to a persistent `particle` ID.

When that CSV is passed to export_imaris_like_from_pipeline.py via
--trackpy-csv (or placed next to the TIF so it is auto-detected), the Parent
column in the output Excel becomes a persistent track ID that matches Imaris.

OUTPUT FORMAT
─────────────
CSV with columns:  frame | id | particle | x | y
  frame    : 0-based frame index  (matches frame_i0 in the converter)
  id       : ACDC Cell_ID for this detection in this frame
  particle : persistent track ID assigned by TrackPy (gap-closed)
  x, y     : centroid position in pixels (kept for debugging / QC)

USAGE
─────
# Run on all positions:
    python run_trackpy_linking.py --exp_root segmentation_input --all

# Run on one position:
    python run_trackpy_linking.py --exp_root segmentation_input --select field1_B3_1_NIR

# Custom linking parameters:
    python run_trackpy_linking.py --exp_root segmentation_input --all \\
        --search-range-px 20 --memory 3 --min-frames 5

HIGH-DENSITY / FAST-MOVING CELLS — VELOCITY PREDICTOR
───────────────────────────────────────────────────────
Standard TrackPy uses a fixed search circle centred on the cell's CURRENT
position.  This breaks when cells move faster than the inter-cell spacing
(a common situation in confluent monolayers imaged at 1 h/frame), because
the search circle then contains many other cells and the Hungarian solver
either picks the wrong one or explodes with "subnetwork too large".

The fix is to use trackpy.predict.NearestVelocityPredict, which shifts the
search circle to the cell's PREDICTED next position based on its recent
velocity.  The search radius then only needs to cover the RESIDUAL error
after prediction (typically 10-15 px), not the full displacement.

--use-predictor  is ON by default.  Disable with --no-predictor only if
cells move very slowly (< inter-cell distance per frame) or if tracks are
too short for reliable velocity estimation (<3 frames).

PARAMETERS — how to set them
──────────────────────────────
--search-range-px  (default 15)
    With predictor ON  : residual search radius after velocity prediction.
                         10-20 px is appropriate for most datasets.
    With predictor OFF : max total displacement per frame.
                         Must be > actual per-frame displacement, which
                         causes subnetwork explosion in dense fields.

--predictor-span  (default 3)
    Number of past frames used to estimate each cell's velocity.
    span=1 → uses only the last step (noisy but responsive).
    span=3 → averages 3 steps (smoother, better for steady migration).
    For the first `span` frames of each track, predictor falls back to
    standard linking automatically.

--memory  (default 3)
    Maximum number of frames a cell may disappear and still be re-linked.
    This is the gap-closing parameter.  Set to 3 to bridge short detection
    gaps.  Too high → unrelated detections linked together.

--min-frames  (default 20)
    Tracks shorter than this are dropped from the output.  Filters out
    spurious single-frame or very-short detections.  Set to 1 to keep all.
    For datasets where Imaris tracks span nearly the full acquisition,
    set this close to the total frame count (e.g. 20 for a 47-frame movie).

WHERE THE OUTPUT IS SAVED
──────────────────────────
For each position the CSV is written to:
    <exp_root>/Position_X/Images/<field>_<location>_trackpy_tracks.csv

This is exactly where export_imaris_like_from_pipeline.py auto-detects it
(first search location in find_trackpy_csv()).  No --trackpy-csv flag needed.

REQUIREMENTS
─────────────
    pip install trackpy scipy
    (numpy, pandas, tifffile, scikit-image are already required by pipeline)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import trackpy as tp
    tp.quiet()  # suppress trackpy progress spam
except ImportError:
    sys.exit("[ERROR] TrackPy not installed.  Run:  pip install trackpy")

try:
    from skimage.measure import regionprops
    import tifffile as tiff
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}\n  Run: pip install scikit-image tifffile")


# ─────────────────────────────────────────────────────────────────
# Position map helpers (shared interface with other pipeline scripts)
# ─────────────────────────────────────────────────────────────────

def _k(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

def _v(s: str) -> str:
    return (s or "").strip()

def get_first(row: Dict[str, str], *keys: str, default: str = "") -> str:
    for kk in keys:
        val = row.get(_k(kk), "")
        if val:
            return val
    return default

def find_mapping_file(exp_root: Path, user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            p = exp_root / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Mapping file not found: {p}")
        return p
    for name in ["position_map.csv", "positions_index.csv", "position_map.json", "positions_index.json"]:
        c = exp_root / name
        if c.exists():
            return c
    csvs = sorted([p.name for p in exp_root.glob("*.csv")])
    raise FileNotFoundError(
        f"No mapping file found in: {exp_root}\n"
        f"Found CSVs: {csvs if csvs else 'NONE'}\n"
        f"Pass explicitly with --map <file>"
    )

def load_mapping(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [{_k(str(k)): _v(str(v)) for k, v in r.items()} for r in data]
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({_k(h): _v(r.get(h, "")) for h in (reader.fieldnames or [])})
    return rows

def position_folder_from_row(row: Dict[str, str]) -> str:
    pf = get_first(row, "position_folder", "position_dir", "position_name")
    if pf:
        return pf
    num = get_first(row, "position_number", "position")
    if num.isdigit():
        return f"Position_{int(num)}"
    raw = get_first(row, "Position")
    if raw.lower().startswith("position_"):
        return raw
    raise KeyError(f"Cannot determine position folder from row: {sorted(row.keys())}")

def filter_rows(rows: List[Dict[str, str]], select: str, run_all: bool,
                channel_only: str) -> List[Dict[str, str]]:
    if run_all:
        return rows
    if channel_only:
        ch = _k(channel_only)
        return [r for r in rows if _k(get_first(r, "channel", "channel_name", "ch")) == ch]
    if not select:
        raise SystemExit("Use --all, --channel, or --select")

    sel = select.strip()
    if "|" in sel:
        parts = [p.strip() for p in sel.split("|")]
        if len(parts) != 3:
            raise SystemExit("Bad --select. Use: field1|B3_1|NIR")
        ff, fn, ch = parts
        return [r for r in rows
                if _k(get_first(r, "field_folder", "field", "parent_folder")) == _k(ff)
                and _k(get_first(r, "field_name")) == _k(fn)
                and _k(get_first(r, "channel", "channel_name", "ch")) == _k(ch)]

    toks = sel.split("_")
    if len(toks) < 3:
        raise SystemExit("Bad --select. Use: field1_B3_1_NIR or field1|B3_1|NIR")

    ff = toks[0]
    ch = toks[-1]
    if len(toks) == 4:
        fn = f"{toks[1]}_{toks[2]}"
        return [r for r in rows
                if _k(get_first(r, "field_folder", "field", "parent_folder")) == _k(ff)
                and _k(get_first(r, "field_name")) == _k(fn)
                and _k(get_first(r, "channel", "channel_name", "ch")) == _k(ch)]

    loc = toks[1]
    return [r for r in rows
            if _k(get_first(r, "field_folder", "field", "parent_folder")) == _k(ff)
            and _k(get_first(r, "location", "letter", "site")) == _k(loc)
            and _k(get_first(r, "channel", "channel_name", "ch")) == _k(ch)]


# ─────────────────────────────────────────────────────────────────
# Centroid extraction
# ─────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        hit = cols.get(c.lower().strip())
        if hit is not None:
            return hit
    return None


def centroids_from_acdc_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Read centroids from an ACDC *_acdc_output.csv.

    Returns DataFrame: frame(int), id(int), x(float), y(float)
    Returns None if the file doesn't have the required columns.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    frame_col = _find_col(df, "frame_i", "frame", "t", "time", "Time")
    id_col    = _find_col(df, "Cell_ID", "cell_id", "track_id", "ID")
    x_col     = _find_col(df, "x_centroid", "centroid_x", "x", "X")
    y_col     = _find_col(df, "y_centroid", "centroid_y", "y", "Y")

    if None in (frame_col, id_col, x_col, y_col):
        missing = [n for n, c in zip(["frame","id","x","y"],
                                      [frame_col, id_col, x_col, y_col]) if c is None]
        print(f"  [warn] ACDC CSV missing columns: {missing} — will fall back to NPZ")
        return None

    out = pd.DataFrame({
        "frame": pd.to_numeric(df[frame_col], errors="coerce").astype("Int64"),
        "id":    pd.to_numeric(df[id_col],    errors="coerce").astype("Int64"),
        "x":     pd.to_numeric(df[x_col],     errors="coerce"),
        "y":     pd.to_numeric(df[y_col],     errors="coerce"),
    }).dropna().reset_index(drop=True)
    out["frame"] = out["frame"].astype(int)
    out["id"]    = out["id"].astype(int)
    return out


def centroids_from_npz(npz_path: Path) -> pd.DataFrame:
    """
    Compute centroids from the segmentation NPZ via regionprops.
    Used when ACDC CSV is missing or incomplete.

    Returns DataFrame: frame(int), id(int), x(float), y(float)
    """
    print("  [info] computing centroids from NPZ segmentation (this may take a moment)...")

    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.keys())

    arr = None
    for k in ["segm", "masks", "labels", "mask", "arr_0"] + keys:
        a = z[k]
        if isinstance(a, np.ndarray) and a.size > 0:
            arr = a
            break
    if arr is None:
        raise RuntimeError(f"Cannot find segmentation array in {npz_path}. Keys: {keys}")

    if arr.dtype == object:
        frames_list = [np.asarray(arr[i]) for i in range(len(arr))]
        arr = np.stack(frames_list, axis=0)

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32)

    # If 4D (T,Z,Y,X) use max-projection for centroid
    if arr.ndim == 4:
        arr2d = arr.max(axis=1)
    elif arr.ndim == 3:
        arr2d = arr
    else:
        raise ValueError(f"Unexpected segm shape: {arr.shape}")

    rows = []
    for t, frame in enumerate(arr2d):
        for rp in regionprops(frame.astype(np.int32)):
            cy, cx = rp.centroid
            rows.append({"frame": t, "id": int(rp.label), "x": float(cx), "y": float(cy)})

    return pd.DataFrame(rows)


def find_acdc_output_csv(images_dir: Path) -> Optional[Path]:
    """Find the most recent *_acdc_output.csv that is not a backup."""
    cands = [p for p in images_dir.glob("*acdc_output*.csv")
             if "original" not in p.name.lower()]
    if not cands:
        cands = [p for p in images_dir.glob("*.csv")
                 if "output" in p.name.lower() and "original" not in p.name.lower()
                 and "trackpy" not in p.name.lower()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def find_segm_npz(images_dir: Path) -> Optional[Path]:
    cands = [p for p in images_dir.glob("*.npz")
             if "segm" in p.name.lower()
             and "aligned" not in p.name.lower()
             and "bkgr" not in p.name.lower()]
    if not cands:
        cands = list(images_dir.glob("*.npz"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


# ─────────────────────────────────────────────────────────────────
# TrackPy linking
# ─────────────────────────────────────────────────────────────────

def run_trackpy(
    detections: pd.DataFrame,
    search_range_px: float,
    memory: int,
    min_frames: int,
    adaptive_stop: Optional[float],
    adaptive_step: float,
    use_predictor: bool,
    predictor_span: int,
    pos_label: str,
) -> pd.DataFrame:
    """
    Link detections using TrackPy nearest-neighbour + gap-closing.

    Parameters
    ----------
    detections      : DataFrame with columns frame, id, x, y
    search_range_px : residual search radius in pixels.
                      With predictor ON  → residual after velocity prediction (~15 px).
                      With predictor OFF → full per-frame displacement (may explode
                      in dense fields if > inter-cell distance).
    memory          : max frames a track may disappear (gap closing)
    min_frames      : discard tracks shorter than this
    adaptive_stop   : if not None, TrackPy reduces search_range until
                      this fraction of particles are linked (0.0–1.0).
                      Useful when search_range is too large and causes
                      link collisions. Typical value: 0.95
    adaptive_step   : fractional reduction step for adaptive mode (0.5–0.99)
    use_predictor   : if True, wrap linking in NearestVelocityPredict so the
                      search circle is centred on the predicted next position
                      rather than the current position.  This is the correct
                      approach for fast-moving dense cells: it reduces the
                      effective search radius from ~70 px to ~15 px, eliminating
                      subnetwork explosions without throwing away information.
    predictor_span  : number of past frames used to estimate velocity (default 3).

    Returns
    -------
    DataFrame with columns: frame, id, x, y, particle
      particle = persistent TrackPy track ID (0-based integers)
    """
    df = detections[["frame", "id", "x", "y"]].copy()
    df = df.sort_values("frame").reset_index(drop=True)

    n_det    = len(df)
    n_frames = df["frame"].nunique()
    print(f"  [{pos_label}] {n_det:,} detections across {n_frames} frames")
    print(f"  [{pos_label}] linking: search_range={search_range_px}px  "
          f"memory={memory}  min_frames={min_frames}"
          + (f"  predictor=NearestVelocity(span={predictor_span})" if use_predictor else
             "  predictor=OFF"))

    def _link(sr, adap_stop=None, adap_step=0.95):
        """Inner helper — runs tp.link_df with or without predictor."""
        kwargs = dict(
            search_range=sr,
            memory=memory,
            pos_columns=["x", "y"],
        )
        if adap_stop is not None:
            kwargs["adaptive_stop"] = adap_stop
            kwargs["adaptive_step"] = adap_step

        if use_predictor:
            # NearestVelocityPredict shifts the search window to the predicted
            # next position.  search_range then only needs to cover the residual
            # prediction error, not the full per-frame displacement.
            #
            # API note: use pred.link_df() — NOT "with pred: tp.link_df()".
            # The context-manager form was removed in trackpy 0.5; pred.link_df()
            # is the correct call in both 0.4 and 0.5+.
            pred = tp.predict.NearestVelocityPredict(span=predictor_span)
            return pred.link_df(df, **kwargs)
        else:
            return tp.link_df(df, **kwargs)

    # TrackPy needs columns named 'x','y','frame' exactly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if adaptive_stop is not None:
            # Explicit adaptive mode requested by user
            print(f"  [{pos_label}] adaptive mode: stop={adaptive_stop}  step={adaptive_step}")
            linked = _link(search_range_px, adap_stop=adaptive_stop, adap_step=adaptive_step)

        else:
            # Try standard (or predictor-wrapped) linking first.
            # If the subnetwork still explodes (can happen with predictor OFF and
            # large search_range), auto-retry with adaptive mode.
            try:
                linked = _link(search_range_px)
            except Exception as e:
                if "subnetwork" in str(e).lower():
                    print(f"  [{pos_label}] [warn] Subnetwork too large at search_range="
                          f"{search_range_px}px — auto-retrying with adaptive mode "
                          f"(adaptive_stop=0.95, adaptive_step=0.95).")
                    print(f"  [{pos_label}] [info] To skip this fallback, pass "
                          f"--adaptive-stop 0.95 explicitly, or reduce --search-range-px.")
                    linked = _link(search_range_px, adap_stop=0.95, adap_step=0.95)
                else:
                    raise

    # Filter short tracks
    before_tracks = linked["particle"].nunique()
    if min_frames > 1:
        linked = tp.filter_stubs(linked, threshold=min_frames)
    after_tracks = linked["particle"].nunique()

    n_removed = before_tracks - after_tracks
    print(f"  [{pos_label}] tracks before filter: {before_tracks:,}  "
          f"→ after min_frames={min_frames}: {after_tracks:,}  "
          f"({n_removed} short tracks removed)")

    # Re-index particle IDs to start from 1 (cleaner for human inspection)
    old_to_new = {old: new for new, old in enumerate(sorted(linked["particle"].unique()), start=1)}
    linked["particle"] = linked["particle"].map(old_to_new)

    return linked[["frame", "id", "x", "y", "particle"]].reset_index(drop=True)


def print_track_stats(linked: pd.DataFrame, pos_label: str):
    """Print a brief summary of track length distribution."""
    lengths = linked.groupby("particle")["frame"].count()
    print(f"  [{pos_label}] Track length distribution:")
    print(f"    min={lengths.min()}  median={lengths.median():.0f}  "
          f"mean={lengths.mean():.1f}  max={lengths.max()}  "
          f"total tracks={len(lengths)}")

    # Show histogram bins
    bins = [1, 3, 5, 10, 20, 50, 100, 999999]
    labels = ["1-2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+"]
    counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = ((lengths >= lo) & (lengths < hi)).sum()
        counts.append(n)
    for label, count in zip(labels, counts):
        bar = "█" * min(count, 50)
        print(f"    {label:>6} frames: {count:>5}  {bar}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Re-link ACDC segmentation detections with TrackPy gap-closing to produce persistent track IDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dense/fast-moving cells (default — predictor ON, residual range 15px)
  python run_trackpy_linking.py --exp_root segmentation_input --all

  # Explicitly set predictor span and residual range
  python run_trackpy_linking.py --exp_root segmentation_input --all \\
      --search-range-px 15 --predictor-span 3 --memory 3 --min-frames 20

  # Disable predictor (only for slowly-moving cells, displacement < inter-cell gap)
  python run_trackpy_linking.py --exp_root segmentation_input --all \\
      --no-predictor --search-range-px 20 --memory 3

  # Only a specific position
  python run_trackpy_linking.py --exp_root segmentation_input --select field1_B3_1_NIR

  # After running this, immediately run:
  python export_imaris_like_from_pipeline.py --exp_root segmentation_input --all \\
      --pixel-size-um 1.22 --z-step-um 1.0 --frame-interval-s 3600
  (TrackPy CSVs are auto-detected; no --trackpy-csv flag needed)

Parameter guide
───────────────
--search-range-px (default 15)
  With predictor ON  : residual search radius around the predicted position.
                       10-20 px is right for most datasets.
  With predictor OFF : full per-frame displacement. Must exceed actual cell
                       movement, which causes subnetwork explosion in dense fields.

--use-predictor / --no-predictor (default: ON)
  NearestVelocityPredict shifts the search window to where the cell is
  expected to be next frame, based on its recent velocity. This reduces
  the effective search radius from ~70 px to ~15 px in fast-dense datasets,
  eliminating subnetwork explosions and improving linking accuracy.

--predictor-span (default 3)
  Frames of history used to estimate velocity. 3 is a good default.
  For the first `span` frames of each new track, falls back to standard
  linking automatically.

--memory (default 3)
  Max frames a track may disappear before the gap is bridged.

--min-frames (default 20)
  Tracks shorter than this are dropped. Set close to your total frame count
  to discard fragments and keep only near-complete tracks.

--adaptive-stop (default: not used)
  Last-resort fallback if subnetwork still explodes even with predictor ON.
  Pass 0.95 to let TrackPy reduce search_range until the problem fits.
        """,
    )

    # Position selection (same interface as other pipeline scripts)
    ap.add_argument("--exp_root",  default="segmentation_input",
                    help="Folder containing Position_1/, Position_2/, ... (default: segmentation_input)")
    ap.add_argument("--map",       default="",
                    help="Mapping file inside exp_root (position_map.csv / positions_index.csv / .json)")
    ap.add_argument("--all",       action="store_true", help="Process all positions")
    ap.add_argument("--select",    default="field1_B2_NIR", help="Selection key (default: field1_B2_NIR). e.g. field1_B3_1_NIR or field1|B3_1|NIR")
    ap.add_argument("--channel",   default="", help="Process all positions for a single channel")
    ap.add_argument("--list",      action="store_true", help="List available positions and exit")

    # TrackPy parameters
    ap.add_argument("--search-range-px", type=float, default=15.0,
                    help="Residual search radius in pixels (default: 15). "
                         "With predictor ON this is the residual after velocity prediction — "
                         "10-20 px is appropriate. With --no-predictor this must cover the "
                         "full per-frame displacement.")
    ap.add_argument("--memory",          type=int,   default=3,
                    help="Max frames a track may disappear and still be re-linked (default: 3).")
    ap.add_argument("--min-frames",      type=int,   default=20,
                    help="Drop tracks shorter than this many frames (default: 20). "
                         "Set close to total frame count to keep only near-complete tracks.")
    ap.add_argument("--adaptive-stop",   type=float, default=None,
                    help="Enable TrackPy adaptive mode (0.0–1.0, e.g. 0.95). "
                         "Last resort if subnetwork still explodes with predictor ON.")
    ap.add_argument("--adaptive-step",   type=float, default=0.95,
                    help="Step size for adaptive mode (default: 0.95). Only used with --adaptive-stop.")

    # Velocity predictor
    predictor_grp = ap.add_mutually_exclusive_group()
    predictor_grp.add_argument(
        "--use-predictor", dest="use_predictor", action="store_true", default=True,
        help="Use NearestVelocityPredict — shifts search window to predicted next position "
             "(default: ON). Correct approach for fast-moving dense cells.")
    predictor_grp.add_argument(
        "--no-predictor", dest="use_predictor", action="store_false",
        help="Disable velocity predictor. Only for slowly-moving cells "
             "where displacement << inter-cell distance.")
    ap.add_argument("--predictor-span", type=int, default=3,
                    help="Frames of history used to estimate velocity (default: 3).")

    # Output control
    ap.add_argument("--overwrite",  action="store_true",
                    help="Overwrite existing TrackPy CSVs (default: skip if already exists)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print what would be done without writing any files")

    args = ap.parse_args()

    exp_root = Path(args.exp_root).resolve()
    if not exp_root.exists():
        raise SystemExit(f"[ERROR] exp_root not found: {exp_root}")

    # Load position map
    map_path = find_mapping_file(exp_root, args.map if args.map else None)
    rows     = load_mapping(map_path)
    print(f"[load] mapping file: {map_path}  ({len(rows)} entries)")

    if args.list:
        print("\nAvailable positions:")
        for r in rows:
            ff  = get_first(r, "field_folder", "field", "parent_folder")
            fn  = get_first(r, "field_name", default=get_first(r, "location"))
            ch  = get_first(r, "channel", "channel_name", "ch")
            pos = position_folder_from_row(r)
            print(f"  {ff}_{fn}_{ch}  →  {pos}")
        return

    selected = filter_rows(rows, select=args.select, run_all=args.all,
                           channel_only=args.channel)
    if not selected:
        raise SystemExit("[ERROR] No positions matched. Use --list to see available positions.")

    # Deduplicate: one TrackPy CSV per (field_folder, location) pair,
    # shared across all channels for the same physical well.
    # (Tracks are based on segmentation labels, which are channel-independent.)
    seen_positions = set()
    failures = 0
    processed = 0
    skipped = 0

    print(f"\n[start] processing {len(selected)} selected entries...")
    print(f"  search_range={args.search_range_px}px  "
          f"memory={args.memory}  "
          f"min_frames={args.min_frames}  "
          f"predictor={'NearestVelocity(span='+str(args.predictor_span)+')' if args.use_predictor else 'OFF'}\n")

    for r in selected:
        pos_folder  = position_folder_from_row(r)
        images_dir  = exp_root / pos_folder / "Images"

        ff  = get_first(r, "field_folder", "field", "parent_folder", default="field")
        fn  = get_first(r, "field_name", default=get_first(r, "location", default="loc"))
        loc = get_first(r, "location", default=fn.rsplit("_", 1)[0] if "_" in fn else fn)
        pos_label = f"{ff}_{fn}"

        # Output CSV path: <field>_<location>_trackpy_tracks.csv
        # Saved in Images/ so export_imaris_like_from_pipeline auto-detects it
        out_csv = images_dir / f"{ff}_{loc}_trackpy_tracks.csv"

        # Skip if already done (unless --overwrite)
        if out_csv.exists() and not args.overwrite:
            print(f"[skip] {pos_label} → {out_csv.name} already exists (use --overwrite to redo)")
            skipped += 1
            continue

        # Deduplicate across channels for the same physical position
        dedup_key = (ff, loc, pos_folder)
        if dedup_key in seen_positions:
            print(f"[skip] {pos_label} → already processed for this (field, location, position)")
            skipped += 1
            continue
        seen_positions.add(dedup_key)

        print(f"{'='*60}")
        print(f"[process] {pos_label}  →  {pos_folder}")
        print(f"  Images dir: {images_dir}")

        if not images_dir.exists():
            print(f"  [ERROR] Images dir not found: {images_dir}")
            failures += 1
            continue

        if args.dry_run:
            print(f"  [DRY RUN] would write: {out_csv}")
            continue

        # ── 1. Get centroids ──────────────────────────────────────────────
        detections = None

        # Try ACDC output CSV first (faster, no NPZ loading)
        acdc_csv = find_acdc_output_csv(images_dir)
        if acdc_csv is not None:
            print(f"  Centroids source: ACDC output CSV ({acdc_csv.name})")
            try:
                detections = centroids_from_acdc_csv(acdc_csv)
                if detections is not None:
                    print(f"  Loaded {len(detections):,} detections from CSV")
            except Exception as e:
                print(f"  [warn] Failed to read ACDC CSV: {e}")
                detections = None

        # Fall back to NPZ if CSV failed or was missing
        if detections is None or len(detections) == 0:
            npz_path = find_segm_npz(images_dir)
            if npz_path is None:
                print(f"  [ERROR] No segmentation NPZ found in {images_dir}")
                failures += 1
                continue
            print(f"  Centroids source: NPZ segmentation ({npz_path.name})")
            try:
                detections = centroids_from_npz(npz_path)
                print(f"  Computed {len(detections):,} detections from NPZ")
            except Exception as e:
                print(f"  [ERROR] Failed to compute centroids from NPZ: {e}")
                failures += 1
                continue

        if detections is None or len(detections) == 0:
            print(f"  [ERROR] No detections found — skipping {pos_label}")
            failures += 1
            continue

        # ── 2. Run TrackPy linking ────────────────────────────────────────
        try:
            linked = run_trackpy(
                detections=detections,
                search_range_px=args.search_range_px,
                memory=args.memory,
                min_frames=args.min_frames,
                adaptive_stop=args.adaptive_stop,
                adaptive_step=args.adaptive_step,
                use_predictor=args.use_predictor,
                predictor_span=args.predictor_span,
                pos_label=pos_label,
            )
        except Exception as e:
            print(f"  [ERROR] TrackPy linking failed: {e}")
            print(f"  Tip: ensure --use-predictor is ON (default), or try --adaptive-stop 0.95")
            failures += 1
            continue

        print_track_stats(linked, pos_label)

        # ── 3. Save output CSV ────────────────────────────────────────────
        # Ensure column order matches what the converter expects:
        #   frame | id | particle   (+ x, y for debugging)
        out_df = linked[["frame", "id", "particle", "x", "y"]].copy()
        out_df = out_df.sort_values(["particle", "frame"]).reset_index(drop=True)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print(f"  [saved] {out_csv}")
        print(f"  Rows: {len(out_df):,}  |  Unique tracks: {out_df['particle'].nunique():,}")
        processed += 1

    print(f"\n{'='*60}")
    print(f"[done]  processed={processed}  skipped={skipped}  failures={failures}")
    if processed > 0:
        print(f"\nNext step — run the Imaris-like exporter:")
        print(f"  python export_imaris_like_from_pipeline.py \\")
        print(f"      --exp_root {exp_root} --all \\")
        print(f"      --pixel-size-um <VALUE> --z-step-um <VALUE> --frame-interval-s <VALUE>")
        print(f"\n  (TrackPy CSVs are auto-detected from Images/ — no --trackpy-csv flag needed)")
    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()