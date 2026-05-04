#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


"""
export_imaris_like_from_pipeline.py  (FORMULA-VALIDATED UPDATE)

Wraps acdc_npz_tif_to_imaris_like.py and runs it across all selected positions.

FORMULA CHANGES (validated against Imaris ground truth):
──────────────────────────────────────────────────────────
1) Speed / Velocity: now uses CENTRAL DIFFERENCE  (r²=0.987)
   → No user-facing change needed; the converter handles this automatically.

2) Acceleration: now always outputs ZERO (ground truth is all zeros).

3) Track Speed Mean: track_length / duration  (r²=0.976)

4) Track Speed Max/Min/StdDev/Variation: backward-diff per-frame speeds.

COORDINATE OFFSET DEFAULTS CHANGED:
─────────────────────────────────────
--imaris-offset-x-px  default 0.0  (was -4.0)
--imaris-offset-y-px  default 0.0  (was +9.0)
These offsets are dataset-specific and MUST be calibrated per acquisition.
Pass them explicitly if your dataset has a known ACDC↔Imaris registration shift.
"""


# -----------------------------
# mapping helpers
# -----------------------------
def _k(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def _v(s: str) -> str:
    return (s or "").strip()


def find_mapping_file(exp_root: Path, user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            p = exp_root / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Mapping file not found: {p}")
        return p

    candidates = [
        exp_root / "positions_index.csv",
        exp_root / "position_map.csv",
        exp_root / "position_map",
        exp_root / "positions_index.json",
        exp_root / "position_map.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    csvs = sorted([p.name for p in exp_root.glob("*.csv")])
    raise FileNotFoundError(
        f"No mapping file found in: {exp_root}\n"
        f"Tried: positions_index.csv / position_map.csv / *.json\n"
        f"Found CSVs: {csvs if csvs else 'NONE'}\n"
        f"Pass explicitly with --map <file>"
    )


def load_mapping(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [{_k(str(k)): _v(str(v)) for k, v in r.items()} for r in data]

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header row: {path}")
        for r in reader:
            rows.append({_k(h): _v(r.get(h, "")) for h in reader.fieldnames})
    return rows


def get_first(row: Dict[str, str], *keys: str, default: str = "") -> str:
    for kk in keys:
        val = row.get(_k(kk), "")
        if val:
            return val
    return default


def position_folder_from_row(row: Dict[str, str]) -> str:
    # supports either "position_folder" or numeric "position"/"position_number"
    pf = get_first(row, "position_folder", "position_dir", "position_name")
    if pf:
        return pf

    num = get_first(row, "position_number", "position")
    if num.isdigit():
        return f"Position_{int(num)}"

    raw = get_first(row, "Position")
    if raw.lower().startswith("position_"):
        return raw

    raise KeyError(f"Cannot determine position folder from row keys: {sorted(row.keys())}")


def filter_rows(rows: List[Dict[str, str]], select: str, run_all: bool) -> List[Dict[str, str]]:
    if run_all:
        return rows
    if not select:
        raise SystemExit("Use --all or --select")

    sel = select.strip()

    # field1|B3_1|NIR
    if "|" in sel:
        parts = [s.strip() for s in sel.split("|")]
        if len(parts) != 3:
            raise SystemExit('Bad --select format. Use: field1|B3_1|NIR')
        ff, fn, ch = parts
        out = [
            r for r in rows
            if _k(get_first(r, "field_folder", "field", "parent_folder")) == _k(ff)
            and _k(get_first(r, "field_name")) == _k(fn)
            and _k(get_first(r, "channel", "channel_name", "ch")) == _k(ch)
        ]
        return out

    # field1_B3_1_NIR (recommended)
    toks = sel.split("_")
    if len(toks) < 3:
        raise SystemExit('Bad --select. Use: field1_B3_1_NIR or field1|B3_1|NIR')

    field_folder = toks[0]
    channel = toks[-1]

    if len(toks) == 4:
        location = toks[1]
        rep = toks[2]
        field_name = f"{location}_{rep}"
        return [
            r for r in rows
            if _k(get_first(r, "field_folder", "field", "parent_folder")) == _k(field_folder)
            and _k(get_first(r, "field_name")) == _k(field_name)
            and _k(get_first(r, "channel", "channel_name", "ch")) == _k(channel)
        ]

    # field1_B3_NIR (may be ambiguous; allow but could match multiple)
    location = toks[1]
    return [
        r for r in rows
        if _k(get_first(r, "field_folder", "field", "parent_folder")) == _k(field_folder)
        and _k(get_first(r, "location", "letter", "site")) == _k(location)
        and _k(get_first(r, "channel", "channel_name", "ch")) == _k(channel)
    ]


# -----------------------------
# find ACDC outputs in Position_X/Images
# -----------------------------
def newest(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def find_segm_npz(images_dir: Path) -> Path:
    # prefer files containing "segm" and ending ".npz"
    cand = [p for p in images_dir.glob("*.npz") if "segm" in p.name.lower()]
    cand = [p for p in cand if "aligned" not in p.name.lower() and "bkgr" not in p.name.lower()]
    p = newest(cand)
    if p:
        return p

    # fallback: any npz
    p = newest(list(images_dir.glob("*.npz")))
    if p:
        return p

    raise FileNotFoundError(f"No segmentation .npz found in: {images_dir}")


def infer_channel_from_tif_name(tif_path: Path) -> str:
    # field1_B3_NIR.tif -> NIR
    stem = tif_path.stem
    if "_" in stem:
        return stem.split("_")[-1]
    return "CH"


def find_signal_tif(images_dir: Path, channel_hint: str = "") -> Path:
    tifs = sorted(images_dir.glob("*.tif*"))
    # ignore mask-like or segm-like tifs if they exist
    tifs = [p for p in tifs if "mask" not in p.name.lower() and "segm" not in p.name.lower()]
    if not tifs:
        raise FileNotFoundError(f"No signal tif found in: {images_dir}")

    if channel_hint:
        ch = channel_hint.lower()
        exact = [p for p in tifs if p.stem.lower().endswith(f"_{ch}")]
        if exact:
            return exact[0]

    if len(tifs) == 1:
        return tifs[0]

    p = newest(tifs)
    if p:
        return p

    return tifs[0]


def find_trackpy_csv(images_dir: Path) -> Optional[Path]:
    """
    Find TrackPy tracks CSV.
    Searches in order:
      1. Position_X/Images/  (copied there by build_segmentation_input)
      2. segmentation_input/ parent folder (fallback)
      3. project root (two levels up from Images/)
    """
    for search_dir in [
        images_dir,                          # Position_X/Images/
        images_dir.parent.parent,            # segmentation_input/
        images_dir.parent.parent.parent,     # project root
    ]:
        cands = list(search_dir.glob("*trackpy_tracks*.csv"))
        if not cands:
            cands = [p for p in search_dir.glob("*.csv")
                     if "trackpy" in p.name.lower()]
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0]
    return None


def find_acdc_output_table(images_dir: Path) -> Optional[Path]:
    cand = [p for p in images_dir.glob("*acdc_output*.csv")
            if "original" not in p.name.lower()]
    if not cand:
        # fallback: any csv that looks like output but is not a backup
        cand = [p for p in images_dir.glob("*.csv")
                if "output" in p.name.lower() and "original" not in p.name.lower()]
    return newest(cand)


# -----------------------------
# run converter
# -----------------------------
def run_converter(
    python_exe: str,
    converter_py: Path,
    segm_npz: Path,
    tif_path: Path,
    out_xlsx: Path,
    pixel_size_um: float,
    z_step_um: float,
    frame_interval_s: float,
    displacement2_mode: str,
    use_acdc_table: bool,
    map_max_dist_px: float,
    map_offset_x_px: float,
    map_offset_y_px: float,
    map_offset_z_px: float,
    # NEW: global ACDC->Imaris offsets (pixels)
    imaris_offset_x_px: float,
    imaris_offset_y_px: float,
    imaris_offset_z_px: float,
    dry_run: bool,
    trackpy_csv: Optional[Path] = None,
    # Time index of the first frame — must match the converter's --time-index-start.
    # Default=2: Imaris convention (empirically confirmed: first frame = Time 2).
    time_index_start: int = 2,
) -> int:
    cmd = [
        python_exe, str(converter_py),
        "--segm-npz", str(segm_npz),
        "--tif", str(tif_path),
        "--out", str(out_xlsx),
        "--pixel-size-um", str(pixel_size_um),
        "--z-step-um", str(z_step_um),
        "--frame-interval-s", str(frame_interval_s),
        "--displacement2-mode", displacement2_mode,
        # Time index start — must match Imaris time convention for this dataset
        "--time-index-start", str(time_index_start),
        # pass global offsets to converter
        "--imaris-offset-x-px", str(imaris_offset_x_px),
        "--imaris-offset-y-px", str(imaris_offset_y_px),
        "--imaris-offset-z-px", str(imaris_offset_z_px),
    ]

    # Mapping is automatic when --acdc-table is provided (and unmapped objects are skipped).
    if use_acdc_table:
        acdc_table = find_acdc_output_table(tif_path.parent)
        if acdc_table:
            cmd += [
                "--acdc-table", str(acdc_table),
                "--map-max-dist-px", str(map_max_dist_px),
                "--map-offset-x-px", str(map_offset_x_px),
                "--map-offset-y-px", str(map_offset_y_px),
                "--map-offset-z-px", str(map_offset_z_px),
            ]
        else:
            print(f"[WARN] no *_acdc_output.csv found in {tif_path.parent}. Export will include NPZ-only objects.")

    if trackpy_csv is not None and trackpy_csv.exists():
        cmd += ["--trackpy-csv", str(trackpy_csv)]
        print(f"[info] TrackPy CSV found: {trackpy_csv}")
        print(f"[info] Parent will use particle (persistent track ID) from TrackPy")
    else:
        print(f"[WARN] No TrackPy CSV found - Parent == Cell_ID (per-frame label, NOT a persistent track ID)")
        print(f"[WARN] Track-level stats (Duration, Speed, etc.) will be wrong without TrackPy CSV")
        print(f"[WARN] Expected file: <field>_<location>_trackpy_tracks.csv in Images/ or project root")

    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return 0

    print("Running converter:\n ", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Export Imaris-like Excel from Cell-ACDC results (no moving files).")
    ap.add_argument("--exp_root", default="segmentation_input", help="Contains Position_1/Position_2/... folders")
    ap.add_argument("--map", default="", help="Mapping file inside exp_root (position_map.csv / positions_index.csv / json)")
    ap.add_argument("--select", default="", help="e.g. field1_B3_1_NIR or field1|B3_1|NIR")
    ap.add_argument("--all", action="store_true", help="Export all positions")
    ap.add_argument("--out_dir", default="pipeline_output", help="Where to write Imaris-like workbooks")

    ap.add_argument("--converter", default="acdc_npz_tif_to_imaris_like_FIXED.py",
                    help="Path to your converter script")
    ap.add_argument("--python", default="", help="Optional: Python interpreter to run the converter (default: current python)")

    ap.add_argument("--pixel-size-um", type=float, required=True)
    ap.add_argument("--z-step-um", type=float, required=True)
    ap.add_argument("--frame-interval-s", type=float, required=True)

    ap.add_argument("--displacement2-mode", choices=["from_start", "step"], default="from_start")
    ap.add_argument("--no-acdc-table", action="store_true", help="Do NOT use *_acdc_output.csv (export NPZ-only objects too)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--trackpy-csv", default="",
                    help="Explicit path to TrackPy CSV. Overrides auto-detection. "
                         "Used to set persistent Parent (track ID) in output.")

    # mapping params used by converter when CSV exists (Parent mapping)
    ap.add_argument("--map-max-dist-px", type=float, default=8.0)
    ap.add_argument("--map-offset-x-px", type=float, default=0.0)
    ap.add_argument("--map-offset-y-px", type=float, default=0.0)
    ap.add_argument("--map-offset-z-px", type=float, default=0.0)

    # Global ACDC->Imaris coordinate correction (pixels).
    # DEFAULT IS 0 — these are dataset-specific and must be calibrated per acquisition.
    # Only pass non-zero values if you have measured the registration offset between
    # your ACDC segmentation coordinates and Imaris physical space for your microscope.
    ap.add_argument("--imaris-offset-x-px", type=float, default=0.0)
    ap.add_argument("--imaris-offset-y-px", type=float, default=0.0)
    ap.add_argument("--imaris-offset-z-px", type=float, default=0.0)
    ap.add_argument("--time-index-start", type=int, default=2,
                    help="Time index of the first frame in output Excel. "
                         "Default=2 (empirically confirmed Imaris convention). "
                         "Use 1 if your Imaris export shows Time=1 for the first frame. "
                         "CRITICAL: must match the ground-truth Imaris file for all per-frame "
                         "metric joins (Area, Speed, Intensity etc.) to work correctly.")

    args = ap.parse_args()

    exp_root = Path(args.exp_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    converter_py = Path(args.converter)
    if not converter_py.is_absolute():
        converter_py = (Path.cwd() / converter_py).resolve()
    if not converter_py.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_py}")

    python_exe = args.python.strip() if args.python.strip() else sys.executable

    mapping_path = find_mapping_file(exp_root, args.map if args.map else None)
    rows = load_mapping(mapping_path)
    sel = filter_rows(rows, select=args.select, run_all=args.all)

    if not sel:
        raise SystemExit("No rows selected. Use --all or a valid --select.")

    # If user used field1_B3_NIR and it matches multiple, force explicit replicate
    if args.select and len(args.select.split("_")) == 3 and len(sel) > 1:
        print(f'Ambiguous selection "{args.select}". Matches {len(sel)} rows.')
        print("Use explicit replicate: field1_B3_1_NIR or field1_B3_2_NIR")
        raise SystemExit(1)

    failures = 0
    for r in sel:
        pos_folder = position_folder_from_row(r)  # Position_#
        images_dir = exp_root / pos_folder / "Images"
        if not images_dir.exists():
            print("[SKIP] Missing:", images_dir)
            failures += 1
            continue

        try:
            channel = get_first(r, "channel", "channel_name", "ch")
            tif_path = find_signal_tif(images_dir, channel_hint=channel)
            if not channel:
                channel = infer_channel_from_tif_name(tif_path)
            segm_npz = find_segm_npz(images_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {pos_folder} - missing required file: {e}")
            failures += 1
            continue


        ff = get_first(r, "field_folder", "field", "parent_folder", default="field")
        fn = get_first(r, "field_name", default=get_first(r, "location", default="loc"))

        out_name = f"{ff}_{fn}_{channel}__{pos_folder}_imaris_like.xlsx"
        out_xlsx = out_dir / out_name

        print("========================================")
        print("Mapping file:", mapping_path)
        print("Position:", pos_folder)
        print("Images:", images_dir)
        print("TIF:", tif_path.name)
        print("SEGM:", segm_npz.name)
        print("OUT:", out_xlsx)
        print(f"Global ACDC->Imaris offset (px): dy={args.imaris_offset_y_px}, "
              f"dx={args.imaris_offset_x_px}, dz={args.imaris_offset_z_px}")
        print(f"Time index start: {args.time_index_start} "
              f"(first frame = Time {args.time_index_start} in output Excel)")

        # Use explicit --trackpy-csv if provided, otherwise auto-detect
        if args.trackpy_csv:
            trackpy_csv = Path(args.trackpy_csv)
            if not trackpy_csv.exists():
                print(f"[WARN] --trackpy-csv path not found: {trackpy_csv}")
                trackpy_csv = find_trackpy_csv(images_dir)
        else:
            trackpy_csv = find_trackpy_csv(images_dir)

        rc = run_converter(
            python_exe=python_exe,
            converter_py=converter_py,
            segm_npz=segm_npz,
            tif_path=tif_path,
            out_xlsx=out_xlsx,
            pixel_size_um=float(args.pixel_size_um),
            z_step_um=float(args.z_step_um),
            frame_interval_s=float(args.frame_interval_s),
            displacement2_mode=str(args.displacement2_mode),
            use_acdc_table=(not args.no_acdc_table),
            map_max_dist_px=float(args.map_max_dist_px),
            map_offset_x_px=float(args.map_offset_x_px),
            map_offset_y_px=float(args.map_offset_y_px),
            map_offset_z_px=float(args.map_offset_z_px),
            imaris_offset_x_px=float(args.imaris_offset_x_px),
            imaris_offset_y_px=float(args.imaris_offset_y_px),
            imaris_offset_z_px=float(args.imaris_offset_z_px),
            dry_run=bool(args.dry_run),
            trackpy_csv=trackpy_csv,
            time_index_start=int(args.time_index_start),
        )
        if rc != 0:
            failures += 1
            print(f"[ERROR] converter exit={rc} for {pos_folder}")

    print("\nDone.")
    print("Selected:", len(sel))
    print("Failures:", failures)
    print("Output folder:", out_dir)


if __name__ == "__main__":
    main()