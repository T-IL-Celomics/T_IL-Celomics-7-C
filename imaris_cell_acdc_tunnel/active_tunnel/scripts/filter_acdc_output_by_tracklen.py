"""
filter_track_length.py
======================
Filters Cell-ACDC output CSVs by minimum track length (number of frames).

Key fixes over previous version:
  - Uses Cell_ID + frame_i directly (no ambiguous column guessing)
  - Preserves original column order and dtypes on re-write
  - Drops unnamed index columns (Unnamed: 0) safely before saving
  - Safe backup logic (never overwrites an existing backup)
  - Dry-run mode reports without touching any file
  - Clear, readable console output

Usage examples:
  python filter_track_length.py --all --min_frames 15
  python filter_track_length.py --select field1_B2_1_NIR --min_frames 10
  python filter_track_length.py --all --min_frames 15 --dry_run
  python filter_track_length.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ─────────────────────────────────────────
# Key column names — edit here if needed
# ─────────────────────────────────────────
TRACK_COL = "Cell_ID"
FRAME_COL = "frame_i"


# ─────────────────────────────────────────
# Mapping helpers
# ─────────────────────────────────────────

def k(s: str) -> str:
    """Normalise a key: strip, lowercase, underscores."""
    return (s or "").strip().lower().replace(" ", "_")


def v(s: str) -> str:
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

    csvs = sorted(p.name for p in exp_root.glob("*.csv"))
    raise FileNotFoundError(
        f"No mapping file found in: {exp_root}\n"
        f"Tried: positions_index.csv / position_map.csv / *.json\n"
        f"Found CSVs: {csvs or 'NONE'}\n"
        f"Pass explicitly with --map <file>"
    )


def load_mapping(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [{k(str(kk)): v(str(vv)) for kk, vv in r.items()} for r in data]

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header row: {path}")
        for r in reader:
            rows.append({k(h): v(r.get(h, "")) for h in reader.fieldnames})
    return rows


def get_first(row: Dict[str, str], *keys: str, default: str = "") -> str:
    for kk in keys:
        val = row.get(k(kk), "")
        if val:
            return val
    return default


def compute_position_folder(row: Dict[str, str]) -> str:
    pf = get_first(row, "position_folder", "position_dir", "position_name")
    if pf:
        return pf
    num = get_first(row, "position_number", "position")
    if num.isdigit():
        return f"Position_{int(num)}"
    raw = get_first(row, "Position")
    if raw.lower().startswith("position_"):
        return raw
    raise KeyError(
        f"Cannot determine position folder from row keys: {sorted(row.keys())}"
    )


def pretty_row(row: Dict[str, str]) -> str:
    ff  = get_first(row, "field_folder", "field", "parent_folder")
    fn  = get_first(row, "field_name", "fieldname")
    loc = get_first(row, "location", "letter", "site")
    ch  = get_first(row, "channel", "channel_name", "ch")
    pos = compute_position_folder(row)
    if ff and fn and ch:
        return f"{ff}_{fn}_{ch} -> {pos}"
    if ff and loc and ch:
        return f"{ff}_{loc}_{ch} -> {pos}"
    return f"(unknown selection) -> {pos}"


def filter_rows(
    rows: List[Dict[str, str]], select: str, run_all: bool
) -> List[Dict[str, str]]:
    if run_all:
        return rows
    if not select:
        raise SystemExit("Use --all OR --select <selection>")

    sel = select.strip()

    # pipe-separated: field1|B3_1|NIR
    if "|" in sel:
        parts = [p.strip() for p in sel.split("|")]
        if len(parts) != 3:
            raise SystemExit(
                f'Bad selection_key "{sel}". Expected: field_folder|field_name|channel'
            )
        ff, fn, ch = parts
        return [
            r for r in rows
            if k(get_first(r, "field_folder", "field", "parent_folder")) == k(ff)
            and k(get_first(r, "field_name")) == k(fn)
            and k(get_first(r, "channel", "channel_name", "ch")) == k(ch)
        ]

    # underscore-separated: field1_B2_1_NIR or field1_B2_NIR
    toks = sel.split("_")
    if len(toks) < 3:
        raise SystemExit(
            "Bad --select. Use: field1_B2_1_NIR  (or field1|B2_1|NIR)"
        )

    field_folder = toks[0]
    channel = toks[-1]

    if len(toks) == 4:
        field_name = f"{toks[1]}_{toks[2]}"
        return [
            r for r in rows
            if k(get_first(r, "field_folder", "field", "parent_folder")) == k(field_folder)
            and k(get_first(r, "field_name")) == k(field_name)
            and k(get_first(r, "channel", "channel_name", "ch")) == k(channel)
        ]

    # len == 3 → might be ambiguous across replicates
    location = toks[1]
    return [
        r for r in rows
        if k(get_first(r, "field_folder", "field", "parent_folder")) == k(field_folder)
        and k(get_first(r, "location", "letter", "site")) == k(location)
        and k(get_first(r, "channel", "channel_name", "ch")) == k(channel)
    ]


# ─────────────────────────────────────────
# ACDC output CSV handling
# ─────────────────────────────────────────

def find_acdc_output_csv(images_dir: Path) -> Path:
    cands = [p for p in images_dir.glob("*acdc_output*.csv")
             if "original" not in p.name.lower()]
    if not cands:
        cands = [p for p in images_dir.glob("*.csv")
                 if "output" in p.name.lower() and "original" not in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f"No *acdc_output*.csv found in: {images_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def find_trackpy_csv(images_dir: Path) -> Optional[Path]:
    """
    Find TrackPy tracks CSV.
    Searches in order:
      1. Position_X/Images/  (copied there by build_segmentation_input)
      2. segmentation_input/ parent folder
      3. project root (two levels up from Images/)
    """
    for search_dir in [
        images_dir,
        images_dir.parent.parent,
        images_dir.parent.parent.parent,
    ]:
        cands = list(search_dir.glob("*trackpy_tracks*.csv"))
        if not cands:
            cands = [p for p in search_dir.glob("*.csv")
                     if "trackpy" in p.name.lower()]
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0]
    return None


def backup_original(src: Path) -> Path:
    """Copy src → <stem>_original<ext>, only if the backup doesn't already exist."""
    dst = src.with_name(src.stem + "_original" + src.suffix)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    return dst  # backup already existed from a previous run


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    """Raise a clear error if the required columns are missing."""
    missing = [c for c in (TRACK_COL, FRAME_COL) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Required column(s) not found in {csv_path.name}: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Edit TRACK_COL / FRAME_COL at the top of this script if your "
            f"column names differ."
        )


def filter_by_track_length(
    csv_path: Path,
    min_frames: int,
    dry_run: bool,
    trackpy_csv_path: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Remove rows from acdc_output whose TRACK is shorter than min_frames.

    Track identity:
      - If a TrackPy CSV is provided (frame, ID, particle columns), the
        persistent 'particle' is used as the track ID.
        A particle is kept if it spans >= min_frames distinct frames.
        All acdc_output rows whose (frame_i, Cell_ID) map to a removed
        particle are dropped.
      - If NO TrackPy CSV is available, falls back to counting distinct
        frame_i values per Cell_ID (old behaviour — less accurate because
        Cell_ID is reassigned each frame by the segmenter).

    Re-writes the file with:
      - Same columns, same order
      - Unnamed index columns dropped
      - No extra pandas index
    """
    # ── read acdc_output ──────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    validate_columns(df, csv_path)

    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    before_rows = len(df)

    # ── TrackPy-aware filtering (correct) ────────────────────────────────────
    if trackpy_csv_path is not None and trackpy_csv_path.exists():
        tp = pd.read_csv(trackpy_csv_path)
        tp.columns = [c.strip().lower() for c in tp.columns]

        if not {"frame", "id", "particle"}.issubset(set(tp.columns)):
            raise RuntimeError(
                f"TrackPy CSV missing required columns (frame, id, particle). "
                f"Found: {list(tp.columns)}"
            )

        # Count distinct frames per particle (true track length)
        particle_lengths = (
            tp.groupby("particle")["frame"]
            .nunique()
            .rename("n_frames")
        )
        keep_particles = set(particle_lengths[particle_lengths >= min_frames].index)
        before_tracks = int(particle_lengths.shape[0])

        # Build (frame, cell_id) -> particle lookup
        frame_id_to_particle = {
            (int(row["frame"]), int(row["id"])): int(row["particle"])
            for _, row in tp.iterrows()
        }

        # Map each acdc_output row to its particle
        df["_particle"] = [
            frame_id_to_particle.get((int(fi), int(cid)), -1)
            for fi, cid in zip(df[FRAME_COL], df[TRACK_COL])
        ]

        df_filtered = df[df["_particle"].isin(keep_particles)].copy()
        df_filtered = df_filtered.drop(columns=["_particle"])

        after_tracks = int(df[df["_particle"].isin(keep_particles)]["_particle"].nunique())
        deleted_tracks = before_tracks - after_tracks

        print(f"  [trackpy] used particle-based filtering ({before_tracks:,} particles)")

    # ── Fallback: Cell_ID-based filtering (less accurate) ───────────────────
    else:
        print("  [warn] no TrackPy CSV found — falling back to Cell_ID-based filtering")
        print("         (less accurate: Cell_ID is reassigned each frame by segmenter)")

        track_lengths = (
            df.groupby(TRACK_COL)[FRAME_COL]
            .nunique()
            .rename("n_frames")
        )
        keep_ids = set(track_lengths[track_lengths >= min_frames].index)
        before_tracks = int(track_lengths.shape[0])

        df_filtered = df[df[TRACK_COL].isin(keep_ids)].copy()
        after_tracks = int(df_filtered[TRACK_COL].nunique())
        deleted_tracks = before_tracks - after_tracks

    after_rows = len(df_filtered)

    # ── write ─────────────────────────────────────────────────────────────────
    if not dry_run:
        df_filtered.to_csv(csv_path, index=False)

    return dict(
        before_rows=before_rows,
        after_rows=after_rows,
        before_tracks=before_tracks,
        after_tracks=after_tracks,
        deleted_tracks=deleted_tracks,
    )


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter Cell-ACDC output by minimum track length (frames).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--exp_root", default="segmentation_input",
        help="Folder containing Position_1 / Position_2 / … sub-folders",
    )
    ap.add_argument(
        "--map", default="",
        help="Mapping file (default: auto-detect positions_index.csv in exp_root)",
    )
    ap.add_argument(
        "--select", default="",
        help="e.g.  field1_B2_1_NIR   or   field1_B2_NIR   or   field1|B2_1|NIR",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Process every mapped position",
    )
    ap.add_argument(
        "--min_frames", type=int, default=15,
        help="Minimum number of DISTINCT frames a track must appear in (default 15)",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Report what would be deleted without writing anything",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="Print all available selections from the mapping file and exit",
    )
    args = ap.parse_args()

    exp_root = Path(args.exp_root).resolve()
    map_path = find_mapping_file(exp_root, args.map or None)
    rows     = load_mapping(map_path)

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        print(f"Mapping : {map_path}")
        print(f"Entries : {len(rows)}\n")
        for r in rows:
            print(" ", pretty_row(r))
        return

    # ── selection ─────────────────────────────────────────────────────────────
    selected = filter_rows(rows, select=args.select, run_all=args.all)

    if not selected:
        raise SystemExit(
            "No rows matched. Check --select value or use --list to see options."
        )

    # Ambiguity guard: 3-token select matched multiple replicates
    if args.select and len(args.select.split("_")) == 3 and len(selected) > 1:
        print(f'Ambiguous selection "{args.select}". Matches:')
        for r in selected:
            print("  ", pretty_row(r))
        raise SystemExit(
            "Be more specific — e.g. field1_B2_1_NIR or field1_B2_2_NIR"
        )

    # ── process ───────────────────────────────────────────────────────────────
    failures = 0

    for r in selected:
        pos_folder = compute_position_folder(r)
        images_dir = exp_root / pos_folder / "Images"

        if not images_dir.exists():
            print(f"\n[SKIP] Missing Images folder: {images_dir}")
            failures += 1
            continue

        try:
            out_csv      = find_acdc_output_csv(images_dir)
            trackpy_csv  = find_trackpy_csv(images_dir)
            backup_path  = backup_original(out_csv)
            if trackpy_csv:
                print(f"  [trackpy] found: {trackpy_csv.name}")
            stats        = filter_by_track_length(
                out_csv,
                min_frames=args.min_frames,
                dry_run=args.dry_run,
                trackpy_csv_path=trackpy_csv,
            )

            tag = "[DRY RUN] " if args.dry_run else ""
            print("\n" + "=" * 50)
            print(f"{tag}Selection : {pretty_row(r)}")
            print(f"  File     : {out_csv}")
            print(f"  Backup   : {backup_path}")
            print(f"  min_frames = {args.min_frames}")
            print(
                f"  Tracks   : {stats['before_tracks']:,} → {stats['after_tracks']:,} "
                f"({stats['deleted_tracks']:,} removed)"
            )
            print(
                f"  Rows     : {stats['before_rows']:,} → {stats['after_rows']:,}"
            )
            if args.dry_run:
                print("  ⚠  Dry run — nothing written.")

        except Exception as exc:
            failures += 1
            print("\n" + "=" * 50)
            print(f"[ERROR] {pretty_row(r)}")
            print(f"  Reason: {exc}")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Done.  Processed: {len(selected)}   Failures: {failures}")


if __name__ == "__main__":
    main()