from __future__ import annotations

import argparse
import configparser
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


# -----------------------------
# Helpers
# -----------------------------
def k(s: str) -> str:
    """normalize key"""
    return (s or "").strip().lower().replace(" ", "_")


def v(s: str) -> str:
    return (s or "").strip()


def find_index_file(exp_root: Path, user_path: Optional[str]) -> Path:
    """Pick mapping file:
       - if user provided --index_csv, use it
       - else try common defaults inside exp_root
    """
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            p = (exp_root / p)
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {p}")
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

    nearby = sorted([p.name for p in exp_root.glob("*.csv")])
    raise FileNotFoundError(
        f"No mapping/index file found inside: {exp_root}\n"
        f"Looked for: positions_index.csv / position_map.csv (etc)\n"
        f"CSV files found: {nearby if nearby else 'NONE'}\n"
        f"Pass it explicitly with --index_csv <file>"
    )


def load_index(index_path: Path) -> List[Dict[str, str]]:
    """Load CSV or JSON; normalize keys."""
    if index_path.suffix.lower() == ".json":
        data = json.loads(index_path.read_text(encoding="utf-8"))
        rows = []
        for r in data:
            rr = {k(str(kk)): v(str(vv)) for kk, vv in r.items()}
            rows.append(rr)
        return rows

    rows = []
    with index_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header row: {index_path}")
        for r in reader:
            rr = {k(h): v(r.get(h, "")) for h in reader.fieldnames}
            rows.append(rr)
    return rows


def get_first(row: Dict[str, str], *keys: str, default: str = "") -> str:
    """Return first existing non-empty value from given keys (already normalized)."""
    for kk in keys:
        val = row.get(k(kk), "")
        if val != "":
            return val
    return default


def compute_position_folder(row: Dict[str, str]) -> str:
    """
    Accept many schemas:
      - position_folder: "Position_7"
      - position: "7"
      - position_number: "7"
      - Position: "7" (will normalize to "position")
    """
    pf = get_first(row, "position_folder", "position_dir", "position_name")
    if pf:
        return pf

    num = get_first(row, "position_number", "position")
    if num and num.isdigit():
        return f"Position_{int(num)}"

    raw = get_first(row, "Position")
    if raw:
        raw = raw.strip()
        if raw.lower().startswith("position_"):
            return raw

    raise KeyError(
        "Mapping row does not contain position_folder or a numeric Position.\n"
        f"Row keys: {sorted(row.keys())}"
    )


def pretty_row(row: Dict[str, str]) -> str:
    ff = get_first(row, "field_folder", "field", "parent_folder")
    fn = get_first(row, "field_name", "fieldname")
    loc = get_first(row, "location", "letter", "site")
    ch = get_first(row, "channel", "channel_name", "ch")
    pos = compute_position_folder(row)
    if ff and fn and ch:
        return f"{ff}_{fn}_{ch} -> {pos}"
    if ff and loc and ch:
        return f"{ff}_{loc}_{ch} -> {pos}"
    return f"(unknown selection) -> {pos}"


# -----------------------------
# Selection logic
# -----------------------------
def filter_rows(rows: List[Dict[str, str]], select: str, run_all: bool, channel_only: str) -> List[Dict[str, str]]:
    if run_all:
        return rows

    if channel_only:
        ch = k(channel_only)
        return [r for r in rows if k(get_first(r, "channel", "channel_name", "ch")) == ch]

    if not select:
        raise SystemExit("Use --all OR --channel OR --select")

    sel = select.strip()

    if "|" in sel:
        parts = [p.strip() for p in sel.split("|")]
        if len(parts) != 3:
            raise SystemExit(f'Bad selection_key "{sel}". Expected: field_folder|field_name|channel')
        ff, fn, ch = parts
        out = [
            r for r in rows
            if k(get_first(r, "field_folder", "field", "parent_folder")) == k(ff)
            and k(get_first(r, "field_name")) == k(fn)
            and k(get_first(r, "channel", "channel_name", "ch")) == k(ch)
        ]
        return out

    toks = sel.split("_")
    if len(toks) < 3:
        raise SystemExit(f'Bad --select "{sel}". Use: field1_B3_1_NIR or field1|B3_1|NIR')

    field_folder = toks[0]
    channel = toks[-1]

    if len(toks) == 4:
        location = toks[1]
        rep = toks[2]
        field_name = f"{location}_{rep}"
        out = [
            r for r in rows
            if k(get_first(r, "field_folder", "field", "parent_folder")) == k(field_folder)
            and k(get_first(r, "field_name")) == k(field_name)
            and k(get_first(r, "channel", "channel_name", "ch")) == k(channel)
        ]
        return out

    location = toks[1]
    out = [
        r for r in rows
        if k(get_first(r, "field_folder", "field", "parent_folder")) == k(field_folder)
        and k(get_first(r, "location", "letter", "site")) == k(location)
        and k(get_first(r, "channel", "channel_name", "ch")) == k(channel)
    ]
    return out


# -----------------------------
# INI selection + writing
# -----------------------------
def choose_ini_template(ini_dir: Path, channel: str) -> Path:
    ch = channel.strip()
    ch_low = ch.lower()

    exact = ini_dir / f"acdc_segm_track_workflow_{ch}.ini"
    if exact.exists():
        return exact

    inis = list(ini_dir.glob("*.ini"))
    if not inis:
        raise FileNotFoundError(f"No .ini files in: {ini_dir}")

    matches = [p for p in inis if p.name.lower().endswith(f"_{ch_low}.ini")]
    if not matches:
        raise FileNotFoundError(
            f'No INI matched channel="{ch}". Expected like acdc_segm_track_workflow_{ch}.ini in {ini_dir}'
        )
    matches.sort(key=lambda p: len(p.name))
    return matches[0]


def write_per_position_ini(template_ini: Path, out_ini: Path, position_folder_abs: Path) -> None:
    """
    FIXED:
    Cell-ACDC CLI uses [paths_to_segment] and [paths_to_track] sections.
    If you only patch [paths_info], Cell-ACDC will still run on the old path.
    """
    cp = configparser.ConfigParser()
    cp.optionxform = str  # keep case
    cp.read(template_ini)

    # REQUIRED sections for CLI workflows
    for section in ("paths_to_segment", "paths_to_track"):
        if section not in cp.sections():
            cp[section] = {}
        cp[section]["paths"] = "\n" + str(position_folder_abs)

    # Optional: keep also paths_info updated if present
    if "paths_info" in cp.sections():
        cp["paths_info"]["paths"] = "\n" + str(position_folder_abs)

    out_ini.parent.mkdir(parents=True, exist_ok=True)
    with out_ini.open("w", encoding="utf-8") as f:
        cp.write(f)


def run_acdc(acdc_cmd: str, ini_path: Path, yes: bool, dry_run: bool) -> int:
    cmd = [acdc_cmd, "-p", str(ini_path)]
    if yes:
        cmd.append("-y")
    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return 0
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="segmentation_input", help="folder containing Position_1, Position_2, ...")
    ap.add_argument("--index_csv", default="", help="mapping file (position_map.csv / positions_index.csv / .json)")
    ap.add_argument("--ini_dir", required=True, help="folder containing acdc_segm_track_workflow_<CHANNEL>.ini files")
    ap.add_argument("--acdc_cmd", default="acdc", help='Cell-ACDC CLI command (default "acdc")')

    ap.add_argument("--all", action="store_true")
    ap.add_argument("--channel", default="")
    ap.add_argument("--select", default="")
    ap.add_argument("--list", action="store_true")

    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    exp_root = Path(args.exp_root).resolve()
    ini_dir = Path(args.ini_dir).resolve()

    index_path = find_index_file(exp_root, args.index_csv if args.index_csv else None)
    rows = load_index(index_path)

    if args.list:
        print(f"Mapping file: {index_path}")
        print(f"Entries: {len(rows)}\n")
        for r in rows:
            print(pretty_row(r))
        return

    selected = filter_rows(rows, select=args.select, run_all=args.all, channel_only=args.channel)

    if not selected:
        print(f"No rows matched. Mapping file: {index_path}")
        print("Try: --list")
        raise SystemExit(1)

    if args.select and len(args.select.split("_")) == 3 and len(selected) > 1:
        print(f'Ambiguous selection "{args.select}". Matches:')
        for r in selected:
            print("  ", pretty_row(r))
        raise SystemExit('Use explicit replicate: field1_B3_1_NIR or field1_B3_2_NIR')

    runs_dir = exp_root / "_cli_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for r in selected:
        ch = get_first(r, "channel", "channel_name", "ch")
        if not ch:
            raise KeyError(f"Row is missing channel. Row keys: {sorted(r.keys())}")

        pos_folder = compute_position_folder(r)  # Position_#
        position_folder_abs = exp_root / pos_folder

        if not position_folder_abs.exists():
            raise FileNotFoundError(f"Position folder not found: {position_folder_abs}")

        template_ini = choose_ini_template(ini_dir, ch)

        ff = get_first(r, "field_folder", "field", "parent_folder", default="field")
        fn = get_first(r, "field_name", default=get_first(r, "location", default="loc"))
        out_ini = runs_dir / f"{ff}_{fn}_{ch}__{pos_folder}.ini"

        write_per_position_ini(template_ini, out_ini, position_folder_abs)
        rc = run_acdc(args.acdc_cmd, out_ini, yes=args.yes, dry_run=args.dry_run)
        if rc != 0:
            failures += 1
            print(f"[ERROR] exit={rc} for {out_ini}")

    print("\nDone.")
    print("Selected:", len(selected))
    print("Failures:", failures)
    print("Generated INIs:", runs_dir)


if __name__ == "__main__":
    main()