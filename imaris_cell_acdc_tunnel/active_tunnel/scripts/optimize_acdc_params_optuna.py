#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_acdc_params_optuna.py
================================
Two-phase Optuna optimisation of Cell-ACDC segmentation + tracking parameters
against an Imaris ground-truth Excel file.

Phase 1 — Segmentation  (slow: reruns ACDC each trial)
  Optimises: gauss_sigma, threshold_method, min_area
  Tracking params stay at base INI values.
  Best segm.npz is saved to optuna_trials/best_phase1_segm.npz.

Phase 2 — Tracking only  (fast: skips ACDC, reuses best segm.npz)
  Optimises: search_range, memory
  Uses segm.npz saved by Phase 1 (or current segm.npz if --phase 2).

--phase both  runs Phase 1 then Phase 2 automatically.

Usage (from project root, imaris_xls env)
------------------------------------------
  python scripts/optimize_acdc_params_optuna.py
      --exp-root  segmentation_input
      --select    field1_B2_1_NIR
      --gt        field1_B2_1_NIR_gt.xlsx
      --pixel-size-um  0.108
      --frame-interval-s  3600
      --n-trials  30
      --phase     both

  Or override position directory directly:
      --position-dir  segmentation_input/Position_1
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────
# Search spaces per phase
# ──────────────────────────────────────────────────────────────────
SPACE_PHASE1 = {
    # Thresholding model parameters
    "gauss_sigma":       ("float",       0.5,    5.0),
    # For "categorical", the second element is the list of choices (third is unused)
    "threshold_method":  ("categorical",
                          ["threshold_otsu", "threshold_li", "threshold_triangle",
                           "threshold_isodata", "threshold_yen"],
                          None),
    # min_area goes into [standard_postprocess_features]
    "min_area":          ("int",         10,     500),
}

SPACE_PHASE2 = {
    "search_range": ("float",  5.0,  60.0),
    "memory":       ("int",    0,    10),
}

_POSITION_FEATURES = ["Position X", "Position Y"]

_REGISTRY = [
    ("Position X",           ["Position"],               "Position X", "Position X"),
    ("Position Y",           ["Position"],               "Position Y", "Position Y"),
    ("Area",                 ["Area"],                   "Value",      "Value"),
    ("Displacement²",        ["Displacement^2"],         "Value",      "Value"),
    ("Sphericity",           ["Sphericity"],             "Value",      "Value"),
    ("Ellipsoid Axis Len A", ["Ellipsoid Axis Length A"],"Value",      "Value"),
    ("Ellipsoid Axis Len B", ["Ellipsoid Axis Length B"],"Value",      "Value"),
    ("Intensity Mean",       ["Intensity Mean Ch=1"],    "Value",      "Value"),
]


# ──────────────────────────────────────────────────────────────────
# Position resolution from position_map.csv
# ──────────────────────────────────────────────────────────────────
def _k(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def resolve_position_dir(exp_root: Path, select: str) -> Path:
    """
    Look up position_map.csv in exp_root and find the Position_N folder
    that matches the select key (e.g. field1_B2_1_NIR).
    """
    map_path = exp_root / "position_map.csv"
    if not map_path.exists():
        raise FileNotFoundError(f"position_map.csv not found in {exp_root}")

    toks = select.strip().split("_")
    if len(toks) < 3:
        raise ValueError(f"Bad --select '{select}'. Expected e.g. field1_B2_1_NIR")

    field_folder = toks[0]
    channel      = toks[-1]
    # field_name is middle tokens joined: B2_1
    field_name   = "_".join(toks[1:-1])

    with map_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rk = {_k(k): str(v).strip() for k, v in row.items()}
            ff  = rk.get("field_folder", rk.get("field", ""))
            fn  = rk.get("field_name",   "")
            loc = rk.get("location", "")
            ch  = rk.get("channel", rk.get("channel_name", ""))
            pos_num = rk.get("position", rk.get("position_number", ""))

            folder_match   = _k(ff) == _k(field_folder)
            channel_match  = _k(ch) == _k(channel)
            # accept either full field_name (e.g. D2_1) or just location (e.g. D2)
            name_match = (_k(fn) == _k(field_name) or _k(loc) == _k(field_name))

            if folder_match and channel_match and name_match:
                if pos_num.isdigit():
                    return exp_root / f"Position_{int(pos_num)}"
                raise ValueError(f"Non-numeric position number: {pos_num}")

    raise ValueError(
        f"No match for select='{select}' in {map_path}.\n"
        f"Available entries: check position_map.csv."
    )


# ──────────────────────────────────────────────────────────────────
# INI helpers
# ──────────────────────────────────────────────────────────────────
def load_ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(str(path))
    return cp


def write_trial_ini(base_ini: Path, out_ini: Path,
                    position_dir: Path, params: dict) -> None:
    cp = load_ini(base_ini)

    # ACDC expects the path to the TIF file, not the position directory
    tif_path = find_tif(position_dir)
    if tif_path is None:
        raise FileNotFoundError(f"No TIF found in {position_dir}")
    path_str = str(tif_path.resolve())

    seg_keys = {"gauss_sigma", "threshold_method"}
    trk_keys = {"search_range", "memory"}

    for k, val in params.items():
        if k in seg_keys and "segmentation_model_params" in cp.sections():
            cp["segmentation_model_params"][k] = str(val)
        elif k == "min_area" and "standard_postprocess_features" in cp.sections():
            cp["standard_postprocess_features"][k] = str(int(val))
        if k in trk_keys and "tracker_params" in cp.sections():
            cp["tracker_params"][k] = str(val)

    for section in ("paths_to_segment", "paths_to_track", "paths_info"):
        if section not in cp.sections():
            cp[section] = {}
        cp[section]["paths"] = "\n" + path_str

    # Headless safety: never ask the user to draw an ROI
    if "initialization" in cp.sections():
        cp["initialization"]["use_ROI"] = "False"

    # Skip heavy measurements — we only need segm.npz + tracking for scoring
    for meas_sec in ("measurements",):
        if meas_sec in cp.sections():
            cp.remove_section(meas_sec)

    out_ini.parent.mkdir(parents=True, exist_ok=True)
    with out_ini.open("w", encoding="utf-8") as f:
        cp.write(f)


# ──────────────────────────────────────────────────────────────────
# Subprocess helpers
# ──────────────────────────────────────────────────────────────────
def run_acdc(acdc_cmd: str, ini_path: Path,
             acdc_env: str = "", conda_bat: str = "") -> bool:
    """Run the ACDC CLI without measurements (trial inis have [measurements] stripped).

    Strategy (in order of preference):
    1. If conda_bat contains a path to a Python executable (ends in python.exe / python)
       → call it directly:  <python> -m cellacdc -p ini -y
       This is the fastest and most reliable approach — no conda activation needed.
    2. If conda_bat points to _conda.exe
       → use  _conda.exe run -p <env_path> acdc -p ini -y
    3. If acdc_env is given but no conda_bat
       → try  conda run -n acdc_env acdc -p ini -y
    4. Otherwise fall back to acdc_cmd directly.

    Recommended: pass --conda-bat to the Python executable from the acdc env, e.g.
        --conda-bat C:/Users/.../anaconda3/envs/acdc/python.exe
    """
    import time as _time
    import threading as _threading
    ini_str = str(ini_path)

    # Build environment — ensure the acdc env's DLL directories are on PATH
    # (needed on Windows when spawning from a different Python environment)
    import os as _os
    env = _os.environ.copy()
    if conda_bat:
        cname = Path(conda_bat).name.lower()
        if "python" in cname:
            # Use the acdc entry-point script — NOT python -m cellacdc
            # (__main__.py has no module-level run() call, so -m does nothing)
            scripts_dir = Path(conda_bat).parent / "Scripts"
            acdc_exe = scripts_dir / "acdc.exe"
            if not acdc_exe.exists():
                acdc_exe = scripts_dir / "acdc"  # Linux/Mac
            cmd = [str(acdc_exe), "-p", ini_str, "-y"]
            # Add the acdc env's library paths so Qt/MKL DLLs are found
            env_dir = str(Path(conda_bat).parent)
            dll_dirs = [
                env_dir,
                str(scripts_dir),
                str(Path(conda_bat).parent / "Library" / "bin"),
                str(Path(conda_bat).parent / "Library" / "mingw-w64" / "bin"),
                str(Path(conda_bat).parent / "Library" / "usr" / "bin"),
            ]
            existing_path = env.get("PATH", "")
            env["PATH"] = _os.pathsep.join(dll_dirs) + _os.pathsep + existing_path
        elif "_conda" in cname:
            # _conda.exe with -p <full_path>
            env_path = str(Path(conda_bat).parent / "envs" / acdc_env)
            cmd = [conda_bat, "run", "-p", env_path, "acdc", "-p", ini_str, "-y"]
        else:
            # conda.bat — must use cmd /c on Windows
            if sys.platform == "win32":
                bat_w = conda_bat.replace("/", "\\")
                cmd = ["cmd", "/c", bat_w, "run", "-n", acdc_env,
                       "--no-capture-output", "acdc", "-p", ini_str, "-y"]
            else:
                cmd = [conda_bat, "run", "-n", acdc_env,
                       "--no-capture-output", "acdc", "-p", ini_str, "-y"]
    elif acdc_env:
        # No conda_bat — hope 'conda' is on PATH
        cmd = ["conda", "run", "-n", acdc_env,
               "--no-capture-output", "acdc", "-p", ini_str, "-y"]
    else:
        cmd = [acdc_cmd, "-p", ini_str, "-y"]

    # Heartbeat thread — prints every 30 s so the user can confirm ACDC is alive
    _stop = _threading.Event()
    def _hb():
        t = 0
        while not _stop.wait(30):
            t += 30
            print(f"    [acdc] still running... {t}s", flush=True)
    _hb_thread = _threading.Thread(target=_hb, daemon=True)
    _hb_thread.start()

    t0 = _time.time()
    try:
        result = subprocess.run(cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL,
                                env=env,
                                timeout=300)
    except subprocess.TimeoutExpired:
        _stop.set()
        print(f"    [acdc] TIMEOUT after 300 s -- skipping trial")
        return False
    finally:
        _stop.set()

    elapsed = _time.time() - t0

    def _safe(b, n=300):
        return (b or b"").decode("ascii", errors="ignore").strip()[:n]

    if result.returncode != 0:
        print(f"    [acdc] FAILED exit={result.returncode}  ({elapsed:.0f}s)")
        err = _safe(result.stderr, 400)
        if err:
            print(f"    [acdc] STDERR: {err}")
        return False
    print(f"    [acdc] done in {elapsed:.0f}s")
    return True


def run_export(python_exe: str, converter_py: Path,
               segm_npz: Path, tif_path: Path, out_xlsx: Path,
               pixel_size_um: float, z_step_um: float,
               frame_interval_s: float, time_index_start: int,
               offset_x_px: float, offset_y_px: float) -> bool:
    cmd = [
        python_exe, str(converter_py),
        "--segm-npz",           str(segm_npz),
        "--tif",                str(tif_path),
        "--out",                str(out_xlsx),
        "--pixel-size-um",      str(pixel_size_um),
        "--z-step-um",          str(z_step_um),
        "--frame-interval-s",   str(frame_interval_s),
        "--time-index-start",   str(time_index_start),
        "--imaris-offset-x-px", str(offset_x_px),
        "--imaris-offset-y-px", str(offset_y_px),
        "--displacement2-mode", "from_start",
    ]
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0


# ──────────────────────────────────────────────────────────────────
# File discovery inside a position dir
# ──────────────────────────────────────────────────────────────────
def find_segm_npz(position_dir: Path) -> Optional[Path]:
    images_dir = position_dir / "Images"
    if not images_dir.exists():
        images_dir = position_dir
    cands = [p for p in images_dir.glob("*segm*.npz")
             if "aligned" not in p.name.lower() and "bkgr" not in p.name.lower()]
    if not cands:
        cands = list(images_dir.glob("*.npz"))
    return sorted(cands)[-1] if cands else None


def find_tif(position_dir: Path) -> Optional[Path]:
    images_dir = position_dir / "Images"
    if not images_dir.exists():
        images_dir = position_dir
    tifs = [p for p in sorted(images_dir.glob("*.tif*"))
            if "mask" not in p.name.lower() and "segm" not in p.name.lower()]
    return tifs[0] if tifs else None


# ──────────────────────────────────────────────────────────────────
# Inline scoring (no subprocess)
# ──────────────────────────────────────────────────────────────────
def _nc(s: str) -> str:
    import re
    return re.sub(r"[\s_\-\^\(\)=²]+", "", str(s)).lower()


def _find_col(df: pd.DataFrame, *candidates) -> Optional[str]:
    norm = {_nc(c): c for c in df.columns}
    for c in candidates:
        hit = norm.get(_nc(c))
        if hit:
            return hit
    return None


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _load_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    engine = "xlrd" if str(path).lower().endswith(".xls") else "openpyxl"
    xl = pd.ExcelFile(str(path), engine=engine)
    out = {}
    for name in xl.sheet_names:
        try:
            df = xl.parse(name)
            df.columns = [str(c).strip() for c in df.columns]
            out[name] = df
        except Exception:
            pass
    return out


def _find_sheet(sheets, *candidates) -> Optional[pd.DataFrame]:
    norm = {_nc(k): k for k in sheets}
    for c in candidates:
        hit = norm.get(_nc(c))
        if hit:
            return sheets[hit].copy()
    return None


def _track_centroids(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    for sheet_name in ("Track Position", "Position"):
        df = _find_sheet(sheets, sheet_name)
        if df is None:
            continue
        xcol = _find_col(df, "Track Position X Mean", "Position X", "X")
        ycol = _find_col(df, "Track Position Y Mean", "Position Y", "Y")
        par  = _find_col(df, "Parent", "ID")
        if None in (xcol, ycol, par):
            continue
        df[xcol] = _to_num(df[xcol])
        df[ycol] = _to_num(df[ycol])
        df[par]  = _to_num(df[par])
        df = df.dropna(subset=[xcol, ycol, par])
        grp = df.groupby(par)[[xcol, ycol]].mean().reset_index()
        grp.columns = ["Parent", "cx", "cy"]
        return grp.reset_index(drop=True)
    return pd.DataFrame(columns=["Parent", "cx", "cy"])


def _hungarian_match(pipe_c: pd.DataFrame, imar_c: pd.DataFrame,
                     ox_um: float, oy_um: float,
                     max_dist_um: float) -> pd.DataFrame:
    if pipe_c.empty or imar_c.empty:
        return pd.DataFrame(columns=["pipe_parent", "imar_parent"])
    P = pipe_c[["cx", "cy"]].values.copy().astype(float)
    P[:, 0] += ox_um; P[:, 1] += oy_um
    I = imar_c[["cx", "cy"]].values.astype(float)
    # Guard: drop rows with NaN coordinates (NaN in linear_sum_assignment = segfault)
    p_valid = ~np.isnan(P).any(axis=1)
    i_valid = ~np.isnan(I).any(axis=1)
    if not p_valid.any() or not i_valid.any():
        return pd.DataFrame(columns=["pipe_parent", "imar_parent"])
    P = P[p_valid]; pipe_c = pipe_c.iloc[np.where(p_valid)[0]].reset_index(drop=True)
    I = I[i_valid]; imar_c = imar_c.iloc[np.where(i_valid)[0]].reset_index(drop=True)
    dist = np.sqrt(((P[:, None] - I[None]) ** 2).sum(2))
    cap  = dist.copy()
    cap[cap > max_dist_um] = 1e9
    cap = np.nan_to_num(cap, nan=1e9, posinf=1e9)  # final safety for linear_sum_assignment
    ri, ci = linear_sum_assignment(cap)
    mask = dist[ri, ci] <= max_dist_um
    return pd.DataFrame({
        "pipe_parent": pipe_c.iloc[ri[mask]]["Parent"].values,
        "imar_parent": imar_c.iloc[ci[mask]]["Parent"].values,
    })


def _remap(pipe_sheets: Dict[str, pd.DataFrame],
           track_map: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if track_map.empty:
        return pipe_sheets
    id_map = dict(zip(_to_num(track_map["pipe_parent"]),
                      _to_num(track_map["imar_parent"])))
    out = {}
    for name, df in pipe_sheets.items():
        df = df.copy()
        pc = _find_col(df, "Parent", "ID")
        if pc is None:
            out[name] = df; continue
        new_ids = _to_num(df[pc]).map(id_map)
        df = df[new_ids.notna()].copy()
        df[pc] = new_ids[new_ids.notna()].astype(int).values
        out[name] = df
    return out


def _r2(pipe: np.ndarray, gt: np.ndarray) -> float:
    mask = np.isfinite(pipe) & np.isfinite(gt)
    n = int(mask.sum())
    if n < 2:
        return np.nan
    x = gt[mask].astype(float)
    y = pipe[mask].astype(float)
    # Pearson r² — pure numpy, no BLAS/LAPACK (avoids DLL crash on Windows)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt((xm * xm).sum() * (ym * ym).sum()))
    if denom == 0.0:
        return np.nan
    r = float((xm * ym).sum()) / denom
    return r * r


def compute_score(pipe_sheets: Dict[str, pd.DataFrame],
                  gt_sheets:   Dict[str, pd.DataFrame],
                  pixel_size_um: float,
                  max_dist_um:   float,
                  objective:     str) -> Tuple[float, int]:
    """Returns (score, n_matched)."""
    pipe_c = _track_centroids(pipe_sheets)
    imar_c = _track_centroids(gt_sheets)

    # offset grid search ±10 px at 1 px steps
    best_ox, best_oy, best_n = 0.0, 0.0, 0
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            tm = _hungarian_match(pipe_c, imar_c,
                                  dx * pixel_size_um, dy * pixel_size_um,
                                  max_dist_um)
            if len(tm) > best_n:
                best_n = len(tm); best_ox = dx * pixel_size_um; best_oy = dy * pixel_size_um

    track_map = _hungarian_match(pipe_c, imar_c, best_ox, best_oy, max_dist_um)
    n_matched  = len(track_map)

    if objective == "n_matched":
        return float(n_matched), n_matched

    pipe_r = _remap(pipe_sheets, track_map)
    r2s    = []

    for label, sheet_names, pc, gc in _REGISTRY:
        if objective == "position_r2" and label not in _POSITION_FEATURES:
            continue
        p_df = _find_sheet(pipe_r,    *sheet_names)
        g_df = _find_sheet(gt_sheets, *sheet_names)
        if p_df is None or g_df is None:
            continue
        tc_p = _find_col(p_df, "Time"); par_p = _find_col(p_df, "Parent")
        vc_p = _find_col(p_df, pc, "Value")
        tc_g = _find_col(g_df, "Time"); par_g = _find_col(g_df, "Parent")
        vc_g = _find_col(g_df, gc, "Value")
        if None in (tc_p, par_p, vc_p, tc_g, par_g, vc_g):
            continue
        pm = pd.DataFrame({"T": _to_num(p_df[tc_p]).astype("Int64"),
                            "P": _to_num(p_df[par_p]).astype("Int64"),
                            "v": _to_num(p_df[vc_p])}).dropna(subset=["T","P"])
        gm = pd.DataFrame({"T": _to_num(g_df[tc_g]).astype("Int64"),
                            "P": _to_num(g_df[par_g]).astype("Int64"),
                            "v": _to_num(g_df[vc_g])}).dropna(subset=["T","P"])
        merged = pm.merge(gm, on=["T","P"], how="inner", suffixes=("_p","_g"))
        if merged.empty:
            continue
        v = _r2(merged["v_p"].to_numpy(), merged["v_g"].to_numpy())
        if np.isfinite(v):
            r2s.append(v)

    return (float(np.mean(r2s)) if r2s else -1.0), n_matched


# ──────────────────────────────────────────────────────────────────
# Phase 1 — segmentation optimisation
# ──────────────────────────────────────────────────────────────────
def run_phase1(base_ini: Path, position_dir: Path, gt_sheets: dict,
               out_dir: Path, python_exe: str, converter_py: Path,
               acdc_cmd: str, pixel_size_um: float, z_step_um: float,
               frame_interval_s: float, time_index_start: int,
               offset_x_px: float, offset_y_px: float,
               objective: str, n_trials: int,
               study_name: str, storage: Optional[str],
               acdc_env: str = "", conda_bat: str = "") -> optuna.Study:

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Segmentation optimisation  ({n_trials} trials)")
    print(f"{'='*60}")

    best_score: List[float] = [-np.inf]
    best_segm_dst = out_dir / "best_phase1_segm.npz"

    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(
        study_name=f"{study_name}_phase1",
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    OFFSET_RANGE = 2.0   # ± pixels around the centre offset

    def objective_fn(trial: optuna.Trial) -> float:
        try:
            return _objective_fn_inner(trial)
        except BaseException as exc:
            import traceback as _tb
            msg = _tb.format_exc().encode("ascii", errors="replace").decode("ascii")
            print(f"    [trial {trial.number}] ERROR: {msg[:800]}", flush=True)
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            return -1.0

    def _objective_fn_inner(trial: optuna.Trial) -> float:
        params = {}
        for name, (kind, lo, hi) in SPACE_PHASE1.items():
            if kind == "float":
                params[name] = trial.suggest_float(name, lo, hi)
            elif kind == "int":
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, lo)  # lo = list of choices

        # offset search centred on the provided baseline ± OFFSET_RANGE
        trial_ox = trial.suggest_float("offset_x", offset_x_px - OFFSET_RANGE,
                                                     offset_x_px + OFFSET_RANGE)
        trial_oy = trial.suggest_float("offset_y", offset_y_px - OFFSET_RANGE,
                                                     offset_y_px + OFFSET_RANGE)

        trial_dir = out_dir / f"p1_trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [P1 trial {trial.number}] "
              + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in params.items())
              + f"  ox={trial_ox:+.2f}  oy={trial_oy:+.2f}")

        trial_ini = trial_dir / "trial.ini"
        write_trial_ini(base_ini, trial_ini, position_dir, params)

        if not run_acdc(acdc_cmd, trial_ini, acdc_env=acdc_env, conda_bat=conda_bat):
            return -1.0

        segm_npz = find_segm_npz(position_dir)
        tif_path = find_tif(position_dir)
        if segm_npz is None or tif_path is None:
            print("    [trial] segm_npz or tif not found after ACDC run")
            return -1.0

        out_xlsx = trial_dir / "pipeline.xlsx"
        if not run_export(python_exe, converter_py, segm_npz, tif_path,
                          out_xlsx, pixel_size_um, z_step_um, frame_interval_s,
                          time_index_start, trial_ox, trial_oy):
            print("    [trial] converter FAILED", flush=True)
            return -1.0

        pipe_sheets = _load_sheets(out_xlsx)
        score, n_matched = compute_score(pipe_sheets, gt_sheets,
                                         pixel_size_um, 15.0, objective)

        print(f"    -> score={score:.4f}  matched={n_matched}", flush=True)

        # save best segm.npz
        if score > best_score[0]:
            best_score[0] = score
            shutil.copy2(segm_npz, best_segm_dst)
            print(f"    -> NEW BEST -- segm.npz saved")

        (trial_dir / "result.json").write_text(
            json.dumps({"trial": trial.number, "phase": 1,
                        "score": score, "n_matched": n_matched,
                        "offset_x": trial_ox, "offset_y": trial_oy,
                        **params},
                       indent=2), encoding="utf-8")
        return score

    study.optimize(objective_fn, n_trials=n_trials,
                   show_progress_bar=True)
    return study


# ──────────────────────────────────────────────────────────────────
# Phase 2 — tracking optimisation  (no ACDC re-run)
# ──────────────────────────────────────────────────────────────────
def run_phase2(base_ini: Path, position_dir: Path, gt_sheets: dict,
               out_dir: Path, python_exe: str, converter_py: Path,
               pixel_size_um: float, z_step_um: float,
               frame_interval_s: float, time_index_start: int,
               offset_x_px: float, offset_y_px: float,
               objective: str, n_trials: int,
               study_name: str, storage: Optional[str],
               fixed_segm_npz: Optional[Path] = None) -> optuna.Study:

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Tracking optimisation  ({n_trials} trials)")
    print(f"  (ACDC skipped — reusing segm.npz)")
    print(f"{'='*60}")

    # determine which segm.npz to use
    if fixed_segm_npz is None:
        # try best from phase 1
        fixed_segm_npz = out_dir / "best_phase1_segm.npz"
        if not fixed_segm_npz.exists():
            fixed_segm_npz = find_segm_npz(position_dir)
    if fixed_segm_npz is None or not fixed_segm_npz.exists():
        raise FileNotFoundError("No segm.npz found for Phase 2. "
                                "Run Phase 1 first or provide --segm-npz.")

    tif_path = find_tif(position_dir)
    if tif_path is None:
        raise FileNotFoundError(f"No TIF found in {position_dir}")

    print(f"  Using segm.npz: {fixed_segm_npz}")
    print(f"  Using TIF     : {tif_path}")

    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(
        study_name=f"{study_name}_phase2",
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    OFFSET_RANGE = 2.0   # ± pixels around the centre offset

    def objective_fn(trial: optuna.Trial) -> float:
        params = {}
        for name, (kind, lo, hi) in SPACE_PHASE2.items():
            params[name] = (trial.suggest_float(name, lo, hi)
                            if kind == "float"
                            else trial.suggest_int(name, int(lo), int(hi)))

        # offset search centred on the provided baseline ± OFFSET_RANGE
        trial_ox = trial.suggest_float("offset_x", offset_x_px - OFFSET_RANGE,
                                                     offset_x_px + OFFSET_RANGE)
        trial_oy = trial.suggest_float("offset_y", offset_y_px - OFFSET_RANGE,
                                                     offset_y_px + OFFSET_RANGE)

        trial_dir = out_dir / f"p2_trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [P2 trial {trial.number}] "
              + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in params.items())
              + f"  ox={trial_ox:+.2f}  oy={trial_oy:+.2f}")

        # write trial INI (only tracking params change)
        trial_ini = trial_dir / "trial.ini"
        write_trial_ini(base_ini, trial_ini, position_dir, params)

        # patch tracker export path so trackpy CSV goes to trial_dir
        cp = load_ini(trial_ini)
        if "tracker_params" in cp.sections():
            orig_export = cp["tracker_params"].get("export_to", "")
            if orig_export:
                cp["tracker_params"]["export_to"] = str(
                    trial_dir / Path(orig_export).name)
        with trial_ini.open("w", encoding="utf-8") as f:
            cp.write(f)

        out_xlsx = trial_dir / "pipeline.xlsx"
        if not run_export(python_exe, converter_py, fixed_segm_npz, tif_path,
                          out_xlsx, pixel_size_um, z_step_um, frame_interval_s,
                          time_index_start, trial_ox, trial_oy):
            return -1.0

        pipe_sheets = _load_sheets(out_xlsx)
        score, n_matched = compute_score(pipe_sheets, gt_sheets,
                                         pixel_size_um, 15.0, objective)

        print(f"    -> score={score:.4f}  matched={n_matched}")

        (trial_dir / "result.json").write_text(
            json.dumps({"trial": trial.number, "phase": 2,
                        "score": score, "n_matched": n_matched,
                        "offset_x": trial_ox, "offset_y": trial_oy,
                        **params},
                       indent=2), encoding="utf-8")
        return score

    study.optimize(objective_fn, n_trials=n_trials,
                   show_progress_bar=True)
    return study


# ──────────────────────────────────────────────────────────────────
# Summary + best INI
# ──────────────────────────────────────────────────────────────────
def save_summary(studies: List[Tuple[str, optuna.Study]], out_dir: Path):
    all_rows = []
    for phase_label, study in studies:
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {"phase": phase_label, "trial": t.number, "score": t.value}
            row.update(t.params)
            all_rows.append(row)
    if not all_rows:
        print("[summary] No completed trials.")
        return
    df = pd.DataFrame(all_rows).sort_values("score", ascending=False)
    path = out_dir / "optuna_summary.csv"
    df.to_csv(str(path), index=False)
    print(f"\n[summary] {len(df)} total trials -> {path}")


def write_best_ini(studies: List[Tuple[str, optuna.Study]],
                   base_ini: Path, position_dir: Path, out_ini: Path):
    """Merge best params from all phases into one INI."""
    merged_params = {}
    for _, study in studies:
        if study.best_trial is not None:
            merged_params.update(study.best_params)

    # offset_x / offset_y are scored params but not INI keys — separate them
    offset_params = {k: merged_params.pop(k)
                     for k in ("offset_x", "offset_y")
                     if k in merged_params}

    write_trial_ini(base_ini, out_ini, position_dir, merged_params)
    print(f"\n[best INI] written -> {out_ini}")
    print("[best params]")
    for k, v in merged_params.items():
        fmt = f"{v:.6g}" if isinstance(v, float) else str(v)
        print(f"  {k:<28} = {fmt}")
    if offset_params:
        print("[best offsets  (use these in compare / export)]")
        for k, v in offset_params.items():
            print(f"  {k:<28} = {v:+.4f} px")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Two-phase Optuna optimisation for Cell-ACDC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # position resolution
    ap.add_argument("--position-dir",  default="",
                    help="Direct path to Position_N folder (overrides --exp-root + --select)")
    ap.add_argument("--exp-root",      default="segmentation_input",
                    help="segmentation_input folder (default: segmentation_input)")
    ap.add_argument("--select",        default="",
                    help="Selection key e.g. field1_B2_1_NIR (used with --exp-root)")

    # required
    ap.add_argument("--gt",            required=True,
                    help="Imaris ground-truth Excel file")
    ap.add_argument("--base-ini",      required=True,
                    help="Base INI template e.g. inis/acdc_segm_track_workflow_NIR.ini")
    ap.add_argument("--pixel-size-um", type=float, required=True)

    # optional physics
    ap.add_argument("--z-step-um",        type=float, default=1.0)
    ap.add_argument("--frame-interval-s", type=float, default=3600.0)
    ap.add_argument("--time-index-start", type=int,   default=2)
    ap.add_argument("--offset-x",         type=float, default=-3.4)
    ap.add_argument("--offset-y",         type=float, default=-9.0)

    # optimisation
    ap.add_argument("--phase",    choices=["1","2","both"], default="both",
                    help="1=segmentation only, 2=tracking only, both (default: both)")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Trials per phase (default: 30)")
    ap.add_argument("--objective",
                    choices=["position_r2","mean_r2","n_matched"],
                    default="position_r2")
    ap.add_argument("--segm-npz", default="",
                    help="Fixed segm.npz for phase 2 (default: best from phase 1)")

    # infrastructure
    ap.add_argument("--acdc-cmd",   default="acdc",
                    help="ACDC executable or 'acdc' (default). Overridden when --acdc-env is set.")
    ap.add_argument("--acdc-env",   default="",
                    help="Conda env that contains the acdc CLI (e.g. 'acdc'). "
                         "When set, ACDC is invoked via: conda run -n <env> acdc ...")
    ap.add_argument("--conda-bat",  default="",
                    help="Path to conda.bat (needed when --acdc-env is set on Windows)")
    ap.add_argument("--converter",  default="scripts/acdc_npz_tif_to_imaris_like.py")
    ap.add_argument("--python",     default="")
    ap.add_argument("--study-name", default="acdc_optuna")
    ap.add_argument("--storage",    default="",
                    help="Optuna storage URL (empty=in-memory, "
                         "e.g. sqlite:///optuna_trials/study.db)")
    ap.add_argument("--out-dir",    default="optuna_trials")
    args = ap.parse_args()

    # ── resolve paths ─────────────────────────────────────────────
    base_ini     = Path(args.base_ini).resolve()
    gt_path      = Path(args.gt).resolve()
    out_dir      = Path(args.out_dir).resolve()
    converter_py = Path(args.converter).resolve()
    python_exe   = args.python.strip() if args.python.strip() else sys.executable
    storage      = args.storage.strip() if args.storage.strip() else None

    if args.position_dir:
        position_dir = Path(args.position_dir).resolve()
    elif args.select and args.exp_root:
        exp_root     = Path(args.exp_root).resolve()
        position_dir = resolve_position_dir(exp_root, args.select)
    else:
        ap.error("Provide either --position-dir  OR  --exp-root + --select")

    for p, n in [(base_ini,"base-ini"), (gt_path,"gt"),
                 (position_dir,"position-dir"), (converter_py,"converter")]:
        if not p.exists():
            raise FileNotFoundError(f"--{n} not found: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_segm = Path(args.segm_npz).resolve() if args.segm_npz else None

    print(f"\n{'='*60}")
    print(f"  Cell-ACDC Optuna Optimiser")
    print(f"{'='*60}")
    print(f"  Phase          : {args.phase}")
    print(f"  Trials/phase   : {args.n_trials}")
    print(f"  Objective      : {args.objective}")
    print(f"  Position dir   : {position_dir}")
    print(f"  Ground truth   : {gt_path}")
    print(f"  Base INI       : {base_ini}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Offset X/Y px  : {args.offset_x:+.4f} / {args.offset_y:+.4f}")

    gt_sheets = _load_sheets(gt_path)

    acdc_env  = args.acdc_env.strip()
    conda_bat = args.conda_bat.strip()
    if acdc_env:
        print(f"  ACDC env       : {acdc_env}  (conda run)")
    else:
        print(f"  ACDC cmd       : {args.acdc_cmd}")

    common = dict(
        base_ini=base_ini, position_dir=position_dir, gt_sheets=gt_sheets,
        out_dir=out_dir, python_exe=python_exe, converter_py=converter_py,
        pixel_size_um=float(args.pixel_size_um),
        z_step_um=float(args.z_step_um),
        frame_interval_s=float(args.frame_interval_s),
        time_index_start=int(args.time_index_start),
        offset_x_px=float(args.offset_x),
        offset_y_px=float(args.offset_y),
        objective=args.objective,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=storage,
    )

    completed_studies: List[Tuple[str, optuna.Study]] = []

    if args.phase in ("1", "both"):
        s1 = run_phase1(acdc_cmd=args.acdc_cmd,
                        acdc_env=acdc_env, conda_bat=conda_bat, **common)
        completed_studies.append(("phase1", s1))

    if args.phase in ("2", "both"):
        s2 = run_phase2(fixed_segm_npz=fixed_segm, **common)
        completed_studies.append(("phase2", s2))

    save_summary(completed_studies, out_dir)
    write_best_ini(completed_studies, base_ini, position_dir,
                   out_dir / "best_NIR.ini")

    print(f"\n[done] results in: {out_dir}")


if __name__ == "__main__":
    main()
