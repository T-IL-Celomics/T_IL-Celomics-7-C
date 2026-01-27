#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visual compare Cell-ACDC vs Imaris positions (overlay dots) with an interactive frame slider.

- Reads ACDC CSV (frame_i, <channel>_PositionX_um, <channel>_PositionY_um)
- Reads Imaris .xls (auto-detects Position sheet, reads Time, PositionX, PositionY)
- Aligns time: Imaris_Time = ACDC_frame_i + imaris_time_offset
- Optionally applies translation to ACDC: x_corr = x + tx, y_corr = y + ty
- Shows overlay scatter; slider to move through frames

Fix included:
- After updating scatter offsets, explicitly re-autoscale axes so points become visible.
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons


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


# -------- Plot scaling helper --------
def autoscale_from_points(ax, arrays, pad_frac=0.05):
    """
    Set x/y limits from a list of (N,2) arrays.
    Matplotlib does NOT always autoscale when updating scatter offsets,
    so we force limits based on current points.
    """
    arrays = [a for a in arrays if isinstance(a, np.ndarray) and a.size]
    if not arrays:
        return

    pts = np.vstack(arrays)
    x = pts[:, 0]
    y = pts[:, 1]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    xr = xmax - xmin
    yr = ymax - ymin
    if xr == 0:
        xr = 1.0
    if yr == 0:
        yr = 1.0

    padx = xr * pad_frac
    pady = yr * pad_frac

    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)


# -------- Imaris helpers --------
def detect_imaris_position_sheet(imaris_xls: str):
    xf = pd.ExcelFile(imaris_xls, engine="xlrd")
    best_name, best_df, best_score = None, None, -1

    for sh in xf.sheet_names:
        if "position" not in sh.lower():
            continue
        if "track" in sh.lower():
            continue

        try:
            df = pd.read_excel(imaris_xls, sheet_name=sh, engine="xlrd")
        except Exception:
            continue

        df.columns = [clean_colname(c) for c in df.columns]
        cols = [_norm(c) for c in df.columns]

        score = 0
        for w in ["id", "time", "positionx", "positiony"]:
            if any(w in c for c in cols):
                score += 2
        if df.shape[0] > 1000:
            score += 1

        if score > best_score:
            best_score = score
            best_name, best_df = sh, df

    if best_df is None or best_score < 6:
        raise ValueError("Could not auto-detect Imaris per-object Position sheet.")
    return best_name, best_df


def extract_imaris_cols(df: pd.DataFrame):
    df.columns = [clean_colname(c) for c in df.columns]
    id_col = find_col_by_candidates(df, ["ID", "Id", "Object ID", "Track ID"])
    t_col = find_col_by_candidates(df, ["Time", "Time Index", "time", "t"])

    x_col = None
    y_col = None
    for c in df.columns:
        nc = _norm(c)
        if x_col is None and "positionx" in nc and "mean" not in nc:
            x_col = c
        if y_col is None and "positiony" in nc and "mean" not in nc:
            y_col = c

    if id_col is None or t_col is None or x_col is None or y_col is None:
        raise KeyError(f"Imaris missing needed columns. Found: {list(df.columns)[:60]}")

    return id_col, t_col, x_col, y_col


# -------- ACDC helpers --------
def load_acdc(acdc_csv: str, channel: str):
    df = pd.read_csv(acdc_csv, sep=None, engine="python")
    df.columns = [clean_colname(c) for c in df.columns]

    fcol = find_col_by_candidates(df, ["frame_i", "frame", "Frame"])
    if fcol is None:
        raise KeyError("ACDC missing frame_i column.")

    idcol = find_col_by_candidates(df, ["Cell_ID", "Cell ID", "cell_id"])
    if idcol is None:
        raise KeyError("ACDC missing Cell_ID column.")

    xcol = find_col_by_candidates(df, [f"{channel}_PositionX_um", "PositionX_um", "PosX_um"])
    ycol = find_col_by_candidates(df, [f"{channel}_PositionY_um", "PositionY_um", "PosY_um"])
    if xcol is None or ycol is None:
        raise KeyError(f"ACDC missing {channel}_PositionX_um / {channel}_PositionY_um.")

    df[fcol] = pd.to_numeric(df[fcol], errors="coerce")
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna(subset=[fcol, xcol, ycol]).copy()

    return df, fcol, idcol, xcol, ycol


def load_imaris(imaris_xls: str):
    sh, df = detect_imaris_position_sheet(imaris_xls)
    id_col, t_col, x_col, y_col = extract_imaris_cols(df)

    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[t_col, x_col, y_col]).copy()
    df[id_col] = df[id_col].astype(str)

    return sh, df, id_col, t_col, x_col, y_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc", required=True)
    ap.add_argument("--imaris", required=True)
    ap.add_argument("--channel", default="NIR")
    ap.add_argument("--frame", type=int, default=48, help="User frame (e.g. 48)")
    ap.add_argument("--frame-is-1based", action="store_true", help="If set, frame=48 => frame_i=47")
    ap.add_argument("--imaris-time-offset", type=int, default=1, help="Imaris_Time = ACDC_frame_i + offset")
    ap.add_argument("--tx", type=float, default=0.0, help="translation X (um) applied to ACDC")
    ap.add_argument("--ty", type=float, default=0.0, help="translation Y (um) applied to ACDC")
    ap.add_argument("--dot", type=float, default=8.0, help="dot size")
    ap.add_argument("--alpha", type=float, default=0.55, help="dot alpha")
    ap.add_argument("--no-autoscale", action="store_true", help="Disable autoscaling after updates")
    args = ap.parse_args()

    # Load once
    acdc_df, fcol, idcol, axcol, aycol = load_acdc(args.acdc, args.channel)
    im_sheet, im_df, im_idcol, im_tcol, im_xcol, im_ycol = load_imaris(args.imaris)

    frames = np.sort(acdc_df[fcol].unique())
    if frames.size == 0:
        raise RuntimeError("No frames found in ACDC file.")

    # frame conversion
    start_frame_i = (args.frame - 1) if args.frame_is_1based else args.frame
    if start_frame_i not in set(frames.tolist()):
        start_frame_i = int(frames[np.argmin(np.abs(frames - start_frame_i))])

    # plotting
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(bottom=0.20, right=0.82)

    ax.set_title(f"ACDC vs Imaris overlay | Imaris sheet: {im_sheet}")
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    # IMPORTANT: "box" avoids weird aspect/datalim behavior when we set limits ourselves.
    ax.set_aspect("equal", adjustable="box")

    # scatter handles
    sc_acdc = ax.scatter([], [], s=args.dot, alpha=args.alpha, label="ACDC (shifted)")
    sc_im = ax.scatter([], [], s=args.dot, alpha=args.alpha, label="Imaris")
    ax.legend(loc="upper left")

    # checkboxes (toggle visibility)
    rax = plt.axes([0.84, 0.65, 0.14, 0.20])
    check = CheckButtons(rax, ["Show ACDC", "Show Imaris"], [True, True])

    def update(frame_i: int):
        im_time = int(frame_i + args.imaris_time_offset)

        ac_sub = acdc_df[acdc_df[fcol] == frame_i]
        im_sub = im_df[im_df[im_tcol] == im_time]

        ac_xy = ac_sub[[axcol, aycol]].to_numpy(float)
        ac_xy = ac_xy + np.array([args.tx, args.ty], float)
        im_xy = im_sub[[im_xcol, im_ycol]].to_numpy(float)

        sc_acdc.set_offsets(ac_xy if ac_xy.size else np.zeros((0, 2)))
        sc_im.set_offsets(im_xy if im_xy.size else np.zeros((0, 2)))

        if not args.no_autoscale:
            autoscale_from_points(
                ax,
                [
                    ac_xy if sc_acdc.get_visible() else np.zeros((0, 2)),
                    im_xy if sc_im.get_visible() else np.zeros((0, 2)),
                ],
            )

        ax.set_title(
            f"Overlay | ACDC frame_i={frame_i}  <->  Imaris time={im_time}  | "
            f"ACDC n={len(ac_xy)} Imaris n={len(im_xy)} | tx={args.tx:.3f}, ty={args.ty:.3f}"
        )
        fig.canvas.draw_idle()

    # slider
    ax_slider = plt.axes([0.12, 0.08, 0.62, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="ACDC frame_i",
        valmin=float(frames.min()),
        valmax=float(frames.max()),
        valinit=float(start_frame_i),
        valstep=frames.astype(float).tolist(),
    )

    def on_slider(val):
        update(int(val))

    slider.on_changed(on_slider)

    def on_check(label):
        if label == "Show ACDC":
            sc_acdc.set_visible(not sc_acdc.get_visible())
        elif label == "Show Imaris":
            sc_im.set_visible(not sc_im.get_visible())

        # re-run update to rescale based on what's visible
        update(int(slider.val))

    check.on_clicked(on_check)

    # initial draw
    update(start_frame_i)

    # also print basic info
    print("Loaded:")
    print(f"  ACDC frames: {int(frames.min())} .. {int(frames.max())} (count={len(frames)})")
    print(f"  Imaris sheet: {im_sheet}")
    print(f"  Using time alignment: Imaris_Time = ACDC_frame_i + {args.imaris_time_offset}")
    print(f"  Applying translation to ACDC: tx={args.tx} um, ty={args.ty} um")
    print(f"  ACDC columns: frame={fcol}, id={idcol}, x={axcol}, y={aycol}")
    print(f"  Imaris columns: time={im_tcol}, id={im_idcol}, x={im_xcol}, y={im_ycol}")

    plt.show()


if __name__ == "__main__":
    main()
