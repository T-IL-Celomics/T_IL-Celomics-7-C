"""
formula_search.py
=================
Systematic brute-force search for the correct formula for every Imaris-like metric.

For a small set of matched cell pairs (pipeline segm.npz + Imaris GT xlsx) it
tries every plausible formula variant, unit conversion and offset combination,
then reports which formula gives the best match (highest R², lowest MAE).

Usage:
  python scripts/formula_search.py \
    --segm-npz  segmentation_input/Position_2/Images/field1_D2_segm.npz \
    --tif       segmentation_input/Position_2/Images/field1_D2_NIR.tif \
    --gt        results/C-1_MI006-nir_D2_1.xls \
    --pixel-size-um 0.108 \
    --z-step-um 1.0 \
    --frame 0 \
    --n-cells 40
"""
from __future__ import annotations
import argparse, sys, warnings, io
from pathlib import Path
from itertools import product

# Force UTF-8 output on Windows to handle special characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("stdlib ok", flush=True)
import numpy as np
print("numpy ok", flush=True)
import pandas as pd
print("pandas ok", flush=True)
import tifffile
print("tifffile ok", flush=True)
from skimage.measure import regionprops, label as sk_label, marching_cubes
print("skimage ok", flush=True)

warnings.filterwarnings("ignore")

PX = 0.108      # default; overridden by --pixel-size-um
ZS = 1.0        # default; overridden by --z-step-um

# ──────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────
def load_segm(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    arr = np.array(data["arr_0"])
    data.close()
    return arr


def load_gt_sheets(path: str) -> dict[str, pd.DataFrame]:
    engine = "xlrd" if path.lower().endswith(".xls") else "openpyxl"
    xl = pd.ExcelFile(path, engine=engine)
    out = {}
    for name in xl.sheet_names:
        try:
            df = xl.parse(name)
            df.columns = [str(c).strip() for c in df.columns]
            out[name] = df
        except Exception:
            pass
    return out


# ──────────────────────────────────────────────────────────────────
# Pure-numpy helpers (no scipy)
# ──────────────────────────────────────────────────────────────────
def _greedy_match(D: np.ndarray):
    """Greedy nearest-neighbour matching on distance matrix D (rows=pipe, cols=gt).
    Each row and column used at most once. Returns (row_indices, col_indices)."""
    used_r, used_c = set(), set()
    order = np.argsort(D.ravel())
    rows, cols = [], []
    for idx in order:
        r, c = divmod(int(idx), D.shape[1])
        if r in used_r or c in used_c:
            continue
        rows.append(r); cols.append(c)
        used_r.add(r); used_c.add(c)
        if len(rows) == min(D.shape):
            break
    return np.array(rows), np.array(cols)


# ──────────────────────────────────────────────────────────────────
# Match pipeline cells → GT cells for one frame
# (Hungarian on Euclidean distance between geometric centroids)
# ──────────────────────────────────────────────────────────────────
def match_cells(props, gt_pos_df, frame_time, offset_x_um, offset_y_um,
                px_um, max_dist_um=5.0):
    """
    Returns list of (prop, gt_row) matched pairs.
    gt_pos_df must have columns Position X, Position Y, Time, Parent.
    """
    gt_frame = gt_pos_df[gt_pos_df["Time"] == frame_time].copy()
    if gt_frame.empty:
        return []

    # pipeline centroids (geometric) shifted by offset
    pipe_pts = []
    for p in props:
        cy, cx = p.centroid
        x_um = (cx + 0.5) * px_um + offset_x_um
        y_um = (cy + 0.5) * px_um + offset_y_um
        pipe_pts.append([x_um, y_um])
    pipe_pts = np.array(pipe_pts) if pipe_pts else np.empty((0, 2))

    gt_pts = gt_frame[["Position X", "Position Y"]].values.astype(float)
    if pipe_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return []

    # Pure numpy distance matrix
    D = np.sqrt(((pipe_pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=2))
    # Greedy nearest-neighbour matching (no scipy needed)
    ri, ci = _greedy_match(D)
    pairs = []
    for r, c in zip(ri, ci):
        if D[r, c] <= max_dist_um:
            pairs.append((props[r], gt_frame.iloc[c]))
    return pairs


# ──────────────────────────────────────────────────────────────────
# Scoring helper
# ──────────────────────────────────────────────────────────────────
def score(pred: np.ndarray, actual: np.ndarray):
    """Returns (r2, mae, slope, intercept) or NaN tuple on failure."""
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[mask], actual[mask]
    n = len(pred)
    if n < 3 or np.std(actual) == 0 or np.std(pred) == 0:
        return dict(r2=np.nan, mae=np.nan, n=n, slope=np.nan, intercept=np.nan)
    # Pure numpy OLS
    xm, ym = pred.mean(), actual.mean()
    ssxx = float(np.sum((pred - xm) ** 2))
    ssyy = float(np.sum((actual - ym) ** 2))
    ssxy = float(np.sum((pred - xm) * (actual - ym)))
    sl = ssxy / ssxx if ssxx > 0 else np.nan
    ic = ym - sl * xm
    r2 = (ssxy ** 2) / (ssxx * ssyy) if (ssxx * ssyy) > 0 else np.nan
    mae = float(np.mean(np.abs(pred - actual)))
    return dict(r2=r2, mae=mae, n=n, slope=sl, intercept=ic)


def best_formula(results: list[dict]) -> dict:
    valid = [r for r in results if np.isfinite(r.get("r2", np.nan))]
    if not valid:
        return {"formula": "N/A", "r2": np.nan, "mae": np.nan}
    return max(valid, key=lambda r: (r["r2"], -r.get("mae", 1e9)))


# ──────────────────────────────────────────────────────────────────
# Mesh helpers
# ──────────────────────────────────────────────────────────────────
def surface_area_2d_slab(mask2d, z_step, px):
    """Marching cubes on [empty, mask, empty] slab."""
    slab = np.zeros((3,) + mask2d.shape, dtype=np.float32)
    slab[1] = mask2d.astype(np.float32)
    if np.all(slab == 0):
        return 0.0
    try:
        verts, faces, _, _ = marching_cubes(slab, level=0.5,
                                             spacing=(z_step, px, px))
        if faces.shape[0] == 0:
            return 0.0
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())
    except Exception:
        return 0.0


def surface_area_2d_padded(mask2d, z_step, px):
    """Padded slab variant."""
    slab = np.zeros((3,) + mask2d.shape, dtype=np.float32)
    slab[1] = mask2d.astype(np.float32)
    slab = np.pad(slab, 1, mode="constant", constant_values=0)
    try:
        verts, faces, _, _ = marching_cubes(slab, level=0.5,
                                             spacing=(z_step, px, px))
        if faces.shape[0] == 0:
            return 0.0
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────
# Per-cell formula variants
# ──────────────────────────────────────────────────────────────────
def cell_features(prop, img_frame, px_um, z_step_um):
    """Compute every possible variant for a single cell/frame."""
    # build full-frame boolean mask for this label
    full_mask = np.zeros(img_frame.shape[:2], dtype=bool)
    full_mask[prop.coords[:, 0], prop.coords[:, 1]] = True

    coords_y = prop.coords[:, 0].astype(float)
    coords_x = prop.coords[:, 1].astype(float)
    n_px = int(prop.area)
    intens = img_frame[prop.coords[:, 0], prop.coords[:, 1]].astype(float)
    wsum = float(intens.sum())

    # --- geometric centroid (pixel index) ---
    cy_idx = float(np.mean(coords_y))
    cx_idx = float(np.mean(coords_x))

    # --- intensity-weighted centroid (pixel index) ---
    if wsum > 0:
        cim_y_idx = float(np.sum(coords_y * intens) / wsum)
        cim_x_idx = float(np.sum(coords_x * intens) / wsum)
    else:
        cim_y_idx, cim_x_idx = cy_idx, cx_idx

    # --- covariance eigenvalues (unbiased and biased) ---
    # Use analytical 2x2 eigendecomposition (z=0, so 3rd eigenvalue is always 0)
    # This avoids LAPACK calls that cause STATUS_STACK_BUFFER_OVERRUN on this system
    x_um = (coords_x + 0.5) * px_um
    y_um = (coords_y + 0.5) * px_um

    def cov2_eig(x, y, bias):
        """Analytical 2x2 covariance eigenvalues. Returns [la, lb, 0] sorted desc."""
        n = len(x)
        if n < 2:
            return np.array([0., 0., 0.])
        ddof = 0 if bias else 1
        xm, ym = x.mean(), y.mean()
        dx, dy = x - xm, y - ym
        var_x = float(np.sum(dx*dx)) / (n - ddof)
        var_y = float(np.sum(dy*dy)) / (n - ddof)
        cov_xy = float(np.sum(dx*dy)) / (n - ddof)
        # Eigenvalues of [[var_x, cov_xy],[cov_xy, var_y]]
        trace = var_x + var_y
        det   = var_x * var_y - cov_xy * cov_xy
        disc  = max(0.0, (trace/2)**2 - det)
        la = trace/2 + np.sqrt(disc)
        lb = trace/2 - np.sqrt(disc)
        return np.array([max(0.0, la), max(0.0, lb), 0.0])

    eig_unbias = cov2_eig(x_um, y_um, bias=False)
    eig_bias   = cov2_eig(x_um, y_um, bias=True)

    # regionprops major/minor axis (in pixels)
    # Compute analytically from pixel coords to avoid LAPACK (which crashes on this system)
    # skimage formula: axis_length = 4*sqrt(eigenvalue of inertia_tensor in pixel coords)
    dx_px = coords_x - coords_x.mean()
    dy_px = coords_y - coords_y.mean()
    n_all = len(dx_px)
    # Inertia tensor (pixel space, biased/mean-normalized like skimage)
    a_px = float(np.sum(dy_px*dy_px)) / n_all   # var_y in px²
    b_px = -float(np.sum(dy_px*dx_px)) / n_all  # -cov_xy
    c_px = float(np.sum(dx_px*dx_px)) / n_all   # var_x
    trace_px = a_px + c_px
    det_px   = a_px * c_px - b_px * b_px
    disc_px  = max(0.0, (trace_px/2)**2 - det_px)
    l1_px = trace_px/2 + np.sqrt(disc_px)
    l2_px = max(0.0, trace_px/2 - np.sqrt(disc_px))
    maj_px = 4.0 * np.sqrt(max(0.0, l1_px))
    min_px = 4.0 * np.sqrt(max(0.0, l2_px))

    # --- surface area variants ---
    sa_slab      = surface_area_2d_slab(prop.image, z_step_um, px_um)
    sa_padded    = surface_area_2d_padded(prop.image, z_step_um, px_um)

    # perimeter-based
    perim_px = float(prop.perimeter) if hasattr(prop, "perimeter") else 0.0

    # --- volume ---
    vol_px3 = n_px * px_um * px_um * z_step_um   # slab volume

    return dict(
        n_px=n_px,
        intens=intens,
        wsum=wsum,
        cy_idx=cy_idx, cx_idx=cx_idx,
        cim_y_idx=cim_y_idx, cim_x_idx=cim_x_idx,
        eig_unbias=eig_unbias, eig_bias=eig_bias,
        maj_px=maj_px, min_px=min_px,
        sa_slab=sa_slab, sa_padded=sa_padded,
        perim_px=perim_px,
        vol_px3=vol_px3,
        bbox_image=prop.image,
        full_mask=full_mask,
        img_frame=img_frame,
        prop=prop,
    )


# ──────────────────────────────────────────────────────────────────
# Per-feature search functions
# ──────────────────────────────────────────────────────────────────
def search_area(cells, gt_vals, px_um, z_step_um):
    results = []
    for name, fn in [
        ("n_px * px^2",          lambda c: c["n_px"] * px_um**2),
        ("n_px * px^2 * z",      lambda c: c["n_px"] * px_um**2 * z_step_um),
        ("sa_slab (marching)",   lambda c: c["sa_slab"]),
        ("sa_padded (marching)", lambda c: c["sa_padded"]),
        ("n_px * px^2 / 2",      lambda c: c["n_px"] * px_um**2 / 2),
        ("2 * n_px * px^2",      lambda c: 2.0 * c["n_px"] * px_um**2),
        ("4 * n_px * px^2",      lambda c: 4.0 * c["n_px"] * px_um**2),
        ("pi * (n_px*px^2/pi)",  lambda c: np.pi * c["n_px"] * px_um**2),  # circ approx
    ]:
        pred = np.array([fn(c) for c in cells])
        s = score(pred, gt_vals)
        s["formula"] = name
        results.append(s)
    return results


def search_n_voxels(cells, gt_vals):
    results = []
    for name, fn in [
        ("n_px",            lambda c: float(c["n_px"])),
        ("regionprops.area",lambda c: float(c["prop"].area)),
    ]:
        pred = np.array([fn(c) for c in cells])
        s = score(pred, gt_vals)
        s["formula"] = name
        results.append(s)
    return results


def search_volume(cells, gt_vals, px_um, z_step_um):
    results = []
    variants = [
        ("n_px*px²*z",     lambda c: c["n_px"] * px_um**2 * z_step_um),
        ("n_px*px³",       lambda c: c["n_px"] * px_um**3),
        ("n_px*px²",       lambda c: c["n_px"] * px_um**2),
        ("n_px*px²*z/1000", lambda c: c["n_px"] * px_um**2 * z_step_um / 1000),
    ]
    for name, fn in variants:
        pred = np.array([fn(c) for c in cells])
        s = score(pred, gt_vals)
        s["formula"] = name
        results.append(s)
    return results


def search_surface_area(cells, gt_vals):
    results = []
    for name, fn in [
        ("slab_marching_cubes",  lambda c: c["sa_slab"]),
        ("padded_marching_cubes",lambda c: c["sa_padded"]),
    ]:
        pred = np.array([fn(c) for c in cells])
        s = score(pred, gt_vals)
        s["formula"] = name
        results.append(s)
    return results


def search_sphericity(cells, gt_vals, px_um, z_step_um):
    results = []
    def wadell(c):
        v = c["vol_px3"]
        a = c["sa_slab"]
        if a <= 0 or v <= 0: return np.nan
        return (np.pi**(1/3) * (6*v)**(2/3)) / a
    def wadell_padded(c):
        v = c["vol_px3"]
        a = c["sa_padded"]
        if a <= 0 or v <= 0: return np.nan
        return (np.pi**(1/3) * (6*v)**(2/3)) / a
    def circularity(c):
        p = c["perim_px"]
        a = c["n_px"]
        if p <= 0: return np.nan
        return 4 * np.pi * a / (p**2)
    def circularity_um(c):
        p = c["perim_px"] * px_um
        a = c["n_px"] * px_um**2
        if p <= 0: return np.nan
        return 4 * np.pi * a / (p**2)
    for name, fn in [
        ("Wadell(slab)", wadell),
        ("Wadell(padded)", wadell_padded),
        ("Circularity_px", circularity),
        ("Circularity_um", circularity_um),
    ]:
        pred = np.array([fn(c) for c in cells])
        s = score(pred, gt_vals)
        s["formula"] = name
        results.append(s)
    return results


def search_ellipsoid_len(cells, gt_vals, axis_idx, px_um):
    """axis_idx: 0=A(longest), 1=B, 2=C(shortest)"""
    results = []
    variants = []
    # eigenvalue-based
    for bias_name, eig_key in [("unbias", "eig_unbias"), ("bias", "eig_bias")]:
        for mult_name, mult_fn in [
            ("2√λ",   lambda v: 2.0*np.sqrt(v)),
            ("√(5λ)", lambda v: np.sqrt(5*v)),
            ("√λ",    lambda v: np.sqrt(v)),
            ("2λ",    lambda v: 2*v),
            ("λ",     lambda v: v),
        ]:
            name = f"{mult_name}_{bias_name}_ax{axis_idx}"
            key = eig_key
            ai = axis_idx
            mfn = mult_fn
            variants.append((name, key, ai, mfn))
    for name, key, ai, mfn in variants:
        try:
            pred = np.array([mfn(c[key][ai]) for c in cells])
            s = score(pred, gt_vals)
            s["formula"] = name
            results.append(s)
        except Exception:
            pass
    # regionprops major/minor axis
    if axis_idx == 0:
        pred = np.array([c["maj_px"] * px_um for c in cells])
        s = score(pred, gt_vals); s["formula"] = "regionprops.major*px"; results.append(s)
        pred = np.array([c["maj_px"] * px_um / 2 for c in cells])
        s = score(pred, gt_vals); s["formula"] = "regionprops.major*px/2"; results.append(s)
    elif axis_idx == 1:
        pred = np.array([c["min_px"] * px_um for c in cells])
        s = score(pred, gt_vals); s["formula"] = "regionprops.minor*px"; results.append(s)
        pred = np.array([c["min_px"] * px_um / 2 for c in cells])
        s = score(pred, gt_vals); s["formula"] = "regionprops.minor*px/2"; results.append(s)
    return results


def search_ellipticity(cells, gt_vals, which="oblate"):
    results = []
    def get_lens(c, eig_key):
        lens = sorted([float(c[eig_key][i]) for i in range(3)], reverse=True)
        return lens[0], lens[1], lens[2]  # LA >= LB >= LC

    for eig_key in ["eig_unbias", "eig_bias"]:
        # standard Imaris formulas
        def oblate(c, k=eig_key):
            la, lb, lc = get_lens(c, k)
            if lb <= 0: return np.nan
            return 1.0 - lc/lb
        def prolate(c, k=eig_key):
            la, lb, lc = get_lens(c, k)
            if la <= 0: return np.nan
            return 1.0 - lb/la
        # alternative: using raw axis lengths (not sorted)
        def oblate_raw(c, k=eig_key):
            la, lb, lc = float(c[k][0]), float(c[k][1]), float(c[k][2])
            if lb <= 0: return np.nan
            return 1.0 - lc/lb
        def prolate_raw(c, k=eig_key):
            la, lb, lc = float(c[k][0]), float(c[k][1]), float(c[k][2])
            if la <= 0: return np.nan
            return 1.0 - lb/la

        if which == "oblate":
            for name, fn in [(f"1-LC/LB_{eig_key}", oblate),
                             (f"1-LC/LB_raw_{eig_key}", oblate_raw)]:
                pred = np.array([fn(c) for c in cells])
                s = score(pred, gt_vals); s["formula"] = name; results.append(s)
        else:
            for name, fn in [(f"1-LB/LA_{eig_key}", prolate),
                             (f"1-LB/LA_raw_{eig_key}", prolate_raw)]:
                pred = np.array([fn(c) for c in cells])
                s = score(pred, gt_vals); s["formula"] = name; results.append(s)
    return results


def search_position(cells, gt_pos_x, gt_pos_y, px_um,
                    offsets_x, offsets_y):
    results_x, results_y = [], []
    for ox in offsets_x:
        for oy in offsets_y:
            for name_x, fn_x in [
                ("geom_centroid", lambda c, o=ox: (c["cx_idx"] + 0.5)*px_um + o),
                ("geom_idx_only", lambda c, o=ox: c["cx_idx"]*px_um + o),
                ("CIM",           lambda c, o=ox: (c["cim_x_idx"]+0.5)*px_um + o),
                ("CIM_idx",       lambda c, o=ox: c["cim_x_idx"]*px_um + o),
            ]:
                for name_y, fn_y in [
                    ("geom_centroid", lambda c, o=oy: (c["cy_idx"] + 0.5)*px_um + o),
                    ("CIM",           lambda c, o=oy: (c["cim_y_idx"]+0.5)*px_um + o),
                ]:
                    px_pred = np.array([fn_x(c) for c in cells])
                    py_pred = np.array([fn_y(c) for c in cells])
                    sx = score(px_pred, gt_pos_x)
                    sx["formula"] = f"X:{name_x}  ox={ox:+.2f}"
                    sy = score(py_pred, gt_pos_y)
                    sy["formula"] = f"Y:{name_y}  oy={oy:+.2f}"
                    results_x.append(sx)
                    results_y.append(sy)
    return results_x, results_y


def search_intensity(cells, gt_vals, metric):
    results = []
    def get_stat(c, fn):
        i = c["intens"]
        if i.size == 0: return np.nan
        return fn(i)
    for name, fn in [
        ("mean",   lambda i: float(np.mean(i))),
        ("median", lambda i: float(np.median(i))),
        ("max",    lambda i: float(np.max(i))),
        ("min",    lambda i: float(np.min(i))),
        ("sum",    lambda i: float(np.sum(i))),
        ("std",    lambda i: float(np.std(i, ddof=0))),
        ("sum/n",  lambda i: float(np.sum(i)/len(i))),
    ]:
        pred = np.array([get_stat(c, fn) for c in cells])
        s = score(pred, gt_vals); s["formula"] = name; results.append(s)

    # intensity at rounded CIM
    def intensity_center(c):
        if c["wsum"] <= 0: return np.nan
        im = c["img_frame"]
        ry = int(np.clip(round(c["cim_y_idx"]), 0, im.shape[0]-1))
        rx = int(np.clip(round(c["cim_x_idx"]), 0, im.shape[1]-1))
        return float(im[ry, rx])
    pred = np.array([intensity_center(c) for c in cells])
    s = score(pred, gt_vals); s["formula"] = "pixel_at_CIM"; results.append(s)

    # intensity at geometric centroid
    def intensity_geom(c):
        im = c["img_frame"]
        ry = int(np.clip(round(c["cy_idx"]), 0, im.shape[0]-1))
        rx = int(np.clip(round(c["cx_idx"]), 0, im.shape[1]-1))
        return float(im[ry, rx])
    pred = np.array([intensity_geom(c) for c in cells])
    s = score(pred, gt_vals); s["formula"] = "pixel_at_geom_centroid"; results.append(s)
    return results


# ──────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────
def print_results(feature_name, results, top_n=5):
    valid = [r for r in results if np.isfinite(r.get("r2", np.nan))]
    valid.sort(key=lambda r: (-r["r2"], r.get("mae", 1e9)))
    print(f"\n{'='*70}")
    print(f"  {feature_name}   (n={valid[0]['n'] if valid else '?'})")
    print(f"{'='*70}")
    if not valid:
        print("  No valid results.")
        return valid[0] if valid else None
    for i, r in enumerate(valid[:top_n]):
        flag = " <-- BEST" if i == 0 else ""
        print(f"  [{i+1}] R²={r['r2']:.4f}  MAE={r['mae']:.4f}"
              f"  slope={r.get('slope',np.nan):.4f}"
              f"  intercept={r.get('intercept',np.nan):.4f}"
              f"  {r['formula']}{flag}")
    return valid[0]


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segm-npz",       required=True)
    ap.add_argument("--tif",            required=True)
    ap.add_argument("--gt",             required=True)
    ap.add_argument("--pixel-size-um",  type=float, default=0.108)
    ap.add_argument("--z-step-um",      type=float, default=1.0)
    ap.add_argument("--frame",          type=int,   default=0,
                    help="Which frame index (0-based) to use for matching")
    ap.add_argument("--n-cells",        type=int,   default=40,
                    help="Max matched pairs to use")
    ap.add_argument("--time-offset",    type=int,   default=1,
                    help="GT Time = frame_index + time_offset (try 1 or 2)")
    ap.add_argument("--offset-x-um",   type=float, default=None,
                    help="Fixed offset X in um (skips grid search if set)")
    ap.add_argument("--offset-y-um",   type=float, default=None,
                    help="Fixed offset Y in um (skips grid search if set)")
    ap.add_argument("--max-dist-um",   type=float, default=3.0)
    args = ap.parse_args()
    print(f"args parsed: px={args.pixel_size_um}", flush=True)

    px = args.pixel_size_um
    zs = args.z_step_um

    print(f"Loading data...", flush=True)
    segm_all = load_segm(args.segm_npz)
    img_all  = tifffile.imread(args.tif)
    gt       = load_gt_sheets(args.gt)
    print(f"  segm={segm_all.shape}  img={img_all.shape}  GT_sheets={len(gt)}", flush=True)

    fi = args.frame
    frame_time = fi + args.time_offset   # Imaris Time index for this frame

    segm_frame = segm_all[fi]
    img_frame  = img_all[fi] if img_all.ndim >= 3 else img_all

    props = regionprops(segm_frame.astype(np.int32), intensity_image=img_frame)
    print(f"Frame {fi} (GT Time={frame_time}):  pipeline={len(props)} cells", flush=True)

    # GT positions for this frame
    gt_pos = gt.get("Position")
    if gt_pos is None:
        sys.exit("ERROR: 'Position' sheet not found in GT.")
    gt_pos_frame = gt_pos[gt_pos["Time"] == frame_time]
    print(f"  GT cells for Time={frame_time}: {len(gt_pos_frame)}", flush=True)

    # ── determine offsets ─────────────────────────────────────────
    if args.offset_x_um is not None and args.offset_y_um is not None:
        best_ox = args.offset_x_um
        best_oy = args.offset_y_um
        print(f"  Offsets: ox={best_ox:+.3f} um  oy={best_oy:+.3f} um", flush=True)
    else:
        # grid search ±15 px around 0
        print("\n  Grid-searching best offset...")
        best_ox, best_oy, best_n = 0.0, 0.0, 0
        for dx in np.arange(-15, 16, 0.5):
            for dy in np.arange(-15, 16, 0.5):
                ox_um = dx * px
                oy_um = dy * px
                pairs = match_cells(props, gt_pos, frame_time,
                                    ox_um, oy_um, px, args.max_dist_um)
                if len(pairs) > best_n:
                    best_n = len(pairs)
                    best_ox, best_oy = ox_um, oy_um
        print(f"  Best offset: ox={best_ox:+.4f} um  oy={best_oy:+.4f} um"
              f"  ({best_ox/px:+.2f} px, {best_oy/px:+.2f} px)"
              f"  matched={best_n}")

    # ── final matching ────────────────────────────────────────────
    pairs = match_cells(props, gt_pos, frame_time,
                        best_ox, best_oy, px, args.max_dist_um)
    pairs = pairs[:args.n_cells]
    print(f"Matched {len(pairs)} cell pairs (max_dist={args.max_dist_um} um)", flush=True)
    if len(pairs) < 5:
        sys.exit("Too few pairs. Try increasing --max-dist-um or adjusting --time-offset.")

    # ── precompute cell features ──────────────────────────────────
    print("Computing cell features...", flush=True)
    cells = [cell_features(p, img_frame, px, zs) for p, _ in pairs]
    print(f"  done: {len(cells)} cells", flush=True)

    # helper to extract GT column for matched pairs
    def gt_col(sheet_name, col="Value"):
        sh = gt.get(sheet_name)
        if sh is None:
            return None
        sh_frame = sh[sh["Time"] == frame_time]
        vals = []
        for _, gt_row in pairs:
            parent = int(gt_row["Parent"])
            row = sh_frame[sh_frame["Parent"] == parent]
            if not row.empty:
                vals.append(float(row[col].iloc[0]))
            else:
                vals.append(np.nan)
        return np.array(vals, dtype=float)

    # ── AREA ──────────────────────────────────────────────────────
    gt_area = gt_col("Area")
    if gt_area is not None:
        r = print_results("AREA (um²)",
                          search_area(cells, gt_area, px, zs))

    # ── NUMBER OF VOXELS ─────────────────────────────────────────
    gt_nvox = gt_col("Number of Voxels")
    if gt_nvox is not None:
        r = print_results("NUMBER OF VOXELS",
                          search_n_voxels(cells, gt_nvox))

    # ── VOLUME ───────────────────────────────────────────────────
    gt_vol = gt_col("Volume")
    if gt_vol is not None:
        r = print_results("VOLUME (um³)",
                          search_volume(cells, gt_vol, px, zs))

    # ── SURFACE AREA (via Sphericity backcalc if direct not avail)
    # Imaris doesn't have a direct SurfaceArea sheet — infer from Sphericity
    gt_sph = gt_col("Sphericity")
    if gt_sph is not None:
        r = print_results("SPHERICITY",
                          search_sphericity(cells, gt_sph, px, zs))

    # ── ELLIPSOID AXIS LENGTHS ─────────────────────────────────
    # Imaris axis convention for flat 2D cells (single z-slice):
    #   Axis A = z half-extent = 0.5 * z_step (constant, ~0.61 um for z=1.22)
    #   Axis B = short in-plane principal axis (our eigenvalue index 1)
    #   Axis C = long  in-plane principal axis (our eigenvalue index 0)
    gt_axA = gt_col("Ellipsoid Axis Length A")
    if gt_axA is not None:
        print(f"\n{'='*70}")
        print(f"  Ellipsoid Axis Length A  (GT constant={np.nanmean(gt_axA):.4f} um)")
        print(f"  Formula: 0.5 * z_step_um  ({0.5*zs:.4f} um) -- constant for 2D")
        print(f"{'='*70}")

    for axis_idx, sheet_name in [(1, "Ellipsoid Axis Length B"),   # short in-plane
                                  (0, "Ellipsoid Axis Length C")]:  # long  in-plane
        gt_ax = gt_col(sheet_name)
        if gt_ax is not None:
            r = print_results(sheet_name,
                              search_ellipsoid_len(cells, gt_ax, axis_idx, px))

    # ── ELLIPTICITY ───────────────────────────────────────────────
    # For 2D flat cells the meaningful ratio is between in-plane axes.
    gt_obl = gt_col("Ellipticity (oblate)")
    if gt_obl is not None:
        extra_obl = []
        for eig_key in ["eig_unbias", "eig_bias"]:
            for name, fn in [
                (f"1-sqrt(lb/la)_{eig_key}",
                 lambda c, k=eig_key: 1.0 - np.sqrt(c[k][1]/c[k][0]) if c[k][0]>0 else np.nan),
                (f"1-lb/la_{eig_key}",
                 lambda c, k=eig_key: 1.0 - c[k][1]/c[k][0] if c[k][0]>0 else np.nan),
                (f"1-(lb/la)^2_{eig_key}",
                 lambda c, k=eig_key: 1.0 - (c[k][1]/c[k][0])**2 if c[k][0]>0 else np.nan),
            ]:
                pred = np.array([fn(c) for c in cells])
                s = score(pred, gt_obl); s["formula"] = name; extra_obl.append(s)
        all_obl = search_ellipticity(cells, gt_obl, "oblate") + extra_obl
        r = print_results("ELLIPTICITY OBLATE", all_obl)

    gt_pro = gt_col("Ellipticity (prolate)")
    if gt_pro is not None:
        extra_pro = []
        z_half = 0.5 * zs
        for eig_key in ["eig_unbias", "eig_bias"]:
            for name, fn in [
                (f"1-sqrt(lb/la)_{eig_key}",
                 lambda c, k=eig_key: 1.0 - np.sqrt(c[k][1]/c[k][0]) if c[k][0]>0 else np.nan),
                (f"z_half/sqrt(la)_{eig_key}",
                 lambda c, k=eig_key, z=z_half: z / np.sqrt(c[k][0]) if c[k][0]>0 else np.nan),
                (f"(z_half/sqrt(la))^2_{eig_key}",
                 lambda c, k=eig_key, z=z_half: (z/np.sqrt(c[k][0]))**2 if c[k][0]>0 else np.nan),
            ]:
                pred = np.array([fn(c) for c in cells])
                s = score(pred, gt_pro); s["formula"] = name; extra_pro.append(s)
        all_pro = search_ellipticity(cells, gt_pro, "prolate") + extra_pro
        r = print_results("ELLIPTICITY PROLATE", all_pro)

    # ── POSITION ─────────────────────────────────────────────────
    gt_px_arr = np.array([float(gt_row["Position X"]) for _, gt_row in pairs])
    gt_py_arr = np.array([float(gt_row["Position Y"]) for _, gt_row in pairs])

    # search around known offset ±2 px at 0.25 px resolution
    ox_candidates = np.arange(best_ox - 2*px, best_ox + 2*px + 0.001, 0.25*px)
    oy_candidates = np.arange(best_oy - 2*px, best_oy + 2*px + 0.001, 0.25*px)
    rx_list, ry_list = search_position(cells, gt_px_arr, gt_py_arr,
                                       px, ox_candidates, oy_candidates)
    print_results("POSITION X (um)", rx_list)
    print_results("POSITION Y (um)", ry_list)

    # ── INTENSITY METRICS ─────────────────────────────────────────
    for metric, sheet_name in [
        ("mean",   "Intensity Mean Ch=1"),
        ("max",    "Intensity Max Ch=1"),
        ("median", "Intensity Median Ch=1"),
        ("min",    "Intensity Min Ch=1"),
        ("sum",    "Intensity Sum Ch=1"),
        ("std",    "Intensity StdDev Ch=1"),
        ("center", "Intensity Center Ch=1"),
    ]:
        gt_i = gt_col(sheet_name)
        if gt_i is not None:
            r = print_results(sheet_name,
                              search_intensity(cells, gt_i, metric))

    print("\n" + "="*70)
    print("  SEARCH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    print("entering main", flush=True)
    try:
        main()
    except Exception as e:
        import traceback
        print(f"CRASH in main: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
