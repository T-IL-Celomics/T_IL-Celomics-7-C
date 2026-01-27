import os, glob
import numpy as np
import tifffile as tiff
import optuna

from cellpose import models
from skimage.measure import label, regionprops
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift

# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = "dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
MSK_DIR = os.path.join(DATASET_DIR, "masks")

MODEL_TYPE = "cyto"   # "cyto" or "nuclei"
USE_GPU = True        # set False if no GPU
N_TRIALS = 20

CALIBRATION_FRAMES = 8
MAX_SHIFT_PX = 40

# -------------------------
# Helpers
# -------------------------
def base_name(p): return os.path.splitext(os.path.basename(p))[0]

def load_gray(path):
    img = tiff.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)

def load_mask_bin(path):
    m = tiff.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0)

def dice(pred, gt, eps=1e-8):
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter) / (pred.sum() + gt.sum() + eps)

def iou(pred, gt, eps=1e-8):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + eps)

def clamp_shift(shift_yx, max_shift=40):
    dy, dx = shift_yx
    dy = float(np.clip(dy, -max_shift, max_shift))
    dx = float(np.clip(dx, -max_shift, max_shift))
    return dy, dx

def apply_shift_bool(mask_bool, dy, dx):
    shifted = ndi_shift(mask_bool.astype(np.uint8), shift=(dy, dx), order=0, mode="constant", cval=0)
    return shifted.astype(bool)

def transforms(mask_bool):
    m = mask_bool
    yield "id", m
    yield "flipud", np.flipud(m)
    yield "fliplr", np.fliplr(m)
    yield "rot90", np.rot90(m, 1)
    yield "rot180", np.rot90(m, 2)
    yield "rot270", np.rot90(m, 3)
    yield "rot90_fliplr", np.fliplr(np.rot90(m, 1))
    yield "rot90_flipud", np.flipud(np.rot90(m, 1))

def estimate_shift(ref_bool, mov_bool):
    ref = ref_bool.astype(np.float32)
    mov = mov_bool.astype(np.float32)
    shift_yx, error, phasediff = phase_cross_correlation(ref, mov, upsample_factor=10)
    return shift_yx

# Cellpose API compat (v3/v4)
def make_cellpose_model(gpu: bool, model_type: str):
    if hasattr(models, "Cellpose"):
        return models.Cellpose(gpu=gpu, model_type=model_type)
    return models.CellposeModel(gpu=gpu, model_type=model_type)

def cellpose_eval(model, img, **kwargs):
    out = model.eval(img, **kwargs)
    if isinstance(out, tuple):
        # usually (masks, flows, styles, diams)
        masks = out[0]
        return masks
    return out

# -------------------------
# Load dataset
# -------------------------
img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.tif"))) + sorted(glob.glob(os.path.join(IMG_DIR, "*.tiff")))
if len(img_paths) == 0:
    raise FileNotFoundError(f"No TIFF images in {IMG_DIR}")

mask_paths = []
for ip in img_paths:
    bn = base_name(ip)
    mp = os.path.join(MSK_DIR, bn + ".tif")
    if not os.path.exists(mp):
        mp = os.path.join(MSK_DIR, bn + ".tiff")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"Missing mask for {bn}")
    mask_paths.append(mp)

print("Images:", len(img_paths))

# -------------------------
# Calibrate offset/orientation using baseline Cellpose
# -------------------------
cp_base = make_cellpose_model(gpu=USE_GPU, model_type=MODEL_TYPE)

def baseline_predict_union(img):
    masks = cellpose_eval(
        cp_base, img,
        channels=[0, 0],
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=50
    )
    return (masks > 0)

def calibrate_alignment(n_frames=8):
    shifts, names = [], []
    for ip, mp in list(zip(img_paths, mask_paths))[:n_frames]:
        img = load_gray(ip)
        pred = baseline_predict_union(img)
        gt0 = load_mask_bin(mp)

        best = None
        for name, gt_var in transforms(gt0):
            if gt_var.shape != pred.shape:
                continue
            shift_yx = estimate_shift(pred, gt_var)
            dy, dx = clamp_shift(shift_yx, MAX_SHIFT_PX)
            gt_aligned = apply_shift_bool(gt_var, dy, dx)
            d = dice(pred, gt_aligned)
            if best is None or d > best["d"]:
                best = {"name": name, "dy": dy, "dx": dx, "d": d}

        if best is None:
            raise RuntimeError(f"Calibration failed for {os.path.basename(ip)} (shape mismatch?)")

        shifts.append((best["dy"], best["dx"]))
        names.append(best["name"])
        print(f"{bn}: best={best['name']} dy={best['dy']:.2f} dx={best['dx']:.2f} dice={best['d']:.4f}")

    # majority vote for transform + median shift
    from collections import Counter
    tname = Counter(names).most_common(1)[0][0]
    chosen = [s for s,nm in zip(shifts, names) if nm == tname] or shifts
    dy_med = float(np.median([s[0] for s in chosen]))
    dx_med = float(np.median([s[1] for s in chosen]))
    return tname, dy_med, dx_med

ALIGN_NAME, ALIGN_DY, ALIGN_DX = calibrate_alignment(CALIBRATION_FRAMES)
print("\nALIGN:", ALIGN_NAME, "dy/dx:", ALIGN_DY, ALIGN_DX)

def apply_alignment(gt_bool):
    for name, gt_var in transforms(gt_bool):
        if name == ALIGN_NAME:
            return apply_shift_bool(gt_var, ALIGN_DY, ALIGN_DX)
    return apply_shift_bool(gt_bool, ALIGN_DY, ALIGN_DX)

# -------------------------
# Estimate diameter range from masks
# -------------------------
def estimate_diameter(max_files=10):
    diams = []
    for p in mask_paths[:max_files]:
        m = apply_alignment(load_mask_bin(p))
        lab = label(m)
        props = regionprops(lab)
        areas = [r.area for r in props if r.area > 20]
        diams.extend([2.0 * np.sqrt(a / np.pi) for a in areas])
    return float(np.median(diams)) if diams else None

d0 = estimate_diameter(min(10, len(mask_paths)))
if d0 is None:
    DIAM_LOW, DIAM_HIGH = 15.0, 60.0
else:
    DIAM_LOW, DIAM_HIGH = max(5.0, 0.7*d0), 1.3*d0

print("Diameter range:", DIAM_LOW, "to", DIAM_HIGH)

# -------------------------
# Optimize
# -------------------------
cp = make_cellpose_model(gpu=USE_GPU, model_type=MODEL_TYPE)

def objective(trial):
    diameter = trial.suggest_float("diameter", DIAM_LOW, DIAM_HIGH)
    flow_th  = trial.suggest_float("flow_threshold", 0.2, 1.0)
    cellprob = trial.suggest_float("cellprob_threshold", -2.0, 2.0)
    min_size = trial.suggest_int("min_size", 50, 2000)

    ds, js = [], []
    for ip, mp in zip(img_paths, mask_paths):
        img = load_gray(ip)
        gt  = apply_alignment(load_mask_bin(mp))

        masks = cellpose_eval(
            cp, img,
            channels=[0, 0],
            diameter=diameter,
            flow_threshold=flow_th,
            cellprob_threshold=cellprob,
            min_size=min_size
        )
        pred = (masks > 0)

        ds.append(dice(pred, gt))
        js.append(iou(pred, gt))

    trial.set_user_attr("mean_iou", float(np.mean(js)))
    return float(np.mean(ds))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

print("\nBEST mean Dice:", study.best_value)
print("BEST params:", study.best_params)
print("BEST mean IoU:", study.best_trial.user_attrs.get("mean_iou"))
print("\nALIGNMENT used:", ALIGN_NAME, (ALIGN_DY, ALIGN_DX))
