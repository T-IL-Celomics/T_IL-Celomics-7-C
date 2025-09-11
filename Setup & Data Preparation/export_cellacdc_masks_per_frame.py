#!/usr/bin/env python3
import argparse, os
import numpy as np
import tifffile as tiff

CAND_KEYS = ("segm", "labels", "label", "masks", "segm_img", "arr_0")

def load_labels(npz_path):
    # safer first
    try:
        npz = np.load(npz_path, allow_pickle=False)
    except Exception:
        npz = np.load(npz_path, allow_pickle=True)

    key = next((k for k in CAND_KEYS if k in npz.files), npz.files[0])
    arr = npz[key]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return arr.astype(np.uint32 if arr.max() > 65535 else np.uint16, copy=False)

def guess_axes(arr, user_axes):
    if user_axes:
        return user_axes.upper()
    if arr.ndim == 2:  return "YX"
    if arr.ndim == 3:  return "TYX"   # if yours is ZYX, pass --axes ZYX
    if arr.ndim == 4:  return "TZYX"
    raise ValueError(f"Unsupported ndim={arr.ndim}")

def ensure_outdir(path): os.makedirs(path, exist_ok=True)

def to_green(mask2d):
    """Return an 8-bit RGB image where mask>0 is pure green."""
    rgb = np.zeros(mask2d.shape + (3,), dtype=np.uint8)
    rgb[..., 1] = (mask2d > 0).astype(np.uint8) * 255
    return rgb

def export_per_frame(arr, axes, outdir, base, split_z=False, green=True):
    axes = axes.upper()

    if axes == "YX":
        out = os.path.join(outdir, f"{base}_t0000.tif")
        tiff.imwrite(out, arr)  # raw labels
        if green:
            tiff.imwrite(out.replace(".tif", "_green.tif"), to_green(arr), photometric="rgb")
        print(f"Saved 1 frame to {outdir}")
        return

    if axes == "TYX":
        T = arr.shape[0]
        for t in range(T):
            out = os.path.join(outdir, f"{base}_t{t:04d}.tif")
            frame = arr[t]
            tiff.imwrite(out, frame)
            if green:
                tiff.imwrite(out.replace(".tif", "_green.tif"), to_green(frame), photometric="rgb")
        print(f"Wrote {T} frames (TYX) to {outdir}")
        return

    if axes == "ZYX":
        Z = arr.shape[0]
        if split_z:
            for z in range(Z):
                out = os.path.join(outdir, f"{base}_z{z:04d}.tif")
                plane = arr[z]
                tiff.imwrite(out, plane)
                if green:
                    tiff.imwrite(out.replace(".tif", "_green.tif"), to_green(plane), photometric="rgb")
            print(f"Wrote {Z} planes (ZYX split) to {outdir}")
        else:
            out = os.path.join(outdir, f"{base}_zstack.tif")
            tiff.imwrite(out, arr)  # raw labels as multipage
            if green:
                green_stack = np.stack([to_green(arr[z]) for z in range(Z)], axis=0)  # (Z, Y, X, 3)
                tiff.imwrite(out.replace(".tif", "_green.tif"), green_stack, photometric="rgb")
            print(f"Saved Z-stack (ZYX) to {outdir}")
        return

    if axes == "TZYX":
        T, Z = arr.shape[:2]
        if split_z:
            for t in range(T):
                for z in range(Z):
                    out = os.path.join(outdir, f"{base}_t{t:04d}_z{z:04d}.tif")
                    plane = arr[t, z]
                    tiff.imwrite(out, plane)
                    if green:
                        tiff.imwrite(out.replace(".tif", "_green.tif"), to_green(plane), photometric="rgb")
            print(f"Wrote {T*Z} planes (TZYX split) to {outdir}")
        else:
            for t in range(T):
                out = os.path.join(outdir, f"{base}_t{t:04d}.tif")
                zstack = arr[t]                                # (Z, Y, X)
                tiff.imwrite(out, zstack)                      # raw labels
                if green:
                    green_stack = np.stack([to_green(zstack[z]) for z in range(Z)], axis=0)  # (Z, Y, X, 3)
                    tiff.imwrite(out.replace(".tif", "_green.tif"), green_stack, photometric="rgb")
            print(f"Wrote {T} Z-stacks (one per time) to {outdir}")
        return

    raise ValueError(f"Unrecognized axes spec '{axes}'")

def main():
    ap = argparse.ArgumentParser(description="Export Cell-ACDC *_segm.npz to per-frame TIFF + green viz.")
    ap.add_argument("inputs", nargs="+", help="Path(s) to *_segm.npz")
    ap.add_argument("-o", "--outdir", default="", help="Output dir (default: alongside input)")
    ap.add_argument("--axes", default="", help="YX, TYX, ZYX, or TZYX (default guesses: 2D→YX, 3D→TYX, 4D→TZYX)")
    ap.add_argument("--split-z", action="store_true", help="When Z present, write one TIFF per z-plane")
    ap.add_argument("--no-green", action="store_true", help="Do not write green visualization TIFFs")
    args = ap.parse_args()

    for segm_path in args.inputs:
        segm_path = os.path.abspath(segm_path)
        arr = load_labels(segm_path)
        axes = guess_axes(arr, args.axes)
        outdir = args.outdir or os.path.dirname(segm_path)
        ensure_outdir(outdir)
        base = os.path.splitext(os.path.basename(segm_path))[0].replace("_segm", "").replace(".npz", "")

        print(f"\nInput: {segm_path}\nshape={arr.shape}  dtype={arr.dtype}  axes={axes}")
        export_per_frame(arr, axes, outdir, base, split_z=args.split_z, green=(not args.no_green))

if __name__ == "__main__":
    main()
