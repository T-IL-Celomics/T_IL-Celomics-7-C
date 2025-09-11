````markdown
# Cell-ACDC `segm.npz` → per-frame TIFF (with green viz)

Small CLI to export **Cell-ACDC** segmentation files (`*_segm.npz`) as **one TIFF per frame**.  
It also writes a **display-friendly green RGB** image per frame so you can quickly inspect masks in any viewer (Fiji, Photopea, etc.).

---

## Features
- ✅ Reads `*_segm.npz` and preserves **label IDs** (0 = background, 1..N = cell IDs)
- ✅ **Axes auto-detect** (2D→`YX`, 3D→`TYX`, 4D→`TZYX`) with `--axes` override
- ✅ **Per-frame export** for time-lapse; optional per-Z split for stacks
- ✅ **Green visualization** TIFFs alongside raw labels (`--no-green` to disable)
- ✅ Works with common keys inside `.npz`: `segm`, `labels`, `label`, `masks`, `segm_img`, `arr_0`

---

## Install
```bash
pip install numpy tifffile
````

---

## Usage

```bash
python export_cellacdc_masks_per_frame.py /path/to/Position_1/Images/sample_s01_segm.npz
```

### CLI options

```
positional arguments:
  inputs                Path(s) to *_segm.npz (one or many)

optional arguments:
  -o, --outdir DIR      Output directory (default: alongside input)
  --axes AXES           YX | TYX | ZYX | TZYX
                        (defaults: 2D→YX, 3D→TYX, 4D→TZYX)
  --split-z             When Z present, write one TIFF per z-plane
  --no-green            Do not write green visualization TIFFs
```

### Examples

**Time-lapse (TYX):** one TIFF per timepoint (+ a green viz)

```bash
python export_cellacdc_masks_per_frame.py sample_s01_segm.npz
# → sample_s01_t0000.tif, sample_s01_t0000_green.tif, sample_s01_t0001.tif, ...
```

**Z-stack only (ZYX):** keep Z as a stack per file

```bash
python export_cellacdc_masks_per_frame.py sample_segm.npz --axes ZYX
# → sample_zstack.tif (+ sample_zstack_green.tif)
```

**Z-stack only (ZYX):** split every Z plane

```bash
python export_cellacdc_masks_per_frame.py sample_segm.npz --axes ZYX --split-z
# → sample_z0000.tif, sample_z0000_green.tif, ...
```

**Time + Z (TZYX):** one Z-stack per timepoint

```bash
python export_cellacdc_masks_per_frame.py sample_s01_segm.npz --axes TZYX
# → sample_s01_t0000.tif (+ sample_s01_t0000_green.tif), sample_s01_t0001.tif, ...
```

**Batch multiple positions; set output dir**

```bash
python export_cellacdc_masks_per_frame.py Pos*/Images/*_segm.npz -o exports/
```

---

## Outputs & naming

For input `sample_s01_segm.npz`:

* **Raw labels:** `sample_s01_t####.tif` (dtype `uint16` or `uint32`, multi-page if Z present)
* **Green viz:** `sample_s01_t####_green.tif` (8-bit RGB; mask in the green channel)

If you pass `--split-z`, names include `_z####`.

---

## How it works (brief)

1. Loads the `.npz` safely (`allow_pickle=False` first; falls back only if needed).
2. Pulls the first plausible key (`segm`, `labels`, …, `arr_0`) → label array.
3. Preserves label IDs by casting to `uint16` or `uint32` (based on max label).
4. Guesses axes (or uses `--axes`) and writes:

   * raw label TIFF(s), and
   * green viz TIFF(s) where `mask>0` is placed in the G channel.

---

## Provisional comparison vs Imaris (optional)

We qualitatively compared Cell-ACDC vs Imaris on a representative frame.

**Steps**

1. Exported Cell-ACDC mask with this tool (use the `_green.tif`).
2. Opened **Imaris mask (red)** and **ACDC green TIFF** in [Photopea](https://www.photopea.com/) (or any editor), stacked as layers.
3. **Overlay interpretation:**

   * **Yellow** = agreement (red ∩ green)
   * **Red-only** = Imaris extra
   * **Green-only** = ACDC extra

> Tip: ensure both masks share the same pixel grid/resolution; misalignment will appear as false disagreements.

---

## Troubleshooting

* **Looks all black:** that’s a *label* TIFF. Use the `_green.tif` files, or in Fiji run *Image → Adjust → Brightness/Contrast (Auto)* and apply a LUT (e.g., Glasbey).
* **Wrong time/Z order:** specify `--axes` explicitly (`TYX`, `ZYX`, `TZYX`).
* **Weird `.npz` keys:** script tries common keys; otherwise it falls back to the first entry.

---

## License

MIT (or update to your preferred license)

---

## Script (for reference)

<details>
<summary>export_cellacdc_masks_per_frame.py</summary>

```python
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
```

</details>
```
::contentReference[oaicite:0]{index=0}
