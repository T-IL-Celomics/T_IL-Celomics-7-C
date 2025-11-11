#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import tifffile as tiff
import csv
import sys

def load_bin_tiff(path):
    img = tiff.imread(path)
    # if RGB, take one channel
    if img.ndim == 3:
        img = img[..., 0]
    return (img > 0)

def list_tiffs(folder):
    # ignore the visualization files if they end with _green.tif
    files = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff"))
    files = [f for f in files if not f.endswith("_green.tif")]
    return sorted(files)

def compute_metrics(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0

    return {
        "intersection": int(inter),
        "union": int(union),
        "iou": float(iou),
        "dice": float(dice),
        "pred_pixels": int(pred_sum),
        "gt_pixels": int(gt_sum),
    }

def main():
    ap = argparse.ArgumentParser(
        description="Compute overlay metrics (IoU/Dice) for two folders of TIFFs: Imaris vs Cell-ACDC."
    )
    ap.add_argument("imaris_dir", help="Folder with Imaris TIFF masks")
    ap.add_argument("cellacdc_dir", help="Folder with Cell-ACDC TIFF masks")
    ap.add_argument("-o", "--out", default="overlay_results.csv", help="CSV output path")
    args = ap.parse_args()

    imaris_files = list_tiffs(args.imaris_dir)
    cellacdc_files = list_tiffs(args.cellacdc_dir)

    if len(imaris_files) == 0:
        print("No TIFFs found in imaris_dir", file=sys.stderr)
        sys.exit(1)
    if len(cellacdc_files) == 0:
        print("No TIFFs found in cellacdc_dir", file=sys.stderr)
        sys.exit(1)
    if len(imaris_files) != len(cellacdc_files):
        print(f"Warning: different number of files ({len(imaris_files)} vs {len(cellacdc_files)}). Will only compare min().")

    n = min(len(imaris_files), len(cellacdc_files))

    rows = []
    for i in range(n):
        im_path = imaris_files[i]
        ca_path = cellacdc_files[i]

        im_bin = load_bin_tiff(im_path)
        ca_bin = load_bin_tiff(ca_path)

        if im_bin.shape != ca_bin.shape:
            print(f"[{i:02d}] SHAPE MISMATCH: {im_bin.shape} vs {ca_bin.shape} -> {os.path.basename(im_path)} / {os.path.basename(ca_path)}", file=sys.stderr)
            # still write a row so CSV is not empty
            rows.append({
                "index": i,
                "imaris_file": os.path.basename(im_path),
                "cellacdc_file": os.path.basename(ca_path),
                "iou": "",
                "dice": "",
                "intersection": "",
                "union": "",
                "pred_pixels": ca_bin.sum(),
                "gt_pixels": im_bin.sum(),
            })
            continue

        metrics = compute_metrics(ca_bin, im_bin)

        rows.append({
            "index": i,
            "imaris_file": os.path.basename(im_path),
            "cellacdc_file": os.path.basename(ca_path),
            **metrics
        })

        print(
            f"[{i:02d}] IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
            f"imaris={os.path.basename(im_path)}  cellacdc={os.path.basename(ca_path)}"
        )

    # write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "imaris_file",
                "cellacdc_file",
                "iou",
                "dice",
                "intersection",
                "union",
                "pred_pixels",
                "gt_pixels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {args.out}")

if __name__ == "__main__":
    main()
