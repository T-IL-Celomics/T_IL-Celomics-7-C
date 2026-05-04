# Cell-ACDC → Imaris-like Excel Pipeline

Automated pipeline that replicates Imaris surface-detection measurements using
Cell-ACDC (open-source). Takes raw timelapse microscopy TIF files and produces
an Excel file with per-cell per-frame metrics (Area, Position X/Y, Sphericity,
Ellipsoid axes, Intensity Mean) that match Imaris output closely.

For all command-line instructions see **[COMMANDS.txt](COMMANDS.txt)**.

---

## How it works

Imaris detects cells with: `Gaussian blur → threshold → connected components`

Cell-ACDC's `thresholding` model does the same thing using scikit-image.
When `gauss_sigma` and `threshold_method` are tuned to match, the cell masks —
and therefore all downstream measurements — agree with Imaris numerically.

The Optuna optimizer (step 2b) automates this tuning by running 50 parameter
combinations and scoring each one against an Imaris ground-truth Excel file.

---

## Folder structure

```
project/
├── COMMANDS.txt                    <- all CLI commands, copy-paste ready
├── README.md
├── requirements.txt
├── scripts/
│   ├── rename_to_acdc_input.py         step 1 — rename raw exports
│   ├── build_segmentation_input.py     step 2 — build Position_N folders
│   ├── optimize_acdc_params_optuna.py  step 2b — Optuna tuning (optional)
│   ├── run_segm_track_from_index.py    step 3 — run ACDC CLI
│   ├── filter_acdc_output_by_tracklen.py step 4 — filter short tracks (optional)
│   ├── export_imaris_like_from_pipeline.py step 5 — export Excel
│   └── acdc_npz_tif_to_imaris_like.py  core converter used by step 5
├── inis/
│   └── thresholding_acdc_segm_track_workflow.ini   base template INI
├── segmentation_input/             built by step 2, read by steps 3-5
│   ├── position_map.csv
│   ├── Position_1/Images/
│   ├── Position_2/Images/
│   └── Position_3/Images/
├── optuna_trials/                  written by optimizer (step 2b)
│   └── thresholding_phase1/
│       ├── best_NIR.ini
│       └── optuna_summary.csv
├── pipeline_output/                written by step 5
│   └── *.xlsx
└── results/                        Imaris ground-truth Excel files
    └── C-1_MI006-nir_D2_1.xls
```

---

## Pipeline overview

| Step | Script | Env | Run when |
|------|--------|-----|----------|
| 0 | ACDC GUI (manual) | `acdc` | First time only — create base INI |
| 1 | `rename_to_acdc_input.py` | `imaris_xls` | New raw export from microscope |
| 2 | `build_segmentation_input.py` | `imaris_xls` | After step 1 |
| 2b | `optimize_acdc_params_optuna.py` | `imaris_xls` | First time or new channel — find best params |
| 3 | `acdc.exe` (via INI) | `acdc` | Every batch — runs segmentation + tracking |
| 4 | `filter_acdc_output_by_tracklen.py` | `imaris_xls` | Optional — remove short tracks |
| 5 | `export_imaris_like_from_pipeline.py` | `imaris_xls` | After step 3 — produce final Excel |

---

## Conda environments

Two environments are needed:

- **`acdc`** — Cell-ACDC and all its dependencies (Qt, trackpy, scikit-image).
  Used only for the ACDC GUI (step 0) and ACDC CLI segmentation (step 3).

- **`imaris_xls`** — Lightweight env with pandas, numpy, scipy, optuna, xlrd, openpyxl.
  Used for all Python scripts (steps 1, 2, 2b, 4, 5).

The optimizer script (`step 2b`) runs in `imaris_xls` and spawns ACDC as a
subprocess in the `acdc` env automatically — you do not need to switch envs manually.

---

## Parameters tuned by the optimizer

| Parameter | Range | Effect |
|-----------|-------|--------|
| `gauss_sigma` | 0.5 – 5.0 | Blur before threshold. Higher = smoother/larger cells |
| `threshold_method` | otsu / li / triangle / isodata / yen | Which auto-threshold algorithm |
| `min_area` | 10 – 500 px | Smallest kept cell. Higher removes more fragments |

---

## Target accuracy

After optimization, comparing `pipeline_output/*.xlsx` to `results/*.xls`:

- Position X / Y: R² > 0.95 (centroid matching is robust)
- Area: R² > 0.7
- Sphericity: R² > 0.6
- Ellipsoid axes: R² > 0.5

If scores are below these, run the optimizer again with `--n-trials 100` or
try a different position as the reference for optimization.
