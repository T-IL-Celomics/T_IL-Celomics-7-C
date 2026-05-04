# Cell-ACDC → Imaris-like Excel Pipeline

Automated pipeline that replicates Imaris surface-detection measurements
using Cell-ACDC (open-source, free). Takes raw timelapse microscopy TIF files
and produces an Excel file with per-cell per-frame metrics — Area, Position X/Y,
Sphericity, Ellipsoid axes, Intensity Mean — that match Imaris output closely.

**Quickest way to run:** double-click one of the batch files below.
For full manual commands and explanations see [COMMANDS.txt](COMMANDS.txt).

---

## How it works

Imaris detects cells with: `Gaussian blur → threshold → connected components`

Cell-ACDC's `thresholding` model does exactly the same thing using scikit-image.
When `gauss_sigma` and `threshold_method` are tuned to match, the cell masks
and all downstream measurements agree with Imaris numerically.

The Optuna optimizer (`run_optimization.bat`) automates this tuning by running
50 parameter combinations and scoring each against an Imaris ground-truth file.

---

## Batch files — start here

| Batch file | What it does | When to run |
|------------|-------------|-------------|
| `run_data_prep.bat` | Renames raw microscope exports and builds `segmentation_input/` | Once per new raw data export |
| `run_optimization.bat` | Optuna parameter search — finds best `gauss_sigma`, `threshold_method`, `min_area` | Once per channel, ~40–50 min |
| `run_pipeline.bat` | ACDC segmentation + tracking on all 3 positions, then exports Excel | Every batch after INI is tuned |

**Typical order — new experiment:**
`run_data_prep` → `run_optimization` → *(copy best params to INI)* → `run_pipeline`

**Typical order — repeat batch (same channel, params already known):**
`run_data_prep` → `run_pipeline`

---

## Pipeline steps

| Step | Script | Env | Bat file | Run when |
|------|--------|-----|----------|----------|
| 0 | ACDC GUI (manual) | `acdc` | — | First time only — create base INI visually |
| 1 | `rename_to_acdc_input.py` | `imaris_xls` | `run_data_prep.bat` | New raw export |
| 2 | `build_segmentation_input.py` | `imaris_xls` | `run_data_prep.bat` | After step 1 |
| 2b | `optimize_acdc_params_optuna.py` | `imaris_xls` | `run_optimization.bat` | First time or new channel |
| 3 | `acdc.exe -p <ini> -y` | `acdc` | `run_pipeline.bat` | Every batch |
| 4 | `filter_acdc_output_by_tracklen.py` | `imaris_xls` | *(manual)* | Optional — remove short tracks |
| 5 | `export_imaris_like_from_pipeline.py` | `imaris_xls` | `run_pipeline.bat` | After step 3 |

---

## Conda environments

| Env | Used for |
|-----|---------|
| `acdc` | ACDC GUI (step 0) and ACDC CLI segmentation (step 3) |
| `imaris_xls` | All other Python scripts (steps 1, 2, 2b, 4, 5) |

The optimizer (`run_optimization.bat`) runs in `imaris_xls` and spawns ACDC as a
subprocess in `acdc` automatically — no manual env switching needed.

---

## Folder structure

```
project/
├── run_data_prep.bat           steps 1-2:  rename raw files + build folders
├── run_optimization.bat        step 2b:    Optuna param search
├── run_pipeline.bat            steps 3+5:  ACDC segmentation + export Excel
├── COMMANDS.txt                all manual CLI commands, copy-paste ready
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── rename_to_acdc_input.py               step 1
│   ├── build_segmentation_input.py           step 2
│   ├── optimize_acdc_params_optuna.py        step 2b — Optuna optimizer
│   ├── filter_acdc_output_by_tracklen.py     step 4 (optional)
│   ├── export_imaris_like_from_pipeline.py   step 5
│   ├── acdc_npz_tif_to_imaris_like.py        core converter (used by step 5)
│   └── run_segm_track_from_index.py          alternative ACDC runner (manual use)
│
├── inis/
│   └── thresholding_acdc_segm_track_workflow.ini   base INI template
│
├── segmentation_input/         built by step 2, read by steps 3-5
│   ├── position_map.csv
│   ├── _cli_runs/              per-position INI files used by acdc.exe
│   │   ├── field1_B2_1_NIR__Position_1.ini
│   │   ├── field1_D2_1_NIR__Position_2.ini
│   │   └── field1_E2_1_NIR__Position_3.ini
│   ├── Position_1/Images/      field1_B2  (B2_1)
│   ├── Position_2/Images/      field1_D2  (D2_1)  ← used for optimization
│   └── Position_3/Images/      field1_E2  (E2_1)
│
├── optuna_trials/              written by run_optimization.bat
│   └── thresholding_phase1/
│       ├── best_NIR.ini        ← copy params from here to inis/ after optimization
│       ├── optuna_summary.csv
│       └── p1_trial_NNNN/     one folder per trial
│
├── pipeline_output/            written by run_pipeline.bat (step 5)
│   └── *.xlsx
│
├── results/                    Imaris ground-truth Excel files
│   ├── C-1_MI006-nir_D2_1.xls     Position_2 ground truth (used for optimization)
│   ├── C-1_MI006-nir_E2_1.xls     Position_3 ground truth
│   └── validitons/                 validation scripts and comparison reports
│
└── tools/
    ├── environment_acdc.yml        conda env export for acdc
    ├── imaris_xls.yml              conda env export for imaris_xls
    └── patch_trackpy_cli.py        fix for trackpy CLI crash (if needed)
```

---

## Optimizer parameters

| Parameter | Search range | Effect |
|-----------|-------------|--------|
| `gauss_sigma` | 0.5 – 5.0 | Blur strength before threshold. Higher = smoother, larger cells |
| `threshold_method` | otsu / li / triangle / isodata / yen | Which auto-threshold algorithm |
| `min_area` | 10 – 500 px | Smallest cell kept. Higher removes more fragments |

After optimization: open `optuna_trials/thresholding_phase1/best_NIR.ini`, copy the
three values above into `inis/thresholding_acdc_segm_track_workflow.ini`, then run
`run_pipeline.bat`.

---

## Expected accuracy

After optimization, comparing `pipeline_output/*.xlsx` to `results/*.xls`:

| Metric | Target R² |
|--------|-----------|
| Position X / Y | > 0.95 |
| Area | > 0.70 |
| Sphericity | > 0.60 |
| Ellipsoid axis A / B | > 0.50 |

If scores are below target, re-run the optimizer with `--n-trials 100` or use a
different position as the optimization reference.
