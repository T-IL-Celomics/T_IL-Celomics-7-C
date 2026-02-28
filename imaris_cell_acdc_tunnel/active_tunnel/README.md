# Active Tunnel (Cell-ACDC → Imaris-like Excel)

This folder contains the **production pipeline** (“active tunnel”) used in the project.
It runs end-to-end from **raw microscopy exports** → **Cell-ACDC segmentation + tracking** → **Imaris-like Excel export**.

✅ Includes only pipeline scripts (no validation / no plotting / no notebooks).

---

## Folder structure

```text
active_tunnel/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ scripts/
│  ├─ 01_rename_to_acdc_input.py
│  ├─ 02_build_segmentation_input.py
│  ├─ 03_run_segm_track_from_map.py
│  ├─ 04_filter_acdc_output_by_tracklen.py          (optional)
│  ├─ 05_export_imaris_like_from_pipeline.py
│  └─ 06_acdc_npz_tif_to_imaris_like.py             (converter)
├─ inis/
│  ├─ acdc_segm_track_workflow_NIR.ini
│  ├─ acdc_segm_track_workflow_GREEN.ini
│  └─ acdc_segm_track_workflow_ORANGE.ini
└─ tools/
   ├─ patch_trackpy_cli.py                          (optional CLI fix for TrackPy)
   └─ conda_env/                                    (environment export files)
```

---

## Data layout expected by the pipeline

### After step 02 (segmentation input)
The pipeline produces:

```text
segmentation_input/
├─ position_map.csv
├─ Position_1/
│  └─ Images/
│     ├─ <signal>.tif
│     └─ <metadata>.csv
├─ Position_2/
│  └─ Images/
│     ├─ <signal>.tif
│     └─ <metadata>.csv
...
```

`position_map.csv` is the key index used by later steps to select specific experiments.

---

## Step-by-step usage (end-to-end)

### 1) Rename lab export → ACDC naming
```bash
python scripts/01_rename_to_acdc_input.py Phase3
```

Output example:
```text
ACDC_IN_Phase3/field1/B3_1/NIR/Images/field1_B3_NIR_0.tif ... field1_B3_NIR_47.tif
```

---

### 2) Build `segmentation_input/` + `position_map.csv` (GUI replacement)
```bash
python scripts/02_build_segmentation_input.py --in_root ACDC_IN_Phase3 --out_root segmentation_input
```

---

### 3) Run segmentation + tracking (CLI) using per-channel INI
Run a single selection:
```bash
python scripts/03_run_segm_track_from_map.py --exp_root segmentation_input --ini_dir inis --select field1_B3_1_NIR
```

Run all:
```bash
python scripts/03_run_segm_track_from_map.py --exp_root segmentation_input --ini_dir inis --all
```

Outputs are written by Cell-ACDC into each `Position_X/Images/`, e.g.:
- `*_segm.npz`
- `*_acdc_output.csv`
- tracked outputs (depends on workflow)

---

### 4) (Optional) Filter short tracks in `*_acdc_output.csv`
This step is optional and designed to be “plug-and-play” in the pipeline.

```bash
python scripts/04_filter_acdc_output_by_tracklen.py --exp_root segmentation_input --select field1_B3_1_NIR --min_frames 15
```

Behavior:
- creates backup: `*_acdc_output.original.csv`
- overwrites `*_acdc_output.csv` with filtered rows

---

### 5) Export Imaris-like Excel from the pipeline outputs
```bash
python scripts/05_export_imaris_like_from_pipeline.py ^
  --exp_root segmentation_input ^
  --out_dir pipeline_output ^
  --select field1_B3_1_NIR ^
  --pixel-size-um 0.108 ^
  --z-step-um 1.0 ^
  --frame-interval-s 3600
```

Output example:
```text
pipeline_output/field1_B3_1_NIR__Position_2_imaris_like.xlsx
```

IMPORTANT:
- If `*_acdc_output.csv` exists, the converter exports **ONLY** objects mapped to IDs in that CSV.
  This means if you filtered the CSV (step 04), filtered IDs will NOT appear in the Excel.

---

## Troubleshooting

### TrackPy CLI crash (progressbar signals)
If tracking crashes in CLI with errors like:
`KernelCliSignals ... initProgressBar`
use:
```bash
python tools/patch_trackpy_cli.py
```

---

## Notes
- All paths are relative; run commands from inside `active_tunnel/`.
- For Windows CMD users, use `^` for line continuation (as shown above).
