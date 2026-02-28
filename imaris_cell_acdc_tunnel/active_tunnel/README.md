# Active Tunnel (Cell-ACDC → Imaris-like Excel)

This folder contains the **production pipeline** ("active tunnel") used in the project.  
It runs end-to-end from **raw microscopy exports** → **Cell-ACDC segmentation** → **TrackPy gap-closing** → **Imaris-like Excel export**.

✅ Includes only pipeline scripts (no validation / no plotting / no notebooks).

---

## Folder structure

```text
active_tunnel/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ run_tunnel_interactive_conda_path.bat    ← Windows interactive runner (all 6 steps)
├─ scripts/
│  ├─ rename_to_acdc_input.py              ← step 1: rename raw lab export
│  ├─ build_segmentation_input.py          ← step 2: build segmentation_input/ structure
│  ├─ run_segm_track_from_index.py         ← step 3: Cell-ACDC segmentation + tracking
│  ├─ run_trackpy_linking.py               ← step 4: TrackPy gap-closing (persistent IDs)
│  ├─ filter_acdc_output_by_tracklen.py    ← step 5: filter short tracks (optional)
│  ├─ export_imaris_like_from_pipeline.py  ← step 6: batch Imaris-like Excel export
│  └─ acdc_npz_tif_to_imaris_like.py       ← core converter (called by step 6)
├─ inis/
│  ├─ acdc_segm_track_workflow_NIR.ini
│  ├─ acdc_segm_track_workflow_GREEN.ini
│  └─ acdc_segm_track_workflow_ORANGE.ini
└─ tools/
   ├─ patch_trackpy_cli.py                 ← optional CLI fix for TrackPy progressbar crash
   └─ conda_env/                           ← environment export files
```

---

## Why 6 steps? (key design decision)

Cell-ACDC assigns a new `Cell_ID` to each cell **independently per frame**: the same cell may receive label 7 at t=0, label 312 at t=1, and label 89 at t=2. If a cell's segmentation is missed for 1–2 frames, Cell-ACDC reports it as two separate short tracks.

**Step 4 (TrackPy gap-closing)** re-links these per-frame detections using TrackPy's nearest-neighbour linker with a velocity predictor and gap-closing (`memory` parameter), producing a **persistent `particle` ID** per cell that survives across missed frames — matching how Imaris reports tracks. The resulting `trackpy_tracks.csv` is auto-detected by the exporter in step 6 and used as the `Parent` column in the output workbook.

Without step 4, all track-level statistics (Track Duration, Track Speed, Track Straightness, etc.) are undefined because each "track" is a single frame.

---

## Conda environments

| Env name        | Used in steps | Key packages                        |
|-----------------|---------------|-------------------------------------|
| `acdc`          | 1, 2, 3, 5    | Cell-ACDC, tifffile                 |
| `acdc_trackpy`  | 4             | trackpy, scikit-image, scipy        |
| `imaris_xls`    | 6             | openpyxl, pandas, scikit-image      |

The Windows `.bat` runner prompts for all three environment names and activates the correct one at each step automatically.

---

## Data layout

### Raw input (before step 1)
```text
PhaseX_DATA/
├─ NIR/
│  └─ <location>/
│     └─ <field>/
│        ├─ 2025y12m18d_09h00m.tif
│        ├─ 2025y12m18d_10h00m.tif
│        └─ ...
├─ GREEN/
└─ ORANGE/
```

### After step 1 (renamed)
```text
ACDC_IN_PhaseX/
└─ field1/
   └─ B3_1/
      └─ NIR/
         └─ Images/
            ├─ field1_B3_NIR_0.tif
            ├─ field1_B3_NIR_1.tif
            └─ ... field1_B3_NIR_47.tif
```

### After step 2 (segmentation input)
```text
segmentation_input/
├─ position_map.csv                       ← position index used by steps 3–6
├─ Position_1/
│  └─ Images/
│     ├─ field1_B3_NIR.tif               ← stacked 48-frame TIFF
│     └─ field1_B3_metadata.csv
├─ Position_2/
│  └─ Images/
│     ├─ field1_B3_GREEN.tif
│     └─ field1_B3_metadata.csv
└─ ...
```

### After step 3 (Cell-ACDC outputs, inside each Position)
```text
Position_1/Images/
├─ field1_B3_NIR.tif
├─ field1_B3_metadata.csv
├─ field1_B3_NIR_segm.npz               ← segmentation masks (per-frame labels)
└─ field1_B3_NIR_acdc_output.csv        ← per-cell measurements (frame_i, Cell_ID, x, y, ...)
```

### After step 4 (TrackPy linking, auto-saved next to the TIF)
```text
Position_1/Images/
└─ field1_B3_trackpy_tracks.csv         ← frame | id | particle | x | y
                                            particle = persistent gap-closed track ID
```

### After step 6 (export)
```text
pipeline_output/
└─ field1_B3_NIR__Position_1_imaris_like.xlsx    ← 50+ sheet Imaris-like workbook
```

---

## Step-by-step usage (end-to-end)

### Step 1 — Rename raw lab export → ACDC naming convention

Sorts raw TIFFs by datetime embedded in the filename and copies them into the Cell-ACDC folder structure.

```bash
python scripts/rename_to_acdc_input.py Phase3
```

---

### Step 2 — Build `segmentation_input/` + `position_map.csv`

Stacks per-frame TIFFs into multi-page TIFFs (streamed via `TiffWriter`, BigTIFF) and writes Cell-ACDC metadata CSVs. Also auto-copies any pre-existing TrackPy CSVs from the project root.

```bash
python scripts/build_segmentation_input.py Phase3 segmentation_input
```

The `position_map.csv` written here is the key index used by steps 3–6 to select positions. Columns: `Position, field_folder, field_name, location, channel, source_images_dir`.

---

### Step 3 — Run segmentation + tracking (Cell-ACDC CLI)

Reads `position_map.csv`, selects the INI template matching the channel, patches `[paths_to_segment]` and `[paths_to_track]` sections per position, and calls `acdc -p <ini>`.

> **Important:** both `[paths_to_segment]` and `[paths_to_track]` must be patched. Patching only `[paths_info]` (an earlier approach) leaves Cell-ACDC running on stale paths.

```bash
# Single position
python scripts/run_segm_track_from_index.py \
  --exp_root segmentation_input --ini_dir inis --select field1_B3_1_NIR

# All positions
python scripts/run_segm_track_from_index.py \
  --exp_root segmentation_input --ini_dir inis --all
```

Selection key format: `<field_folder>_<location>_<replicate>_<channel>` (e.g. `field1_B3_1_NIR`)  
or pipe-separated: `field1|B3_1|NIR`

---

### Step 4 — TrackPy gap-closing linking (produces persistent track IDs)

This step is **required** for correct track-level statistics in the export.

Re-links Cell-ACDC per-frame detections using TrackPy with:
- **`NearestVelocityPredict`** (default ON): shifts the search window to the cell's predicted next position based on recent velocity. This reduces the effective search radius from ~70 px to ~15 px in dense fields, preventing subnetwork explosions.
- **`memory`** (gap-closing): bridges up to N consecutive missing frames, matching Imaris behaviour.

Centroids are read from `*_acdc_output.csv` (fast); falls back to computing them from `*_segm.npz` via `regionprops` if the CSV is missing.

Output is saved as `<field>_<location>_trackpy_tracks.csv` **inside** `Position_X/Images/`, where the exporter in step 6 auto-detects it — no `--trackpy-csv` flag needed.

```bash
# All positions (default parameters: search_range=15px, memory=3, min_frames=20, predictor ON)
python scripts/run_trackpy_linking.py --exp_root segmentation_input --all

# Custom parameters
python scripts/run_trackpy_linking.py --exp_root segmentation_input --all \
    --search-range-px 15 --memory 3 --min-frames 20 --predictor-span 3

# Single position
python scripts/run_trackpy_linking.py \
    --exp_root segmentation_input --select field1_B3_1_NIR

# Disable velocity predictor (only for slow-moving cells where displacement << inter-cell gap)
python scripts/run_trackpy_linking.py --exp_root segmentation_input --all \
    --no-predictor --search-range-px 20
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--search-range-px` | 15 | With predictor ON: residual search radius after velocity prediction (10–20 px). With predictor OFF: full per-frame displacement — must exceed actual movement. |
| `--memory` | 3 | Max frames a track may disappear before the gap is bridged. |
| `--min-frames` | 20 | Tracks shorter than this are dropped. Set close to total frame count to discard fragments. |
| `--predictor-span` | 3 | Frames of history used to estimate velocity. |
| `--adaptive-stop` | — | Last-resort fallback (e.g. `0.95`) if subnetwork still explodes with predictor ON. |
| `--overwrite` | off | By default, skips positions where the CSV already exists. |

The output CSV columns are: `frame | id | particle | x | y`  
where `particle` is the persistent gap-closed track ID used as `Parent` in the export.

---

### Step 5 — (Optional) Filter short tracks in `*_acdc_output.csv`

Removes rows from `*_acdc_output.csv` whose TrackPy `particle` spans fewer than `min_frames` distinct frames. Uses `(frame, Cell_ID) → particle` lookup from the TrackPy CSV — not the raw `Cell_ID` — so filtering is done at the true track level.

Creates a backup `*_acdc_output_original.csv` on first run (never overwritten on subsequent runs).

```bash
# All positions
python scripts/filter_acdc_output_by_tracklen.py \
    --exp_root segmentation_input --all --min_frames 15

# Single position (with dry-run preview)
python scripts/filter_acdc_output_by_tracklen.py \
    --exp_root segmentation_input --select field1_B3_1_NIR --min_frames 15 --dry_run
```

Objects removed here will **not appear** in the step 6 export (the converter skips any NPZ label not present in the filtered ACDC CSV).

---

### Step 6 — Export Imaris-like Excel workbooks

Batch wrapper that auto-discovers `*_segm.npz`, `*.tif`, `*_acdc_output.csv`, and `*_trackpy_tracks.csv` for each position, then calls the core converter (`acdc_npz_tif_to_imaris_like.py`).

TrackPy CSVs written by step 4 are **auto-detected** from `Position_X/Images/` — no `--trackpy-csv` flag needed.

```bash
# Single position
python scripts/export_imaris_like_from_pipeline.py \
  --exp_root segmentation_input --out_dir pipeline_output \
  --select field1_B3_1_NIR \
  --pixel-size-um 0.108 --z-step-um 1.0 --frame-interval-s 3600

# All positions
python scripts/export_imaris_like_from_pipeline.py \
  --exp_root segmentation_input --out_dir pipeline_output \
  --all \
  --pixel-size-um 0.108 --z-step-um 1.0 --frame-interval-s 3600
```

Output: one `*_imaris_like.xlsx` per position with **50+ sheets** matching the Imaris statistics workbook structure (geometry, intensity, kinematics, ellipsoid, track summaries).

**Key acquisition parameters:**

| Flag | Value | Description |
|---|---|---|
| `--pixel-size-um` | 0.108 | XY pixel size in µm |
| `--z-step-um` | 1.0 | Z-step in µm (used for mesh extrusion and ellipsoid Z-axis) |
| `--frame-interval-s` | 3600 | Seconds per frame (used for speed / velocity / acceleration) |

---

## Windows interactive runner

`run_tunnel_interactive_conda_path.bat` is a fully guided runner that prompts for all parameters and runs all 6 steps in sequence from a user-specified start step.

```text
Edit CONDA_BAT at the top of the file if your Anaconda is not at:
  C:\Users\wasee\anaconda3\condabin\conda.bat
```

The runner collects:
- Phase number and start step (1–6)
- Run mode: `all` or `field` + `location` + `channel`
- Conda environments for each step group
- TrackPy parameters: `search_range`, `memory`, `min_frames`
- ACDC filter threshold (0 = skip step 5)
- Pixel size, Z-step, frame interval

> **Implementation note:** the `.bat` uses `goto` labels rather than `if/else` blocks for run-mode branching because `set /p` (interactive input) inside Windows CMD `if/else` blocks does not capture user input correctly.

---

## Troubleshooting

### TrackPy CLI crash (progressbar signals)

If Cell-ACDC's TrackPy tracking (step 3) crashes in CLI mode with:
```
KernelCliSignals ... initProgressBar
```
apply the patch:
```bash
python tools/patch_trackpy_cli.py
```

### TrackPy subnetwork explosion (step 4)

If step 4 prints `Subnetwork too large`:
1. Ensure `--use-predictor` is ON (it is by default).
2. Reduce `--search-range-px` (try 10 px).
3. As a last resort, add `--adaptive-stop 0.95`.

The script auto-retries with adaptive mode if a subnetwork error is caught — check the console output for the retry message.

### Export skips objects

If the output Excel has fewer cells than expected:
- Check that `*_acdc_output.csv` exists (step 3 output). The exporter silently skips any NPZ label not present in the ACDC CSV.
- If step 5 was run, check the filter threshold (`--min_frames`). Use `--dry_run` to preview which rows would be dropped without modifying the CSV.

---

## Notes

- All paths are relative to the project root; run commands from there.
- For Windows CMD, use `^` for line continuation (as shown above).
- The TrackPy CSV naming convention is `<field_folder>_<location>_trackpy_tracks.csv` (e.g. `field1_B3_trackpy_tracks.csv`). This is shared across all channels for the same physical well — step 4 automatically deduplicates.
- Step 4 skips positions where the CSV already exists unless `--overwrite` is passed.
