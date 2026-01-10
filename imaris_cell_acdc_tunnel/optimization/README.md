

# Optimization (Cell-ACDC → Imaris Tunnel)

This folder contains the **experimental / optimization stage** of the pipeline that converts **Cell-ACDC outputs** into an **Imaris-like Excel structure** using a **crosswalk mapping file**.

Goal:

* Start from Cell-ACDC results (CSV).
* Generate an Excel workbook that matches Imaris “Statistics export” style (multiple sheets, same headers).
* Use a **crosswalk CSV** to define how each Imaris parameter is computed from Cell-ACDC columns.
* Support a **2-pass approach**:

  * **Pass 1:** direct mapping + row-wise formulas (per object/time).
  * **Pass 2:** auto-derived metrics (mainly tracking-based) computed from the mapped data (e.g., track length, displacement, speed stats).

---

## Folder contents (typical)

* `acdc_to_imaris_structure_2pass.py`
  Main conversion script (2-pass).
* `crosswalk_init_filled_with_status.csv`
  Crosswalk file that defines mapping/formulas + status.
* `requirements.txt`
  Python dependencies for this stage.
* Outputs (generated):

  * `acdc_imaris_like.xlsx` (final workbook)
  * sheets like `CrosswalkReport`, `Index`, `Info`

---

## Input files you need

### 1) Cell-ACDC results CSV

Example:

* `field1_C2_acdc_output_full_try.csv`

Must include at least:

* a cell ID column (usually `Cell_ID`)
* a time/frame column (usually `frame_i`)
* centroid / position columns (depends on your dataset)

### 2) Crosswalk CSV

Example:

* `crosswalk_init_filled_with_status.csv`

This file tells the script **how to fill each Imaris parameter**.

---

## The crosswalk file (most important)

The crosswalk has (at least) these columns:

* `imaris_label`
  The Imaris parameter key used by the script.
  Examples:

  * `Area`
  * `Position X`
  * `Center of Homogeneous Mass::Center of Homogeneous Mass X`
  * `Overall::Total Number of Voxels`

* `cell_acdc_label`
  The Cell-ACDC column name to use directly (optional if you use a formula).

* `formula`
  Expression used to compute the value (optional if direct mapping is enough).

Optional helpful columns (if present):

* `status` / `will_be_computed_in_second_pass` / notes fields (depends on your version)

### How the script chooses what to do

For each `imaris_label`:

1. If `formula` is filled → compute using the formula.
2. Else if `cell_acdc_label` is filled → copy that column directly.
3. Else → leave empty (sheet header only).

---

## Formula rules (how to write formulas)

###  Use direct column names if they are “Python identifier safe”

If the ACDC column name is something like:

* `Area_um2`
* `volume_um3`
* `x_centroid`
  you can write formulas like:

```text
Area_um2
Area_um2 * 0.5
sqrt(Area_um2)
```

### ✅ Use `COL("...")` for any column name (recommended)

If the column name contains `-`, spaces, or special characters, use:

```text
COL("bbox-3") - COL("bbox-1")
```

Example (bounding box length):

```text
(COL("bbox-3") - COL("bbox-1")) * sqrt(cell_area_um2 / cell_area_pxl)
```

### Allowed operators

* `+  -  *  /  **  ( )`

### Allowed functions

* `sqrt(x)`
* `log(x)`
* `exp(x)`
* `abs(x)`
* `clip(x, a, b)`
* `where(cond, a, b)`

---

## Common crosswalk editing examples

### Example 1: Map Area directly

If Cell-ACDC has `cell_area_um2`:

* `imaris_label = Area`
* `cell_acdc_label = cell_area_um2`
* `formula =` *(empty)*

### Example 2: Compute Speed from velocities

If you have per-frame velocity components:

```text
sqrt(COL("velocity_x_um")**2 + COL("velocity_y_um")**2 + COL("velocity_z_um")**2)
```

### Example 3: Fix “invalid syntax” errors for bbox columns

If your formula was written using backticks like:

```text
(`bbox-3`-`bbox-1`)
```

That will FAIL.

Replace it with:

```text
(COL("bbox-3") - COL("bbox-1"))
```

---

## 2-pass logic (what gets computed automatically)

Some parameters can be computed only after we understand tracks over time (same ID across frames).
Examples:

* Track Length (total traveled distance)
* Track Displacement (start → end distance)
* Track Duration (number of frames)
* Track Speed Mean/Min/Max
* Track Straightness (displacement / length)

These are typically marked in the report as:

* `OK (auto-derived)`
  and the script may generate helper columns such as:
* `track_length_total_um`
* `track_displacement_length_um`
* `track_speed_mean`
* etc.

If something is **EMPTY (no mapping)** but you believe it can be derived in pass 2, we add it to the “auto-derived” logic and then reference it in crosswalk as:

```text
COL("track_length_total_um")
```

---

## How to run

### 1) Install dependencies

Inside your environment:

```bash
pip install -r requirements.txt
```

### 2) Run the script

**Windows (PowerShell / CMD):**

```bash
python acdc_to_imaris_structure_2pass.py --acdc "field1_C2_acdc_output_full_try.csv" --crosswalk "crosswalk_init_filled_with_status.csv" --out "acdc_imaris_like.xlsx"
```

**Linux / Mac:**

```bash
python3 acdc_to_imaris_structure_2pass.py --acdc field1_C2_acdc_output_full_try.csv --crosswalk crosswalk_init_filled_with_status.csv --out acdc_imaris_like.xlsx
```

---

## Outputs

The script creates:

* `acdc_imaris_like.xlsx`

Inside it you’ll see:

* Imaris-like sheets (Area, Position X, Volume, …)
* `Index` sheet: mapping between spec sheet names and Excel sheet names
* `CrosswalkReport`: status for every parameter (OK / EMPTY / ERROR)
* `Info`: detected ID/time columns + input file names

---

## Debugging checklist

### If you see: `Unknown name 'X'`

It means your formula uses `X` but it is not a valid column variable name.
Fix: use `COL("X")` or correct the column name.

### If you see: `COL(): column '...' not found`

The ACDC CSV does not have that column.
Fix: check the exact spelling in the CSV header.

### If you see: `invalid syntax`

Usually caused by backticks or weird characters in the formula.
Fix: rewrite using `COL("...")`.

---

## Notes / next steps

This folder is for **experimenting and validating** against real Imaris exports:

* adjust formulas
* add missing derived metrics
* compare output sheets vs Imaris sheets

Once stable, we will move a cleaned “production-ready” version into:
`imaris_cell_acdc_tunnel/active_tunnel/`

---
