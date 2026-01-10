```markdown
# Imaris ↔ Cell-ACDC Tunnel

This folder contains tools to **convert Cell-ACDC outputs** (CSV tables of cell measurements + tracking) into an **Imaris-like Excel workbook** (multiple sheets + Imaris-style headers), using a **crosswalk mapping file**.

The goal is to make Cell-ACDC results easier to:
- compare against Imaris exports,
- plug into workflows that expect Imaris-style “Statistics / Track / Surface” tables,
- iteratively improve mapping until the output matches the Imaris structure.

---

## Folder structure

```

imaris_cell_acdc_tunnel/
├─ optimization/
│  ├─ (scripts + crosswalk experiments + validation vs Imaris)
│  └─ README.md
└─ active_tunnel/
├─ (future: stable/production-ready version for pipeline)
└─ README.md

````

- **optimization/**: where we are currently working (crosswalk tuning, second-pass derived metrics, debugging).
- **active_tunnel/**: future folder for the “final” stable version that will be integrated into the full project pipeline.

---

## What the “tunnel” does

### Input
1) **Cell-ACDC CSV**  
Example: `field1_C2_acdc_output_full_try.csv`

2) **Crosswalk CSV** (mapping rules)  
Example: `crosswalk_init_filled_with_status.csv`

The crosswalk tells the converter how to fill each Imaris-like sheet/column:
- which ACDC column to use directly, OR
- a formula to compute it, OR
- mark it as “second pass” derived (tracking/summary metrics computed after the first pass).

### Output
An **Imaris-like Excel workbook** with multiple sheets (Area, Position X/Y/Z, Speed, Track stats, etc.), plus:
- `Index` sheet (maps spec sheet name → Excel sheet name)
- `CrosswalkReport` sheet (what worked / missing / errors)
- `Info` sheet (input files, detected time/id columns, notes)

---

## How the crosswalk works

The crosswalk file is a CSV table with (at least) these columns:

- `imaris_label`  
  The Imaris parameter key we want to fill.  
  Examples:
  - `Area`
  - `Position X`
  - `Center of Homogeneous Mass::Center of Homogeneous Mass X`
  - `Overall::Total Number of Surfaces`

- `cell_acdc_label`  
  A column name from the Cell-ACDC CSV (optional if you use `formula`).  
  Example:
  - `Area_um2`
  - `x_centroid`
  - `NIR_mean`

- `formula`  
  A safe expression evaluated row-wise, using ACDC columns.  
  Use formulas when you need scaling, transformations, or to reference columns with special characters.

### Formula rules (important)
You can write formulas using:
- arithmetic: `+ - * / ** ( )`
- ACDC columns that are valid identifiers directly (e.g. `Area_um2`)
- `COL("exact column name")` for any column name (including ones like `bbox-0`, `bbox-1`, etc.)
- basic functions: `sqrt, log, exp, abs, clip, where`

**Examples**
```text
# direct column reference (if the column name is identifier-safe)
Area_um2

# using COL() for a column that contains "-" or spaces
(COL("bbox-3") - COL("bbox-1")) * sqrt(cell_area_um2/cell_area_pxl)

# compute speed from velocity (if velocity exists)
sqrt(COL("velocity_x_um")**2 + COL("velocity_y_um")**2 + COL("velocity_z_um")**2)

# background-corrected mean (example idea)
COL("NIR_mean") - COL("NIR_autoBkgr_bkgrVal_median")
````

### Overall::* entries

Rows beginning with `Overall::` represent **summary statistics**, not per-object rows.
They typically require special functions (example patterns supported by the script):

* `TOTAL_COUNT()`
* `PER_TIME_COUNT()`
* `TOTAL_SUM(ColumnName)`
* `PER_TIME_SUM(ColumnName)`

If your crosswalk has `Overall::...` but no supported formula, it will show as **EMPTY** in `CrosswalkReport`.

---

## Two-pass concept (why “optimization” exists)

Some Imaris-style values are not present directly in the Cell-ACDC CSV, but can be derived **after** we build a consistent per-frame/per-cell table.

Examples of **second-pass** derivations:

* Track length (sum of step distances per track)
* Track duration (number of frames per track)
* Track displacement (distance between first and last position)
* Speed/velocity stats over the track (mean/min/max)

So the pipeline is:

1. **Pass 1:** fill everything possible directly from ACDC + formulas
2. **Pass 2:** compute missing tracking/summary metrics by grouping by Track ID / Cell ID over time

The `optimization/` folder is where we iterate until the output matches the Imaris export as closely as possible.

---

## Quick start (high-level)

1. Create a Python environment and install requirements
   (see `optimization/README.md` for exact commands)

2. Run the converter script:

* Inputs: ACDC CSV + crosswalk CSV
* Output: Imaris-like `.xlsx`

3. Open the output workbook and check `CrosswalkReport`:

* **OK**: the mapping worked
* **EMPTY (no mapping)**: needs crosswalk entry/formula or second-pass derivation
* **ERROR computing**: formula needs fixing (missing column, wrong syntax, etc.)

---

## Next steps

* See **`optimization/README.md`** for the current working script, exact run command, and how we validate against a real Imaris `.xls` export.
* Future stable integration will be documented in **`active_tunnel/README.md`**.

---

**Maintainer notes**

* Keep crosswalk changes versioned (small commits) because formulas evolve during validation.
* Prefer `COL("...")` in formulas whenever a column name contains `-`, spaces, or special characters.

```
```
