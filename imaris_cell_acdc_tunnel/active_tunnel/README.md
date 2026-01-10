# Active Tunnel (Future / Production-Ready)

This folder will contain the **stable, production-ready version** of the “Imaris ↔ Cell-ACDC tunnel” after we finish the optimization stage.

Right now, all development happens in:

* `../optimization/`

When the logic becomes reliable, we will move only the **clean + tested** parts here.

---

## What “Active Tunnel” will do (planned)

### 1) Input (from the pipeline)

* **Cell-ACDC output CSV** (segmentation + tracking + features)
* **Crosswalk CSV** (mapping + formulas)
* Optional: configuration file (channel name, units, time offset, etc.)

### 2) Processing

* **Pass 1:** map ACDC columns → Imaris-like parameters using the crosswalk
* **Pass 2:** compute derived metrics that require full tracks/time-series:

  * Track Length / Duration
  * Track Displacement (X/Y/Z + total)
  * Track Speed (mean/min/max)
  * Track Velocity (mean/min/max)
  * Straightness
  * Other parameters that Imaris reports but ACDC doesn’t provide directly

### 3) Output (for downstream usage)

* An **Imaris-like Excel workbook** (`.xlsx`) with:

  * the same sheet structure and headers as Imaris statistics export
  * filled values wherever possible
  * a report sheet showing what was computed vs missing

---

## What will be included here (when ready)

* A single main script (or small module) that is:

  * minimal dependencies
  * stable CLI interface
  * well tested on multiple experiments
* A “final” crosswalk template + documentation
* Clear logging + error handling
* Unit handling and consistent naming conventions
* Optional: integration hooks for the full project pipeline (the “full solution”)

---

## What will NOT be included here

* experimental scripts
* messy crosswalk drafts
* temporary debugging notebooks/files
* incomplete mapping attempts

Those stay in:

* `../optimization/`

---

## Planned checklist before moving to Active Tunnel

* [ ] Crosswalk format finalized (columns + naming conventions)
* [ ] All key Imaris sheets filled (or clearly explained why not)
* [ ] Auto-derived track metrics validated vs Imaris on real data
* [ ] Reproducible run instructions tested on Windows + Linux
* [ ] Clear “known limitations” documented
* [ ] Test dataset + expected output (small sample) added

---

## Status

**Not implemented yet** — this folder is a placeholder for the final version after optimization is done.
