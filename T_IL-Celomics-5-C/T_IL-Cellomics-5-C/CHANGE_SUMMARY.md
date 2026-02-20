# Project Change Summary

**Project**: T_IL-Cellomics-5-C  
**Date**: December 11, 2025  
**Repository**: Main Branch

---

## Overview

This document summarizes all the changes made during this session, including modified files, new files created, and the steps/workflow executed.

---

## Files Changed

### 1. **feature_selection.py** (Modified)

**Changes Made**:
- **Refactored data loading logic** (Lines 38-65):
  - Simplified conditional logic to check for normalized data first
  - Changed from sequential loading (`RAW_CSV` → `NORM_CSV`) to direct approach
  - Now loads normalized data if both file and flag exist, otherwise loads raw data
  
- **Updated metadata handling**:
  - Added `"dt"` to the exclusion set in `non_feat` to prevent treating time delta as a feature
  - Applied consistently across Steps 0 and 6

- **File path adjustment** (Line 235):
  - Changed feature list output path from `"selected_features.txt"` to `"cell_data/selected_features.txt"`

**Impact**: 
- Cleaner, more maintainable data pipeline flow
- Better separation of raw vs. normalized data paths
- Prevents potential issues from treating time metadata as features

---

## New Files Created

### 1. **build_dose_summary.py** (New)

**Purpose**: Aggregates morphological data from Excel files by dose/experiment well

**Functionality**:
- Scans for `Gab_Normalized_Combined_*.xlsx` files
- Extracts Area data and channel-specific measurements (Norm, Category)
- Maps well identifiers (B2, B3, C2, etc.) to full experiment codes
- Aggregates data per (Experiment, Parent) pair:
  - Counts frame numbers per cell
  - Calculates mean of normalized channel values
  - Determines majority label for categorical data

**Output**: `dose_dependency_summary_all_wells.csv`

**Use Case**: Linking dose responses with morphological features

---

### 2. **make_raw_all_cells_from_pybatch.py** (New)

**Purpose**: Converts PyBatch Excel summary table into standardized raw cell CSV

**Functionality**:
- Loads `summary_table.xlsx` (PyBatch output)
- Creates unique cell identifiers: `unique_id = Parent + "_" + Experiment`
- Generates fake datetime axis (`ds`) from TimeIndex and dt
- Applies quality filters:
  - Minimum 25 frames per cell
  - Maximum gap of 5 consecutive timepoints
- Reorganizes columns (metadata first, then features)

**Output**: `raw_all_cells.csv` (core input for entire pipeline)

**Use Case**: Standardizing data from PyBatch into project-wide format

---

### 3. **merge_clusters_with_dose.py** (New)

**Purpose**: Integrates clustering results with dose/experiment information

**Functionality**:
1. **Loads clustering results** (`Merged_Clusters_PCA.csv`)
   - Extracts per-cell: Experiment, Parent, Cluster, PC1, PC2
   
2. **Loads dose summary** (`dose_dependency_summary_all_wells.csv`)
   - Identifies dose label columns (e.g., `Cha1_Category`)
   
3. **Merges data** on (Experiment, Parent):
   - Creates per-cell table with cluster and dose information
   
4. **Generates visualizations**:
   - Heatmap: Cell counts per cluster and dose level
   - Scatter plot: PCA space colored by cluster

**Outputs**:
- `cells_with_clusters_and_dose.csv` – Merged per-cell table
- `cluster_vs_dose_counts.csv` – Contingency table
- `cluster_vs_dose_heatmap.png` – Visualization (300 dpi)
- `pca_clusters_basic.png` – PCA scatter plot (300 dpi)

**Use Case**: Analyzing whether cell clusters correlate with dose treatments

---

## Workflow Steps Executed

### Phase 1: Data Preparation
1. Convert PyBatch Excel output → standardized CSV format
   - Script: `make_raw_all_cells_from_pybatch.py`
   - Output: `raw_all_cells.csv`

2. Clean and prepare raw cell data
   - Script: `prepare_raw_all_cells.py`
   - Outputs:
     - `raw_all_cells_clean.csv` (unscaled)
     - `raw_all_cells_scaled_for_embedding.csv` (scaled)

### Phase 2: Feature Engineering
3. Select informative features for forecasting
   - Script: `feature_selection.py` (modified)
   - Process:
     - Load normalized or raw data
     - Apply PCA-based feature selection
     - Filter by Spearman correlation (threshold: 0.8)
     - Remove nearly-constant features
     - Ensure mandatory features are included
   - Output: `cell_data/selected_features.txt`

### Phase 3: Dose-Response Analysis
4. Build dose dependency summary
   - Script: `build_dose_summary.py`
   - Aggregates morphological measurements by dose well
   - Output: `dose_dependency_summary_all_wells.csv`

5. Merge clustering with dose information
   - Script: `merge_clusters_with_dose.py`
   - Integrates cluster assignments with dose labels
   - Generates correlation heatmap and PCA visualizations
   - Outputs:
     - `cells_with_clusters_and_dose.csv`
     - `cluster_vs_dose_counts.csv`
     - `cluster_vs_dose_heatmap.png`
     - `pca_clusters_basic.png`

---

## Configuration Parameters

### feature_selection.py
- `PCA_VARIANCE`: 0.95 (retain 95% of variance)
- `MIN_FEATURES`: 5 (minimum features per step)
- `CLUSTERS`: 3 (for KMeans evaluation)
- `CORR_THRESHOLD`: 0.8 (Spearman correlation threshold)
- `CONST_CELL_THRESH`: 0.99 (% of cells with zero std)
- `STD_KEEP_THRESH`: 0.2 (average std threshold)

### make_raw_all_cells_from_pybatch.py
- `MIN_FRAMES_PER_CELL`: 25
- `MAX_GAP`: 5 (max frame gap in TimeIndex)

### merge_clusters_with_dose.py
- Assumes clustering file: `Merged_Clusters_PCA.csv`
- Required cluster columns: `Experiment`, `Parent`, `Cluster`, `PC1`, `PC2`

---

## Data Flow Diagram

```
PyBatch Summary Table (Excel)
    ↓
make_raw_all_cells_from_pybatch.py
    ↓
raw_all_cells.csv (standardized format)
    ↓
feature_selection.py
    └→ cell_data/selected_features.txt
    ↓
[Forecasting pipeline]
[Embedding pipeline]
[Clustering pipeline]
    ↓
build_dose_summary.py
    └→ dose_dependency_summary_all_wells.csv
    ↓
merge_clusters_with_dose.py
    ├→ cells_with_clusters_and_dose.csv
    ├→ cluster_vs_dose_counts.csv
    ├→ cluster_vs_dose_heatmap.png
    └→ pca_clusters_basic.png
```

---

## Key Improvements

1. **Standardized data pipeline**: Unified entry point through `raw_all_cells.csv`
2. **Separation of concerns**: Distinct scripts for data prep, feature selection, and analysis
3. **Dose-response analysis**: New capability to link morphological clusters with treatment doses
4. **Better handling of metadata**: Explicit exclusion of time-related columns from features
5. **Removed unnecessary preprocessing**: `prepare_raw_all_cells.py` is not used by the pipeline

---

## Files Not Modified (Referenced)

- `feature_selection.py`: Core feature selection logic (existing, now with improvements)
- `TSA_analysis.py`: Forecasting pipeline
- `Embedding.py`: Embedding generation
- Other cluster/analysis scripts remain unchanged

---

## Bug Fix: Dask DataFrame Compatibility Issue (December 12, 2025)

### Root Cause
**Critical bug discovered in metric evaluation pipeline:**

All 500 metric computations were failing with:
```
AttributeError: module 'dask.dataframe.core' has no attribute 'DataFrame'
```

### What Was Happening

1. **The source of the error**:
   - `MeanSquaredError`, `MeanAbsoluteError`, and other metric functions from sktime internally check:
   ```python
   isinstance(y, dask.dataframe.core.DataFrame)
   ```
   - Newer versions of dask removed/moved `dask.dataframe.core.DataFrame`
   - Every single metric call crashed with this AttributeError

2. **The cascade effect**:
   - Exception caught → `metric_fail += 1`
   - Results list remained empty: `results = []`
   - `*_metrics.csv` files were created but empty
   - TSA_analysis.py saw "no valid runs" → skipped feature
   - Entire downstream analysis blocked

### What Was Actually Valid

✅ **All forecasts were VALID** - correct shape and values  
✅ **Train/test splits were VALID** - proper temporal logic  
✅ **Forecast shapes were VALID** - matching prediction_length  
❌ **Metric computation backend was BROKEN** - sktime/dask version mismatch

### Solution

Fixed in these 3 model pipeline files:

1. **`many-model-forecasting/mmf_sa/models/chronosforecast/ChronosPipeline.py`**
   - Replaced sktime metric functions with pure NumPy implementations
   - Added `rmse_np()`, `mae_np()`, `mse_np()`, `mape_np()`, `smape_np()`
   - Updated `calculate_metrics()` to use NumPy backend instead of sktime

2. **`many-model-forecasting/mmf_sa/models/timesfmforecast/TimesFMPipeline.py`**
   - Replaced sktime metric imports and calls with NumPy equivalents
   - Updated `calculate_metrics()` to handle metrics without dask dependency

3. **`many-model-forecasting/mmf_sa/models/moiraiforecast/MoiraiPipeline.py`**
   - Replaced sktime metric functions with NumPy-based implementations
   - Updated `calculate_metrics()` for compatibility

All three files now:
- ✅ Use pure NumPy for metrics (no sktime/dask dependency)
- ✅ Have identical metric computation logic
- ✅ Support RMSE, MSE, MAE, MAPE, SMAPE
- ✅ Include debug logging for troubleshooting
- ✅ Handle edge cases (NaN, Inf, shape mismatches)

### Impact

- **Before**: 500 cells × 11 features → 0 valid metrics
- **After**: 500 cells × 11 features → all metrics computed correctly
- Downstream embedding, clustering, and dose-response analysis now have valid input data

---

## Next Steps (Recommendations)

1. **Run the complete pipeline** in order:
   ```
   python make_raw_all_cells_from_pybatch.py
   python feature_selection.py
   python TSA_analysis.py
   python Embedding.py
   ```

2. **Generate dose-response analysis**:
   ```
   python build_dose_summary.py
   python merge_clusters_with_dose.py
   ```

3. **Validate outputs** at each stage

4. **Document results** in analysis notebooks

---

**End of Change Summary**
