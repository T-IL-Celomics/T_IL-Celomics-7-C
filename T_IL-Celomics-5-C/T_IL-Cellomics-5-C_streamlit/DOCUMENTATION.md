# T_IL-Cellomics-5-C Streamlit Pipeline — Comprehensive Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Technology Stack](#2-architecture--technology-stack)
3. [Directory Structure](#3-directory-structure)
4. [Data Formats & Flow](#4-data-formats--flow)
5. [Pipeline Stages (Detailed)](#5-pipeline-stages-detailed)
   - [Stage 0: Dose Extraction](#stage-0-dose-extraction)
   - [Stage 1: Data Preparation](#stage-1-data-preparation)
   - [Stage 2: Feature Selection](#stage-2-feature-selection)
   - [Stage 3: Time-Series Forecasting](#stage-3-time-series-forecasting)
   - [Stage 4: Embedding Generation](#stage-4-embedding-generation)
   - [Stage 5: Curve Fitting](#stage-5-curve-fitting)
   - [Stage 6: Unsupervised Clustering (Embedding-only)](#stage-6-unsupervised-clustering-embedding-only)
   - [Stage 6b: Unsupervised Clustering (Embedding + Fitting)](#stage-6b-unsupervised-clustering-embedding--fitting)
   - [Stage 7: One-way ANOVA](#stage-7-one-way-anova)
   - [Stage 7b: One-way ANOVA (Embedding + Fitting)](#stage-7b-one-way-anova-embedding--fitting)
   - [Stage 8: Descriptive Statistics](#stage-8-descriptive-statistics)
   - [Stage 8b: Descriptive Statistics (Embedding + Fitting)](#stage-8b-descriptive-statistics-embedding--fitting)
   - [Stage 9: Two-Way ANOVA](#stage-9-two-way-anova)
   - [Stage 10: Baseline Comparison](#stage-10-baseline-comparison)
6. [Streamlit App (app.py)](#6-streamlit-app-apppy)
7. [Foundation Models](#7-foundation-models)
8. [Mathematical Models for Curve Fitting](#8-mathematical-models-for-curve-fitting)
9. [Utility & Legacy Scripts](#9-utility--legacy-scripts)
10. [Configuration Reference](#10-configuration-reference)
11. [Environment Variables](#11-environment-variables)
12. [Dependencies](#12-dependencies)
13. [Setup & Deployment](#13-setup--deployment)

---

## 1. Project Overview

This project is an end-to-end analysis pipeline for **cell biology morphokinetic data** from Cellomics imaging experiments. It combines **foundation model time-series forecasting**, **deep learning embeddings**, **mathematical curve fitting**, **unsupervised clustering**, and **statistical testing** to characterize cell populations under different drug treatments and doses.

### Core Objectives

- **Forecast** cell morphokinetic feature trajectories using 16 pre-trained foundation models (Chronos T5, Chronos Bolt, Moirai, TimesFM)
- **Embed** each cell's time-series into a low-dimensional representation using Chronos T5 models + UMAP
- **Fit** 12 mathematical models to each cell's trajectory per feature to extract parameterized signatures
- **Cluster** cells into behavioral sub-populations using KMeans on embedding vectors (and combined embedding + fitting vectors)
- **Statistically compare** clusters across treatments and doses using ANOVA, Two-Way ANOVA, and Tukey HSD
- **Benchmark** foundation models against classical baselines (LSTM, GRU, DLinear, Autoformer)

### Key Innovation

The pipeline introduces two parallel analysis branches:

1. **Embedding-only branch**: Clusters cells using only Chronos T5 embedding vectors + UMAP reduction
2. **Embedding + Fitting branch**: Combines embeddings with curve-fitting parameter vectors for richer cell signatures

These are then compared against a **static (mean-based) clustering** baseline using Two-Way ANOVA.

---

## 2. Architecture & Technology Stack

### Programming Languages & Frameworks

| Component | Technology |
|-----------|-----------|
| GUI | Streamlit (≥1.28.0) |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Foundation Models | Amazon Chronos, Salesforce Moirai (uni2ts), Google TimesFM |
| Forecasting Framework | Many-Model Forecasting (MMF) with PySpark |
| Dimensionality Reduction | UMAP, PCA |
| Clustering | scikit-learn KMeans |
| Curve Fitting | scipy.optimize.curve_fit |
| Statistical Analysis | scipy.stats, statsmodels |
| Visualization | matplotlib, seaborn, plotly |
| Data Processing | pandas, numpy |

### Compute Requirements

- **GPU**: CUDA-capable GPU required (supports multi-GPU: 1-4 GPUs)
- **Java**: Java 17+ required for PySpark
- **Python**: 3.11
- **OS**: Designed for remote Linux GPU servers (runs on Windows for UI only)

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit GUI (app.py)                   │
│              14 tabs × subprocess orchestration            │
├──────────┬───────────┬───────────┬───────────┬────────────┤
│ Stage 0  │ Stage 1   │ Stage 2   │ Stage 3   │ Stage 4    │
│ Dose     │ Data Prep │ Feature   │ Forecast  │ Embedding  │
│ Extract  │           │ Selection │ (MMF)     │ (Chronos)  │
├──────────┴───────────┴───────────┼───────────┼────────────┤
│            Stage 5: Curve Fitting (12 models)              │
├──────────────────────────────────┬────────────────────────┤
│  Branch A: Embedding Clustering  │ Branch B: Emb+Fit      │
│  Stage 6 → 7 → 8                │ Stage 6b → 7b → 8b     │
├──────────────────────────────────┴────────────────────────┤
│       Stage 9: Two-Way ANOVA (dynamic vs static)          │
│       Stage 10: Baselines vs MMF Comparison               │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
T_IL-Cellomics-5-C_streamlit/
├── app.py                                    # Main Streamlit GUI (2850 lines, 14 tabs)
├── my_models_conf.yaml                       # Foundation model configuration
├── well_map.json                             # Well-to-experiment ID mapping
├── requirements.txt                          # Core Python dependencies
├── streamlit_requirements.txt                # Extended Streamlit dependencies
│
├── ── Pipeline Scripts (in execution order) ──
├── build_dose_summary.py                     # Stage 0: Dose extraction
├── make_raw_all_cells_from_pybatch.py        # Stage 1: Data preparation
├── make_raw_all_cells_from_fitted_pybatch.py # Stage 1 alt: From filled data
├── feature_selection.py                      # Stage 2: PCA-based feature selection
├── TSA_analysis.py                           # Stage 3: Forecasting (1 GPU)
├── TSA_analysis_4gpu.py                      # Stage 3: Forecasting (4 GPU sharded)
├── run_4gpus.sh                              # Shell script for 4-GPU parallel
├── Embedding.py                              # Stage 4: Embedding (1 GPU)
├── Embedding_multi_gpu.py                    # Stage 4: Embedding (multi-GPU)
├── fit_cell_trajectory.py                    # Stage 5: Curve fitting
├── fit_cell_trajectory_robust.py             # Stage 5 alt: Robust fitting with bounds
├── embedding_unsupervised_clustering.py      # Stage 6: Clustering (embedding)
├── embedding_fitting_unsupervised_clustering.py  # Stage 6b: Clustering (emb+fit)
├── ANOVA.py                                  # Stage 7: One-way ANOVA
├── embedding_fitting_anova.py                # Stage 7b: One-way ANOVA (emb+fit)
├── descriptive_table_by_cluster.py           # Stage 8: Descriptive stats
├── embedding_fitting_descriptive_table.py    # Stage 8b: Descriptive stats (emb+fit)
├── two_way_anova.py                          # Stage 9: Two-way ANOVA
├── baseline_comparison.py                    # Stage 10: Baselines vs MMF
├── run_baseline_multi_gpu.sh                 # Shell script for multi-GPU baselines
│
├── ── Utility/Analysis Scripts ──
├── prepare_raw_all_cells.py                  # Data cleaning & NaN filling
├── fill_missing_by_best_fit_models.py        # Model-based imputation
├── merge_clusters_with_dose.py               # Merge cluster assignments + dose info
├── compare_embedding_vs_fitting.py           # Random Forest classification comparison
├── 2way_ANOVA_emb_vs_comb.py                 # Embedding vs Emb+Fit two-way ANOVA
├── K3_analysis.py                            # K=3 specific old-vs-new comparison
├── K3_graphs.py                              # K=3 visualization (distribution, ratios)
│
├── ── Output Directories (created at runtime) ──
├── cell_data/                                # Processed data files
├── forecasting/                              # Forecasting results & model selection
├── embeddings/                               # Embedding JSON files
├── fitting/                                  # Curve fitting CSVs and JSONs
├── clustering/                               # Clustering assignments & plots
├── results/                                  # Two-way ANOVA results
├── baseline/                                 # Baseline comparison outputs
├── figures/                                  # Visualization outputs
├── logs/                                     # GPU log files (4-GPU mode)
│
├── many-model-forecasting/                   # Adapted MMF framework
│   ├── mmf_sa/
│   │   ├── LocalForecaster.py
│   │   └── models/
│   │       ├── ChronosPipeline.py
│   │       ├── MoiraiPipeline.py
│   │       ├── TimesFMPipeline.py
│   │       ├── RNNPipeline.py
│   │       └── abstract_model.py
│   └── conf/
│
├── README.md                                 # Project README
├── STREAMLIT_README.md                       # Streamlit user guide
└── CHANGE_SUMMARY.md                         # Changelog
```

---

## 4. Data Formats & Flow

### 4.1 Input Data

**Source**: PyBatch Excel exports from Imaris cell tracking software.

The pipeline expects Excel files (`.xlsx`) containing one sheet per experiment/well, with per-frame measurements for tracked cells.

**Required columns**: `Experiment`, `Parent`, `TimeIndex`, `dt` (time step in minutes), plus 40-50 numeric morphokinetic feature columns.

### 4.2 Core Data Files

| File | Format | Description |
|------|--------|-------------|
| `cell_data/raw_all_cells.csv` | CSV | Standardized input: one row per cell per time frame |
| `cell_data/selected_features.txt` | TXT | One feature name per line (output of feature selection) |
| `cell_data/dose_dependency_summary_all_wells.csv` | CSV | Dose categories per cell |
| `cell_data/MergedAndFilteredExperiment008.csv` | CSV/XLSX | Alternative merged experiment data |
| `well_map.json` | JSON | Maps well IDs (B2, C2, ...) to experiment names |

### 4.3 raw_all_cells.csv Schema

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | string | `{Parent}_{Experiment}` — unique cell identifier |
| `ds` | datetime | Synthetic datetime axis (2000-01-01 + TimeIndex × dt) |
| `Experiment` | string | Full experiment identifier string |
| `Parent` | string | Cell parent/track ID |
| `TimeIndex` | int | Frame number (1-48 typically) |
| `dt` | float | Time step between frames (minutes) |
| *feature1*, *feature2*, ... | float | 40-50 morphokinetic measurement columns |

### 4.4 Morphokinetic Features

Features fall into two categories:

**Morphological** (cell shape): Area, EllipsoidAxisLengthB/C, Ellipticity_oblate/prolate, Sphericity, Eccentricity

**Kinetic** (cell motion): Acceleration, Instantaneous_Speed, Displacement2, Velocity_X/Y, Directional_Change, Current_MSD_1, Min_Distance, Confinement_Ratio, Track_Displacement, and many more

### 4.5 Data Flow Diagram

```
PyBatch Excel Exports
        │
        ▼
┌─── Stage 0 ───┐    ┌─── Stage 1 ───┐
│ Dose Summary   │    │ raw_all_cells  │
│ (per well)     │    │ .csv           │
└───────┬────────┘    └───────┬────────┘
        │                     │
        │              ┌──────▼──────┐
        │              │  Stage 2:   │
        │              │  Feature    │
        │              │  Selection  │
        │              └──────┬──────┘
        │                     │ selected_features.txt
        │              ┌──────▼──────┐
        │              │  Stage 3:   │─── best_model_per_feature.json
        │              │  Forecast   │─── best_t5_model_per_feature.json
        │              │  (MMF)      │─── per-feature metrics CSVs
        │              └──────┬──────┘
        │                     │
        │    ┌────────────────┼────────────────┐
        │    │                │                │
        │    ▼                ▼                ▼
        │  Stage 4:       Stage 5:        (baselines)
        │  Embedding      Curve Fit       Stage 10
        │  (UMAP json)    (12 models)
        │    │                │
        │    ▼                │
        │  Stage 6:           │
        │  Clustering ◄───────┘ Stage 6b:
        │  (emb-only)         │ Clustering
        │    │                │ (emb+fit)
        │    ▼                ▼
        │  Stage 7:         Stage 7b:
        │  ANOVA            ANOVA
        │    │                │
        │    ▼                ▼
        │  Stage 8:         Stage 8b:
        │  Descriptive      Descriptive
        │    │                │
        │    └────────┬───────┘
        │             ▼
        │         Stage 9:
        └────►    Two-Way ANOVA
                 (dynamic vs static)
```

---

## 5. Pipeline Stages (Detailed)

### Stage 0: Dose Extraction

**Script**: `build_dose_summary.py` (146 lines)

**Purpose**: Extract dose/treatment category information from normalized Excel exports and create a unified dose summary.

**Process**:
1. Globs all `Normalized_*.xlsx` files in the script directory
2. Reads each sheet, maps well names (B2, C2, ...) to experiment IDs via `well_map.json`
3. Extracts dose category columns (`Cha*_Category`) from each well
4. Aggregates into a per-cell dose summary CSV

**Output**:
- `cell_data/dose_dependency_summary_all_wells.csv` — Columns: Experiment, Parent, n_frames, Cha1_Category, Cha2_Category, ...

**Parameters**: None (uses `well_map.json` for mapping)

---

### Stage 1: Data Preparation

**Script**: `make_raw_all_cells_from_pybatch.py` (72 lines)

**Purpose**: Convert PyBatch Excel exports into the standardized `raw_all_cells.csv` format required by the pipeline.

**Process**:
1. Loads Excel files from PyBatch output
2. Creates `unique_id` column as `{Parent}_{Experiment}`
3. Builds a synthetic datetime axis (`ds`) from `TimeIndex × dt`
4. Applies quality filters:
   - Minimum 25 frames per cell track
   - Maximum gap of 5 missing frames
5. Organizes columns: metadata first, then feature columns

**Alternative**: `make_raw_all_cells_from_fitted_pybatch.py` — Creates `raw_all_cells.csv` from pre-filled/imputed data tables (using model-based imputation from `fill_missing_by_best_fit_models.py`).

**Output**:
- `cell_data/raw_all_cells.csv`

**Parameters** (env vars):
- `PIPELINE_MIN_FRAMES` — Minimum frames per cell (default: 25)
- `PIPELINE_MAX_GAP` — Maximum gap allowed (default: 5)

---

### Stage 2: Feature Selection

**Script**: `feature_selection.py` (240 lines)

**Purpose**: Select the most informative morphokinetic features using PCA-based analysis with correlation filtering.

**Process**:
1. Load `raw_all_cells.csv`
2. Average features per cell (across time frames)
3. Run PCA on all numeric features
4. Silhouette sweep: test k = 5, 7, 9, 11, ... up to N cells, find optimal k
5. Compute Spearman correlation matrix; remove highly correlated features (threshold = 0.8)
6. Remove near-constant features (coefficient of variation < 0.01)
7. Add mandatory features (user-defined or default set)
8. Save selected feature list + generate visualizations

**Outputs**:
- `cell_data/selected_features.txt` — One feature name per line
- `figures/feature_correlation_heatmap.png`
- `figures/feature_autocorrelation.png`

**Parameters**:
- Correlation threshold: 0.8 (Spearman)
- Silhouette sweep: k in range(5, N, 2)

---

### Stage 3: Time-Series Forecasting

**Scripts**: 
- `TSA_analysis.py` (199 lines) — Single GPU
- `TSA_analysis_4gpu.py` (219 lines) — 4-GPU parallel (feature sharding)
- `run_4gpus.sh` — Shell launcher for 4-GPU mode

**Purpose**: Evaluate all 16 foundation models on each selected feature's time series and identify the best model per feature.

**Process**:
1. Load `raw_all_cells.csv` and selected features
2. Subsample cells (default: 500 cells, configurable)
3. For each feature:
   a. Prepare MMF-compatible DataFrames (`unique_id`, `ds`, `y`)
   b. Initialize `LocalForecaster` with the YAML configuration
   c. Run `evaluate_models()` — backtests all active models
   d. Parse per-model `*_metrics.csv` results
   e. Select best model overall, best Chronos model, best T5 model (by RMSE)
4. Save model selections

**4-GPU Mode**: Features are sharded across GPUs using `SHARD_IDX` and `NUM_SHARDS` environment variables. Each GPU processes its shard independently. Results are saved with `_shard{N}` suffixes and later merged.

**Outputs**:
- `forecasting/best_model_per_feature.json` — Maps feature → best model name
- `forecasting/best_chronos_model_per_feature.json` — Best Chronos-family model
- `forecasting/best_t5_model_per_feature.json` — Best T5 model (used for embeddings)
- `results/{feature}/{model}_metrics.csv` — Per-model backtesting metrics

**Parameters**:
- `PIPELINE_MAX_CELLS` — Max cells to subsample (default: 500, 0 = all)
- Forecasting config from `my_models_conf.yaml`:
  - `prediction_length`: 5
  - `backtest_length`: 5
  - `stride`: 1
  - `metric`: smape
  - `freq`: T (minutely)

---

### Stage 4: Embedding Generation

**Scripts**:
- `Embedding.py` (164 lines) — Single GPU
- `Embedding_multi_gpu.py` (295 lines) — Multi-GPU parallel

**Purpose**: Generate dense embedding vectors for each cell using the best-performing Chronos T5 model per feature, then reduce dimensionality with UMAP.

**Process**:
1. Load `raw_all_cells.csv` and `best_t5_model_per_feature.json`
2. For each feature:
   a. Load the appropriate Chronos T5 model
   b. For each cell, extract the time series and call `model.embed()`
   c. Pool the embedding: select token at position 1, mean over heads → 1D vector
3. For non-Chronos features: use simple mean value as a 1D vector
4. Apply UMAP reduction (default: 3 components) per feature
5. Save combined embedding JSON

**Multi-GPU Mode** (`Embedding_multi_gpu.py`):
- Features are split across N GPUs using Python multiprocessing (`spawn` context)
- Each worker sets `CUDA_VISIBLE_DEVICES` to its GPU
- Workers write partial JSON files; main process merges them

**Outputs**:
- `embeddings/summary_table_Embedding.json` — Per-cell JSON with feature → [dim]-dimensional embedding vectors

**Embedding JSON structure**:
```json
[
  {
    "Experiment": "...",
    "Parent": "...",
    "feature1": [0.123, -0.456, 0.789],
    "feature2": [0.111, 0.222, 0.333],
    ...
  },
  ...
]
```

**Parameters**:
- `PIPELINE_UMAP_DIM` — UMAP output dimensions (default: 3)

---

### Stage 5: Curve Fitting

**Scripts**:
- `fit_cell_trajectory.py` (753 lines) — Standard version
- `fit_cell_trajectory_robust.py` (835 lines) — Robust version with parameter bounds

**Purpose**: Fit 12 mathematical models to each cell's trajectory for every feature, select significant fits, and create embedding-compatible parameter vectors.

**Process**:
1. Load `raw_all_cells.csv`
2. For each cell × feature:
   a. Fit all 12 models using `scipy.optimize.curve_fit` (maxfev=3000 or configurable)
   b. Compute R², RMSE, p-value for each fit
3. Filter significant fits (p-value < 0.05 threshold)
4. Compute NRMSE (Normalized RMSE = RMSE / range)
5. Select top-3 models and best model per cell × feature
6. Generate extensive visualizations:
   - Model distribution plots (per feature)
   - Stacked bar charts (treatment × model)
   - Boxplots (NRMSE by model/feature)
   - Heatmaps (NRMSE by treatment × model)
   - Sunburst plots (Model → Treatment → Feature hierarchy)
7. Extract treatment information from experiment IDs
8. Create JSON embedding vectors from fitting parameters:
   - **Best model log**: `sign(x) × log1p(|x|)` transform
   - **Best model log scaled**: log transform + Z-score standardization
   - Same for top-3 models

**Robust Version** (`fit_cell_trajectory_robust.py`):
- Adds parameter bounds (`BOUNDS` dict) to prevent numerical overflow
- Uses `method="trf"` (Trust Region Reflective) for bounded optimization
- Clips exponential arguments to [-50, 50] range
- Validates predictions over the full x-range before accepting a fit

**Outputs**:
- `fitting/fitting_all_models.csv` — All model fits (Experiment, Parent, Feature, Model, R2, RMSE, pval, param_0..param_4)
- `fitting/fitting_best_with_nrmse.csv` — Best model per cell × feature
- `fitting/fitting_significant_models_with_nrmse.csv` — All significant fits with NRMSE
- `fitting/fitting_top3_with_nrmse.csv` — Top-3 models per cell × feature
- `fitting/fitting_best_model_log.json` — Log-transformed parameter vectors
- `fitting/fitting_best_model_log_scaled.json` — Log + Z-scaled parameter vectors
- `fitting/fitting_top3_models_log.json` / `_log_scaled.json`
- Various PNG figures in `figures/`

**Parameters**:
- `PIPELINE_MAXFEV` — Max curve_fit iterations (default: 3000)
- `PIPELINE_PVAL_THRESHOLD` — P-value threshold (default: 0.05)
- `PIPELINE_TREATMENT_MAP` — Optional JSON mapping well → treatment name

---

### Stage 6: Unsupervised Clustering (Embedding-only)

**Script**: `embedding_unsupervised_clustering.py` (425 lines)

**Purpose**: Cluster cells based on their Chronos T5 embedding vectors using KMeans, with silhouette-based k selection and dose-aware analysis.

**Process**:
1. Load `summary_table_Embedding.json` and merged experiment data
2. Scale embedding vectors per feature (StandardScaler)
3. Flatten all feature embeddings into a single vector per cell
4. Run PCA for visualization (default: 2 components)
5. Silhouette sweep: test k = 2 to 10, select top N best k values
6. For each best k:
   a. Run KMeans clustering
   b. Compute per-treatment silhouette scores
   c. Create PCA scatter plots (by cluster, by treatment)
   d. If dose data available:
      - Create per-treatment dose-specific PCA scatter plots
      - Build contingency tables (dose × cluster)
      - Generate heatmaps (absolute counts + row-normalized percentages)
   e. Merge cluster assignments with original data
7. Save all outputs

**Dose Label Construction**: The script parses experiment IDs to extract channel codes (4-char segments), maps them to treatment names (e.g., "METR", "NNIR"), and builds dose labels like `METR:Pos|GABY:Low`. A configurable control channel (e.g., "NNIR") can be excluded from labels.

**Outputs** (per k value):
- `clustering/cluster_assignments_k{k}.csv`
- `clustering/Merged_Clusters_PCA_k{k}.csv`
- `clustering/PCA_KMeans_k{k}.png`
- `clustering/PCA_KMeans_by_Treatment_k{k}.png`
- `clustering/cluster_vs_dose_k{k}_{treatment}.csv`
- `clustering/cluster_vs_dose_heatmap_k{k}_{treatment}.png`

**Parameters**:
- `PIPELINE_K_MIN` / `PIPELINE_K_MAX` — Silhouette sweep range (default: 2–10)
- `PIPELINE_NUM_BEST_K` — Number of top-k values to process (default: 2)
- `PIPELINE_KMEANS_N_INIT` — KMeans initializations (default: 10)
- `PIPELINE_KMEANS_SEED` — Random seed (default: 42)
- `PIPELINE_PCA_COMPONENTS` — PCA visualization dims (default: 2)
- `PIPELINE_CONTROL_CHANNEL` — Control channel to exclude from dose labels

---

### Stage 6b: Unsupervised Clustering (Embedding + Fitting)

**Script**: `embedding_fitting_unsupervised_clustering.py` (574 lines)

**Purpose**: Combine embedding vectors with curve-fitting parameter vectors for richer cell signatures, then cluster.

**Process**:
1. Load `summary_table_Embedding.json` and `fitting_best_model_log_scaled.json`
2. Scale embedding vectors per feature
3. Combine: for each cell × feature, concatenate `embedding_scaled[feature] + fitting[feature]`
4. Save combined JSON (`embedding_fitting_combined_by_feature_scaled.json`)
5. Flatten all combined vectors into one vector per cell
6. PCA + silhouette sweep + KMeans (same as Stage 6)
7. Dose-aware analysis (same as Stage 6)
8. Generate descriptive tables by cluster (inline — not a separate stage)

**Key Difference from Stage 6**: Each cell's vector is approximately twice as long because it contains both the UMAP-reduced embedding (3D per feature) and the fitting parameters (variable length per feature, depending on the best model's parameter count).

**Outputs**:
- `fitting/embedding_fitting_combined_by_feature_scaled.json`
- `fitting/embedding_fitting_cluster_assignments_k{k}.csv`
- `fitting/embedding_fitting_Merged_Clusters_PCA_k{k}.csv`
- `fitting/embedding_fitting_PCA_KMeans_k{k}.png`
- `fitting/embedding_fitting_PCA_KMeans_by_Treatment_k{k}.png`
- `fitting/embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells_k{k}.xlsx`
- Per-treatment dose analysis files (same pattern as Stage 6)

---

### Stage 7: One-way ANOVA

**Script**: `ANOVA.py` (106 lines)

**Purpose**: Test whether morphokinetic features differ significantly across embedding-based clusters.

**Process**:
1. Auto-discover all `clustering/Merged_Clusters_PCA_k{k}.csv` files
2. For each k value:
   a. Average features per unique cell (Experiment × Parent)
   b. For each feature, run `scipy.stats.f_oneway` across cluster groups
   c. Compute full ANOVA table: SS_between, SS_within, SS_total, df, MS, F-statistic, p-value
3. Style output: highlight p-values below threshold in blue
4. Save to Excel with multi-level column headers

**Outputs**:
- `clustering/ANOVA - OneWay_k{k}.xlsx` — Styled Excel with ANOVA results

**Parameters**:
- `PIPELINE_ANOVA_PVAL` — Highlight threshold (default: 0.05)

---

### Stage 7b: One-way ANOVA (Embedding + Fitting)

**Script**: `embedding_fitting_anova.py` (103 lines)

**Purpose**: Same as Stage 7 but for embedding+fitting clusters.

**Process**: Identical to ANOVA.py but operates on `fitting/embedding_fitting_Merged_Clusters_PCA_k{k}.csv` files.

**Outputs**:
- `fitting/embedding_fitting_ANOVA - OneWay_k{k}.xlsx`

---

### Stage 8: Descriptive Statistics

**Script**: `descriptive_table_by_cluster.py` (70 lines)

**Purpose**: Compute per-cluster summary statistics for all features.

**Process**:
1. Auto-discover `clustering/Merged_Clusters_PCA_k{k}.csv` files
2. For each k value:
   a. Average features per unique cell within each cluster
   b. For each feature × cluster: compute Mean, Std, SE, 95% CI bounds

**Outputs**:
- `clustering/Descriptive_Table_By_Cluster_UniqueCells_k{k}.xlsx`

**Statistics computed per feature per cluster**:
- Mean
- Standard Deviation
- Standard Error (SE = Std / √N)
- 95% Confidence Interval Lower/Upper (Mean ± 1.96 × SE)

**Parameters**:
- `PIPELINE_CI_ZSCORE` — Z-score multiplier (default: 1.96 for 95% CI)

---

### Stage 8b: Descriptive Statistics (Embedding + Fitting)

**Script**: `embedding_fitting_descriptive_table.py` (70 lines)

**Purpose**: Same as Stage 8 but for embedding+fitting clusters.

**Outputs**:
- `fitting/embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells_k{k}.xlsx`

---

### Stage 9: Two-Way ANOVA

**Script**: `two_way_anova.py` (556 lines)

**Purpose**: Compare dynamic clustering methods (embedding-based, embedding+fitting-based) against static (mean-based) clustering using two-way ANOVA.

**Process**: Runs **two separate comparisons** for every measurement feature:

1. **Embedding clustering vs Static clustering**
2. **Embedding+Fitting clustering vs Static clustering**

For each comparison:
1. Load dynamic clustering CSV and static clustering CSV
2. Identify common numeric features
3. Average per unique cell (Experiment × Parent) within each method
4. Stack both methods and run `ols(feature ~ C(Method) * C(Cluster))` per feature
5. Extract three p-values from Type II ANOVA:
   - **Method** main effect (does clustering method matter?)
   - **Cluster** main effect (do clusters differ?)
   - **Method × Cluster** interaction (does the method affect how clusters differ?)
6. Apply FDR correction (Benjamini-Hochberg) across all features
7. Generate Z-scored feature-mean heatmaps (side-by-side comparison)
8. Run Tukey HSD per feature within each method, plot FDR-adjusted p-value heatmaps

**Outputs** (per comparison per k):
- `results/two_way_anova_emb_vs_static_k{k}.xlsx` / `.csv`
- `results/two_way_anova_embfit_vs_static_k{k}.xlsx` / `.csv`
- `results/*_zscore_heatmap.png` — Side-by-side Z-scored feature means
- `results/*_tukey_heatmap.png` — Tukey HSD p-value heatmaps

**Inputs required**:
- Dynamic: `clustering/Merged_Clusters_PCA_k{k}.csv` or `fitting/embedding_fitting_Merged_Clusters_PCA_k{k}.csv`
- Static: `cell_data/static_clustering_raw_data.csv`

---

### Stage 10: Baseline Comparison

**Script**: `baseline_comparison.py` (346 lines)

**Purpose**: Train simple baseline models and compare their forecasting performance against the best MMF foundation model per feature.

**Baseline Models**:

| Model | Description |
|-------|-------------|
| SimpleLSTM | Single-layer LSTM (hidden_size=64, 1 epoch) |
| SimpleGRU | Single-layer GRU (hidden_size=64, 1 epoch) |
| SimpleDLinear | Linear layer on flattened input |
| SimpleAutoformer | Simplified Autoformer with decomposition |

**Process**:
1. Load `raw_all_cells.csv` and selected features
2. **Auto-discover training cells**: Parse existing MMF `*_metrics.csv` files to identify which cells were used for forecasting (ensures fair comparison)
3. For each feature:
   a. Load the best MMF model's metrics for that feature
   b. For each baseline model:
      - Train on the same cells with the same context/prediction split
      - Compute MSE, MAE, RMSE
   c. Compute `% lower` metrics: `(baseline - mmf) / baseline × 100`
      - Positive = MMF is better; Negative = baseline is better
4. Generate comparison table and summary

**Multi-GPU Support**: Uses `run_baseline_multi_gpu.sh` to shard features across N GPUs (same approach as TSA_analysis_4gpu.py).

**Outputs**:
- `baseline/baseline_comparison.csv` — Full comparison table
- `baseline/baseline_comparison.json` — JSON version
- `baseline/baseline_comparison_summary.txt` — Headline summary (avg % improvement per baseline)

---

## 6. Streamlit App (app.py)

### Overview

The `app.py` file (2850 lines) is the main entry point — a Streamlit-based GUI that orchestrates the entire pipeline. It provides a no-code interface to configure, run, and monitor all pipeline stages.

### Launch

```bash
streamlit run app.py
```

### Tab Layout

| Tab # | Name | Stage |
|-------|------|-------|
| 0 | 💊 Dose | Stage 0: Dose Extraction |
| 1 | 📊 Data Prep | Stage 1: Data Preparation |
| 2 | 🎯 Features | Stage 2: Feature Selection |
| 3 | 📈 Forecast | Stage 3: Time-Series Forecasting |
| 4 | 🧬 Embed | Stage 4: Embedding Generation |
| 5 | 📐 Fit | Stage 5: Curve Fitting |
| 6 | 🔮 Cluster | Stage 6: Unsupervised Clustering |
| 7 | 📊 ANOVA | Stage 7: One-way ANOVA |
| 8 | 📋 Descriptive | Stage 8: Descriptive Statistics |
| 9 | 🔗 Emb+Fit Cluster | Stage 6b: Emb+Fit Clustering |
| 10 | 📊 Emb+Fit ANOVA | Stage 7b: Emb+Fit ANOVA |
| 11 | 📋 Emb+Fit Desc | Stage 8b: Emb+Fit Descriptive |
| 12 | 📊 Two-Way ANOVA | Stage 9: Two-Way ANOVA |
| 13 | 🧠 Baselines vs MMF | Stage 10: Baseline Comparison |

### Key Features

#### Pipeline Orchestration
- **"Run All Enabled Stages"** button in the sidebar: Executes all enabled stages in order
- **Skip-existing mode**: Automatically skip stages that have existing outputs
- **Stop controls**: Each stage has a stop button; multi-GPU processes can be interrupted
- **Dependency checking**: Each stage validates that its required inputs exist before running

#### Sidebar Controls
- **GPU Mode**: Select between Single GPU and 4-GPU Parallel (for forecasting/embedding)
- **Stage Toggles**: Enable/disable individual stages (disabled stages are skipped in "Run All")
- **Parameter Configuration**: Per-stage parameters with sensible defaults

#### Per-Tab Features
- View script source code (expandable)
- View existing outputs (CSVs, JSONs, images)
- Configure stage-specific parameters
- Dependency status indicator
- Run button with spinner + output display

#### Model Configuration Editor (Tab 3)
The Forecasting tab includes an interactive YAML editor for `my_models_conf.yaml`:
- Adjust prediction_length, backtest_length, stride, metric
- Toggle individual foundation models on/off
- Preview raw YAML
- Save changes with change detection

#### Process Management
- Background process monitoring for multi-GPU runs
- Elapsed time display
- Per-GPU log file viewers
- Kill process + cleanup orphaned GPU processes
- Session state persistence across Streamlit reruns

### Script Registry

All scripts are registered in the `SCRIPTS` dictionary, which defines for each stage:
- `file`: Python script filename
- `description`: Human-readable description
- `outputs`: Expected output file patterns (for checking existence)
- `deps`: Required input files/stages (for dependency validation)
- `figures`: Output directories for generated visualizations

### Environment Variable Passing

The `get_pipeline_env()` function constructs environment variables for each stage based on the user's UI selections and passes them to `subprocess.Popen`. This allows scripts to run independently (from CLI) or through the Streamlit GUI with identical configuration.

---

## 7. Foundation Models

Configured in `my_models_conf.yaml` (132 lines), the pipeline supports 16 pre-trained foundation models:

### Amazon Chronos T5 (supports `.embed()`)

| Model | HuggingFace ID | Parameters |
|-------|---------------|------------|
| ChronosT5Tiny | amazon/chronos-t5-tiny | ~8M |
| ChronosT5Mini | amazon/chronos-t5-mini | ~20M |
| ChronosT5Small | amazon/chronos-t5-small | ~46M |
| ChronosT5Base | amazon/chronos-t5-base | ~200M |
| ChronosT5Large | amazon/chronos-t5-large | ~710M |

### Amazon Chronos Bolt (faster inference, no `.embed()`)

| Model | HuggingFace ID |
|-------|---------------|
| ChronosBoltTiny | amazon/chronos-bolt-tiny |
| ChronosBoltMini | amazon/chronos-bolt-mini |
| ChronosBoltSmall | amazon/chronos-bolt-small |
| ChronosBoltBase | amazon/chronos-bolt-base |

### Salesforce Moirai

| Model | HuggingFace ID |
|-------|---------------|
| MoiraiSmall | Salesforce/moirai-1.1-R-small |
| MoiraiBase | Salesforce/moirai-1.1-R-base |
| MoiraiLarge | Salesforce/moirai-1.1-R-large |
| MoiraiMoESmall | Salesforce/moirai-1.1-MoE-small |
| MoiraiMoEBase | Salesforce/moirai-1.1-MoE-base |

### Google TimesFM

| Model | HuggingFace ID |
|-------|---------------|
| TimesFM_1_0_200m | google/timesfm-1.0-200m |
| TimesFM_2_0_500m | google/timesfm-2.0-500m |

**Note**: Only Chronos T5 models support the `.embed()` method used for Stage 4 (Embedding Generation). The forecasting stage (Stage 3) uses the best-performing T5 model per feature for embeddings, regardless of which model type achieved the lowest overall RMSE.

---

## 8. Mathematical Models for Curve Fitting

The pipeline fits 12 mathematical models to each cell's trajectory:

| Model | Formula | Parameters |
|-------|---------|------------|
| Linear | $y = ax + b$ | 2 |
| Exponential | $y = a \cdot e^{bx}$ | 2 |
| Logarithmic | $y = a \cdot \ln(bx + 1)$ | 2 |
| Logistic | $y = \frac{c}{1 + a \cdot e^{-bx}}$ | 3 |
| Power | $y = a \cdot x^b$ | 2 |
| Inverse | $y = \frac{a}{x + b}$ | 2 |
| Quadratic | $y = ax^2 + bx + c$ | 3 |
| Sigmoid | $y = d + \frac{a - d}{1 + (x/c)^b}$ | 4 |
| Gompertz | $y = a \cdot e^{-b \cdot e^{-cx}}$ | 3 |
| Weibull | $y = a - b \cdot e^{-cx^2}$ | 3 |
| Poly3 | $y = ax^3 + bx^2 + cx + d$ | 4 |
| Poly4 | $y = ax^4 + bx^3 + cx^2 + dx + e$ | 5 |

### JSON Embedding Vector Construction

For each cell, fitting parameters are transformed into embedding-compatible vectors:

1. **Raw parameters**: `[param_0, param_1, ..., param_N]` for the best model
2. **Log transform**: `sign(x) × log1p(|x|)` — handles negative values
3. **Z-score scaling**: Standardize across all cells
4. **NaN imputation**: Mean imputation via sklearn `SimpleImputer`

---

## 9. Utility & Legacy Scripts

### Active Utility Scripts

| Script | Purpose |
|--------|---------|
| `prepare_raw_all_cells.py` (87 lines) | Clean raw data: drop constant features, interpolate NaN within tracks, report statistics |
| `fill_missing_by_best_fit_models.py` (302 lines) | Model-based imputation: use best-fit curve to fill missing values inside observed windows, then linear interpolation, no extrapolation |
| `merge_clusters_with_dose.py` (135 lines) | Standalone merge: cluster assignments + dose information → contingency tables + heatmaps |

### Legacy/Exploratory Scripts

| Script | Purpose |
|--------|---------|
| `compare_embedding_vs_fitting.py` (143 lines) | Random Forest classification to compare embedding-only vs embedding+fitting clustering quality |
| `2way_ANOVA_emb_vs_comb.py` (134 lines) | Two-way ANOVA comparing embedding vs embedding+fitting (predecessor to `two_way_anova.py`) |
| `K3_analysis.py` (122 lines) | K=3 specific analysis: old project vs new project comparison using two-way ANOVA + Tukey HSD |
| `K3_graphs.py` (301 lines) | K=3 visualization: cell distribution per treatment, normalized feature ratios (G0/G1/G2), morphological vs kinetic feature breakdown |

---

## 10. Configuration Reference

### my_models_conf.yaml

```yaml
forecasting:
  freq: T                    # Minutely frequency
  prediction_length: 5       # Forecast horizon (frames)
  backtest_length: 5         # Backtesting window
  stride: 1                  # Sliding window stride
  metric: smape              # Evaluation metric
  limit_num_series: -1       # -1 = no limit
  active_models:             # List of enabled model keys
    - ChronosT5Tiny
    - ChronosT5Mini
    - ...

models:
  ChronosT5Tiny:
    module: mmf_sa.models.ChronosPipeline
    framework: ChronosT5
    model_name: amazon/chronos-t5-tiny
  ...
```

### well_map.json

Maps well identifiers to full experiment IDs:

```json
{
  "B2": "XXXXX210124XXXXB02XXXXXXXXXX",
  "C2": "XXXXX210124XXXXC02XXXXXXXXXX",
  ...
}
```

12 wells mapped: B2-B4, C2-C4, D2-D4, E2-E4.

---

## 11. Environment Variables

All pipeline scripts are configured via environment variables (set by `app.py` or manually):

### Data Paths
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_RAW_CSV` | `cell_data/raw_all_cells.csv` | Input data file |
| `PIPELINE_MERGED_CSV` | `cell_data/MergedAndFilteredExperiment008.csv` | Merged experiment data |
| `PIPELINE_FEATURES_FILE` | `cell_data/selected_features.txt` | Selected features list |
| `PIPELINE_DOSE_CSV` | auto-detected | Dose dependency summary CSV |
| `PIPELINE_EMBEDDING_JSON` | `embeddings/summary_table_Embedding.json` | Embedding output |
| `PIPELINE_FITTING_JSON` | `fitting/fitting_best_model_log_scaled.json` | Fitting parameters |
| `PIPELINE_STATIC_CSV` | `cell_data/static_clustering_raw_data.csv` | Static clustering data |
| `PIPELINE_CHRONOS_MODEL_DICT` | `forecasting/best_t5_model_per_feature.json` | T5 model selection |

### Forecasting
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_MAX_CELLS` | `500` | Max cells to subsample (0 = all) |
| `SHARD_IDX` | `0` | GPU shard index (4-GPU mode) |
| `NUM_SHARDS` | `1` | Total shards (4-GPU mode) |

### Clustering
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_K_MIN` | `2` | Min k for silhouette sweep |
| `PIPELINE_K_MAX` | `10` | Max k for silhouette sweep |
| `PIPELINE_NUM_BEST_K` | `2` | Number of best-k values to keep |
| `PIPELINE_KMEANS_N_INIT` | `10` | KMeans initializations |
| `PIPELINE_KMEANS_SEED` | `42` | Random seed |
| `PIPELINE_PCA_COMPONENTS` | `2` | PCA dims for visualization |

### Curve Fitting
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_MAXFEV` | `3000` | Max curve_fit iterations |
| `PIPELINE_PVAL_THRESHOLD` | `0.05` | Significance threshold |

### Statistics
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_ANOVA_PVAL` | `0.05` | ANOVA highlight threshold |
| `PIPELINE_CI_ZSCORE` | `1.96` | CI z-score (1.96 = 95%) |

### Embedding
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_UMAP_DIM` | `3` | UMAP output dimensions |

### Treatments
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_TREATMENTS` | `CON0, BRCACON1, ...` | Comma-separated treatment names |
| `PIPELINE_CONTROL_CHANNEL` | (empty) | Control channel to exclude from dose labels |

---

## 12. Dependencies

### Core Dependencies (`requirements.txt`)

```
numpy
pandas
torch
chronos-forecasting
uni2ts
timesfm
omegaconf
scikit-learn
scipy
umap-learn
matplotlib
seaborn
plotly
openpyxl
statsmodels
```

### Extended Streamlit Dependencies (`streamlit_requirements.txt`)

Additional packages for the full Streamlit deployment:
- `streamlit>=1.28.0`
- `pytorch-lightning`
- `jax`, `jaxlib`
- `transformers`
- `pyspark`
- `dask`

### System Requirements
- Python 3.11
- CUDA-capable GPU (required for forecasting and embedding)
- Java 17+ (for PySpark/MMF)
- `bash` (for multi-GPU shell scripts)

---

## 13. Setup & Deployment

### Installation

```bash
# 1. Clone the repository
cd T_IL-Cellomics-5-C_streamlit

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r streamlit_requirements.txt

# 4. Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"

# 5. Launch
streamlit run app.py
```

### Data Preparation

1. Place PyBatch Excel exports in accessible location
2. Configure `well_map.json` with your experiment IDs
3. Run Stage 0 (Dose Extraction) and Stage 1 (Data Preparation)
4. Or manually create `cell_data/raw_all_cells.csv`

### Running the Pipeline

**Option A: Streamlit GUI (recommended)**
- Launch `streamlit run app.py`
- Configure parameters in sidebar
- Click "Run All Enabled Stages" or run individual stages

**Option B: Command Line**
```bash
# Run individual stages
python build_dose_summary.py
python make_raw_all_cells_from_pybatch.py
python feature_selection.py
python TSA_analysis.py            # single GPU
bash run_4gpus.sh                 # 4-GPU parallel
python Embedding.py               # single GPU
python Embedding_multi_gpu.py --num_gpus=4  # multi-GPU
python fit_cell_trajectory.py
python embedding_unsupervised_clustering.py
python ANOVA.py
python descriptive_table_by_cluster.py
python embedding_fitting_unsupervised_clustering.py
python embedding_fitting_anova.py
python embedding_fitting_descriptive_table.py
python two_way_anova.py
python baseline_comparison.py
```

### Known Issues

- **Dask compatibility**: sktime metric functions may crash with newer dask versions. Fixed by using pure NumPy implementations in MMF pipeline files.
- **Windows**: Multi-GPU shell scripts (`run_4gpus.sh`, `run_baseline_multi_gpu.sh`) require WSL or a Linux environment.
- **Memory**: Large models (ChronosT5Large, MoiraiLarge) may require 16+ GB GPU VRAM.

---

*This documentation was generated from a comprehensive analysis of all Python source files, configuration files, and READMEs in the T_IL-Cellomics-5-C_streamlit directory.*
