# Time-Series Cell Motility Analysis for Cancer Characterization

> A computational pipeline for identifying aggressive tumor cell subpopulations through motility analysis, deep learning embeddings, and clustering — validated against the Imaris commercial platform.

---

## Overview

This project explores **tumor cell motility** in breast and ovarian cancer using time-lapse fluorescence microscopy. Cells are tracked under various treatments (e.g., HGF/SF activation or inhibition), and detailed kinetic and morphological features are extracted to reveal distinct behavioral subpopulations.

The pipeline combines **image analysis**, **deep learning**, and **unsupervised clustering** to map and interpret dynamic cellular behaviors.

### Pipeline at a Glance

```
┌─────────────────────────┐
│  Setup & Data           │  Image renaming, overlay evaluation,
│  Preparation            │  mask export for Cell-ACDC
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Segmentation &         │  Cellpose segmentation, TrackPy tracking,
│  Tracking Optimization  │  parameter optimization & QA vs Imaris
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Imaris ↔ Cell-ACDC     │  Convert Cell-ACDC CSVs to Imaris-compatible
│  Tunnel                 │  Excel workbooks via crosswalk mapping
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Feature Extraction     │  Morphokinetic features, protein expression,
│  (PyBatch)              │  speed, displacement, MSD, shape, intensity
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Filtering by Protein   │  Classify cells by expression levels
│  Expression             │  (Pos/Neg/High) across multiple channels
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  TASC – Time-Series     │  Autoencoder embeddings, ANOVA, PCA,
│  Analysis & Clustering  │  k-means, GMM, Wasserstein distance
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Cellomics-5-C          │  Foundation model forecasting (Chronos,
│  Foundation Model       │  Moirai, TimesFM), curve fitting,
│  Analysis Pipeline      │  embedding + clustering, baseline
│                         │  comparison, ANOVA & dose analysis
└─────────────────────────┘
```

---

## Repository Structure

```
T_IL-Celomics-7-C/
├── Setup & Data Preparation/     # Image renaming, overlay evaluation, mask export
├── segmentation_and_tracking_optimization/
│   ├── segmentation/             # Cellpose parameter optimization
│   ├── tracking/                 # TrackPy-based cell tracking
│   └── validation/               # QA metrics vs Imaris (R², RMSE, IoU)
├── imaris_cell_acdc_tunnel/      # Cell-ACDC → Imaris format converter
│   ├── optimization/             # Active development (2-pass pipeline)
│   └── active_tunnel/            # Production-ready version (planned)
├── pybatch/                      # Batch motility feature extraction (Streamlit GUI)
├── filtering_by_protein_expression/  # Multi-channel protein expression filtering
├── TASC/                         # Time-Series Analysis & Clustering
│   ├── TASC_streamlit_app/       # Streamlit GUI for analysis & visualization
│   └── TASC2.4/                  # Jupyter notebook version
└── T_IL-Celomics-5-C/            # Foundation-model analysis pipeline
    ├── T_IL-Cellomics-5-C_streamlit/ # Streamlit GUI for full pipeline
    │   ├── app.py                # Main Streamlit app (13-tab interface)
    │   ├── many-model-forecasting/ # Adapted MMF framework
    │   └── ...                   # 20+ pipeline scripts
    └── T_IL-Cellomics-5-C/       # Original scripts & data outputs
```

---

## Components

### 1. Setup & Data Preparation

Tools to prepare microscopy data for the Cell-ACDC analysis pipeline.

| Tool | Description |
|------|-------------|
| **renameACDC.py** | Renames time-lapse images from Tsarfaty Lab microscope format to Cell-ACDC-compatible directory structure. Sorts frames by embedded timestamps. |
| **compute_overlay_batch.py** | Compares Imaris (reference) vs Cell-ACDC (predicted) segmentation masks at the pixel level. Computes IoU, Dice coefficient, and intersection/union per frame. |
| **export_cellacdc_masks_per_frame.py** | Exports Cell-ACDC `segm.npz` files to individual per-frame TIFF images. Supports multiple array layouts (YX, TYX, ZYX, TZYX) with Y-shift correction. |

### 2. Segmentation & Tracking Optimization

Focuses on **reducing miss-tracked cells** between Cell-ACDC and Imaris results.

- **Segmentation** — Optimizes [Cellpose](https://github.com/MouseLand/cellpose) parameters for fluorescence microscopy images
- **Tracking** — Cell tracking via [TrackPy](https://github.com/soft-matter/trackpy) with tunable linking parameters
- **Validation** — Quantitative comparison against Imaris using R², RMSE, MAE, and track overlap metrics

### 3. Imaris ↔ Cell-ACDC Tunnel

A two-pass pipeline that converts Cell-ACDC CSV outputs into Imaris-compatible Excel workbooks:

- **Pass 1** — Direct column mapping + formula evaluation (supports `COL("...")`, `sqrt`, `log`, etc.)
- **Pass 2** — Derived tracking and summary metrics
- Uses a **crosswalk CSV** to define mappings between Cell-ACDC and Imaris column names

### 4. PyBatch — Batch Feature Extraction

A **Streamlit-based GUI** for batch processing of cell motility data.

**Capabilities:**
- Batch processing of Imaris `.xls` files and Incucyte data
- Calculates speed, displacement, acceleration, MSD, shape descriptors, and intensity metrics
- Exports results to Excel, HTML, and PDF
- Session state management for iterative analysis

**Quick Start:**
```bash
conda create -n pybatch python=3.8
conda activate pybatch
pip install -r pybatch/requirements.txt
cd pybatch
pybatch.bat                        # or: streamlit run pybatch.py
```

> **Note:** Requires Windows with a local Microsoft Excel installation (`pywin32`).

### 5. Filtering by Protein Expression

A **Tkinter GUI** (`GUI_channel.py`) that orchestrates a 3-step filtering pipeline:

1. **Intensity segmentation** per fluorescence channel (`code_imaris_step1.py`)
2. **Normalization & combination** across channels (`Categorized_step2.py`)
3. **Category filtering** — Positive / Negative / High across 1–3 channels (`filter_data.py`)

Produces filtered Excel files for downstream analysis (e.g., `PP`, `PN`, `PPP`, `PPN` combinations).

### 6. TASC — Time-Series Analysis & Clustering

The core analysis module for **representation learning** of dynamic cell behavior.

**Features:**
- **ANOVA** — Statistical testing across experimental groups
- **PCA** — Dimensionality reduction and visualization
- **Autoencoders** — Time-series embedding for motility profiles
- **Clustering** — K-means, GMM, DBSCAN on learned representations
- **Wasserstein Distance** — Heatmaps for distribution comparison between conditions
- **Hierarchical Clustering** — Dendrogram-based grouping
- **KDE** — Kernel density estimation plots
- Publication-quality figure export

**Quick Start:**
```bash
conda create -n tasc-env python=3.8
conda activate tasc-env
pip install -r TASC/TASC_streamlit_app/requirements.txt
cd TASC/TASC_streamlit_app
TASC.bat                           # or: streamlit run TASC_GUI.py
```

### 7. Cellomics-5-C — Foundation Model Analysis Pipeline

The most advanced component of the project: a **GPU-accelerated analysis pipeline** that uses **foundation time-series models** (transformers) to embed, forecast, and cluster single-cell motility data. It investigates BRCA1-knockout breast cancer cells and compares foundation model representations against traditional curve-fitting approaches.

This module is delivered as a **Streamlit web application** (`app.py`) with a 13-tab interface that orchestrates 14 pipeline stages — no coding required.

#### Pipeline Stages

```
Step 0: Dose Extraction          → Per-cell dose-dependency summary from Excel exports
Step 1: Data Preparation         → Clean/filter raw tracking data (min 25 frames, max gap 5)
Step 2: Feature Selection        → PCA-based selection with Spearman correlation filtering
Step 3: Forecasting (MMF)        → Foundation model evaluation (Chronos, Moirai, TimesFM)
Step 4: Embedding Generation     → Chronos T5 embeddings + UMAP dimensionality reduction
Step 5: Curve Fitting             → 12 mathematical models per cell-feature trajectory
Step 6: Unsupervised Clustering  → K-Means with silhouette sweep (k=2–10), dose analysis
Step 7: ANOVA                    → One-way ANOVA per feature across clusters
Step 8: Descriptive Statistics   → Mean, Std, SE, 95% CI per cluster
Step 9: Baseline Comparison      → MMF vs LSTM, GRU, DLinear, Autoformer baselines
```

Steps 6–8 run in **two parallel branches**:
- **Embedding-only** — clusters cells using foundation model embeddings alone
- **Embedding + Fitting** — clusters using both embeddings and mathematical curve-fit parameters

#### Foundation Models (16 total)

| Family | Models | Source |
|--------|--------|--------|
| **Chronos T5** | Tiny, Mini, Small, Base, Large | Amazon |
| **Chronos Bolt** | Tiny, Mini, Small, Base | Amazon |
| **Moirai** | Small, Base, Large, MoE-Small, MoE-Base | IBM |
| **TimesFM** | 1.0-200m, 2.0-500m | Google |

For each of the 49 morphokinetic features, the pipeline evaluates all models, selects the best by RMSE on the last 5 timepoints, and uses the winner's internal representation as the cell embedding.

#### Curve Fitting — 12 Mathematical Models

| Model | Formula | Parameters |
|-------|---------|------------|
| Linear | $y = ax + b$ | 2 |
| Quadratic | $y = ax^2 + bx + c$ | 3 |
| Cubic | $y = ax^3 + bx^2 + cx + d$ | 4 |
| Exponential Growth | $y = a \cdot e^{bx}$ | 2 |
| Exponential Decay | $y = a \cdot e^{-bx} + c$ | 3 |
| Logarithmic | $y = a \cdot \ln(x) + b$ | 2 |
| Power Law | $y = a \cdot x^b$ | 2 |
| Sigmoid / Logistic | $y = \frac{L}{1 + e^{-k(x - x_0)}}$ | 3 |
| Sine | $y = a \cdot \sin(bx + c) + d$ | 4 |
| Gompertz | $y = a \cdot e^{-b \cdot e^{-cx}}$ | 3 |
| Hill Function | $y = \frac{V_{max} \cdot x^n}{K^n + x^n}$ | 3 |
| Saturating Exponential | $y = a(1 - e^{-bx}) + c$ | 3 |

Only statistically significant fits (p < 0.05) are retained. The best model's parameters form a compact vector representation of each cell-feature trajectory.

#### Baseline Comparison Ladder

| Model | Architecture | Purpose |
|-------|-------------|---------|
| SimpleLSTM | Single-layer LSTM (hidden=64) | Classic RNN baseline |
| SimpleGRU | Single-layer GRU (hidden=64) | Classic RNN baseline |
| SimpleDLinear | Linear trend/residual decomposition | Strong linear baseline |
| SimpleAutoformer | FFT auto-correlation + series decomposition | Lightweight transformer baseline |

Baselines are trained per-cell on the same data and predict the last 5 timepoints, matching the MMF evaluation protocol for a fair comparison.

#### Classification Comparison

A Random Forest classifier is trained to predict cluster assignments using:
1. **Embedding-only** vectors
2. **Embedding + Fitting** concatenated vectors

Performance is evaluated via accuracy, confusion matrices, and per-cluster precision/recall/F1 — quantifying the added value of curve-fitting parameters.

#### Two-Way ANOVA & Cross-Project Validation

- Compares current project's clusters against a previous project's static (mean-based) clusters
- Two-way ANOVA (Method × Cluster) with Tukey HSD post-hoc tests and FDR correction
- Z-score heatmaps split by morphological vs. kinetic feature categories

#### Quick Start

```bash
# Requires a Linux server with GPU(s), Java 17, Python 3.11
python3.11 -m venv ~/mmf_gpu_env
source ~/mmf_gpu_env/bin/activate
pip install -r T_IL-Celomics-5-C/T_IL-Cellomics-5-C_streamlit/streamlit_requirements.txt

cd T_IL-Celomics-5-C/T_IL-Cellomics-5-C_streamlit
streamlit run app.py
# Open http://localhost:8501 in your browser
```

> **Requirements:** Linux with CUDA GPU(s), Java 17 (for PySpark), Python 3.11. Multi-GPU parallelism is supported for forecasting, embedding, and baseline comparison stages.

#### Output Structure

| Stage | Output Location | Key Files |
|-------|----------------|-----------|
| Dose Extraction | `cell_data/` | `dose_dependency_summary_all_wells.csv` |
| Data Preparation | `cell_data/` | `raw_all_cells.csv` |
| Feature Selection | `cell_data/` + `figures/` | `selected_features.txt`, `normalized_all_cells.csv` |
| Forecasting | `forecasting/` | `best_model_per_feature.json`, `best_t5_model_per_feature.json` |
| Embedding | `embeddings/` | `Embedding{ID}.json` |
| Curve Fitting | `fitting/` + `figures/` | `fitting_all_models.csv`, `fitting_best_model_log_scaled.json` |
| Clustering | `clustering/` | `cluster_assignments_k*.csv`, PCA scatter plots |
| ANOVA | `clustering/` | `ANOVA - OneWay_k*.xlsx` |
| Descriptive Stats | `clustering/` | `Descriptive_Table_By_Cluster_UniqueCells_k*.xlsx` |
| Emb+Fit branch | `fitting/` | `embedding_fitting_*.csv`, `embedding_fitting_*.xlsx` |
| Baseline Comparison | `baseline/` | `baseline_comparison.csv`, `baseline_comparison_summary.txt` |

---

## Requirements

- **OS:** Windows (required for PyBatch Excel integration; other modules are cross-platform). The Cellomics-5-C pipeline requires **Linux with GPU(s)**.
- **Python:** 3.8+ (3.11 for Cellomics-5-C)
- **Conda** (recommended for environment management)
- **Microsoft Excel** (required by PyBatch via `pywin32`)

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive GUI for PyBatch, TASC, and Cellomics-5-C |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Clustering, PCA, preprocessing |
| `scipy` | Statistical analysis |
| `matplotlib` / `seaborn` | Visualization |
| `cellpose` | Deep learning cell segmentation |
| `trackpy` | Particle tracking |
| `pywin32` | Excel automation (Windows) |
| `chronos` | Amazon Chronos foundation models (Cellomics-5-C) |
| `torch` | PyTorch deep learning backend (Cellomics-5-C) |
| `pyspark` | Distributed forecasting via MMF (Cellomics-5-C) |
| `umap-learn` | UMAP dimensionality reduction (Cellomics-5-C) |

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/T-IL-Celomics/T_IL-Celomics-7-C.git
   cd T_IL-Celomics-7-C
   ```

2. **Prepare your data** using the tools in `Setup & Data Preparation/`:
   - Rename raw microscopy images with `renameACDC.py`
   - Run Cell-ACDC segmentation and tracking

3. **Optimize segmentation & tracking** parameters in `segmentation_and_tracking_optimization/`

4. **Convert Cell-ACDC outputs** to Imaris format via `imaris_cell_acdc_tunnel/`

5. **Extract features** using PyBatch (`pybatch/`)

6. **Filter by protein expression** if multi-channel data is available (`filtering_by_protein_expression/`)

7. **Analyze & cluster** time-series motility data with TASC (`TASC/TASC_streamlit_app/`)

8. **Run the full foundation-model pipeline** with Cellomics-5-C (`T_IL-Celomics-5-C/T_IL-Cellomics-5-C_streamlit/`):
   - Foundation model forecasting & embedding
   - Curve fitting with 12 mathematical models
   - Unsupervised clustering (embedding-only and embedding+fitting branches)
   - Statistical validation (ANOVA, descriptive stats, baseline comparison)

---

## Project Context

This work was developed in the **Tsarfaty Lab** as part of research into cancer cell motility characterization. The goal is to identify aggressive tumor subpopulations through their dynamic movement patterns, with potential applications in:

- Cancer prognosis and treatment response prediction
- Drug efficacy assessment via motility-based biomarkers
- Automated high-throughput cell behavior profiling

---

## License

See individual module directories for specific licensing information.
