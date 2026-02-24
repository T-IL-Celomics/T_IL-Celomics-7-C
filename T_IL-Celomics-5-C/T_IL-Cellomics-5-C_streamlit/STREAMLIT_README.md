# Cellomics-5-C Analysis Pipeline — User Guide

A step-by-step guide for running the Cellomics-5-C analysis pipeline through a graphical interface in your browser. **No coding knowledge is required** — you just click buttons and watch the results.

---

## Table of Contents

1. [What Does This Pipeline Do?](#1-what-does-this-pipeline-do)
2. [What You Need Before Starting](#2-what-you-need-before-starting)
3. [Setting Up the Environment (One-Time Setup)](#3-setting-up-the-environment-one-time-setup)
4. [Launching the App](#4-launching-the-app)
5. [Opening the App in Your Browser](#5-opening-the-app-in-your-browser)
6. [Configuring the Pipeline](#6-configuring-the-pipeline)
7. [Running the Pipeline Steps](#7-running-the-pipeline-steps)
8. [Understanding the Results](#8-understanding-the-results)
9. [Keeping the App Running After You Disconnect](#9-keeping-the-app-running-after-you-disconnect)
10. [Troubleshooting](#10-troubleshooting)
11. [Changelog](#11-changelog)

---

## 1. What Does This Pipeline Do?

This pipeline analyzes the movement and shape (morphokinetics) of cells over time. It takes raw cell tracking data and runs it through a series of analysis steps:

```
Step 0: Dose Extraction         → Build per-cell dose-dependency summary from normalised Excel exports
Step 1: Data Preparation        → Clean and filter the raw tracking data
Step 2: Feature Selection       → Pick the most informative cell measurements
Step 3: Forecasting             → AI models predict cell behavior to find patterns
Step 4: Embedding               → Convert time-series data into compact representations
Step 5: Curve Fitting           → Fit mathematical models to each cell's trajectory
Step 6: Clustering              → Group similar cells together (with dose-category analysis)
Step 7: ANOVA                   → Test if clusters are statistically different
Step 8: Descriptive Statistics  → Summarize each cluster's characteristics
Step 9: Baseline Comparison     → Compare MMF results against 4 baselines (LSTM, GRU, DLinear, Autoformer)
```

Steps 6–8 run in **two parallel branches**:
- **Embedding-only branch** — clusters cells using only the AI embeddings
- **Embedding + Fitting branch** — clusters cells using both AI embeddings and mathematical curve fits (richer information)

The **Run All** button executes steps in the correct dependency order: Dose Extraction runs first (before clustering), followed by Data Preparation through Descriptive Statistics, and finally the Baseline Comparison.

---

## 2. What You Need Before Starting

| Requirement | Details |
|-------------|---------|
| **Remote Linux server** | With GPU(s) — needed for AI models in Steps 3 and 4 |
| **Python 3.11** | Already installed on most servers |
| **CUDA** | GPU drivers for PyTorch — check with `nvidia-smi` |
| **Java 17** | Needed for PySpark (used internally by the forecasting framework) |
| **Input data file** | Your pybatch output — either a `.csv` or `.xlsx` file |
| **Dose Excel files** (optional) | Normalised Excel exports (e.g. `Gab_Normalized_Combined_*.xlsx`) for dose analysis |

### Check GPU availability

Open a terminal on the server and type:
```bash
nvidia-smi
```
You should see your GPU(s) listed. If you get "command not found", CUDA is not installed.

### Check Java

```bash
java -version
```
If not installed:
```bash
sudo apt update && sudo apt install -y openjdk-17-jdk
```

---

## 3. Setting Up the Environment (One-Time Setup)

You only need to do this **once** on a new server.

### Step 3a: Copy the project folder

Copy the entire `T_IL-Cellomics-5-C_streamlit` folder to the server (using `scp`, FileZilla, or any file transfer tool).

### Step 3b: Fix Windows line endings

If the files were created or edited on Windows, run this to prevent hidden errors:
```bash
cd /path/to/T_IL-Cellomics-5-C_streamlit
sed -i 's/\r$//' *.py run_4gpus.sh
```

### Step 3c: Create the Python environment

**Option A — Creating a new virtual environment (recommended):**
```bash
python3.11 -m venv ~/mmf_gpu_env
source ~/mmf_gpu_env/bin/activate
pip install -r streamlit_requirements.txt
```

**Option B — Using an existing environment:**
```bash
source ~/mmf_gpu_env/bin/activate
pip install -r streamlit_requirements.txt
```

> **Note:** It is normal for the installation to take 10–20 minutes. If you see red warning text about version conflicts, that is usually fine — as long as the install finishes without a final error.

### Step 3d: Verify the setup

```bash
source ~/mmf_gpu_env/bin/activate
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
python -c "import chronos; print('Chronos OK')"
```

All three should print without errors. The first one should say `CUDA available: True`.

---

## 4. Launching the App

Every time you want to use the pipeline:

```bash
source ~/mmf_gpu_env/bin/activate
cd /path/to/T_IL-Cellomics-5-C_streamlit
streamlit run app.py
```

You will see output like:
```
  You can now view your Streamlit app in your browser.
  Local URL:  http://localhost:8501
  Network URL: http://10.0.0.5:8501
```

**Do not close this terminal window** — the app stops when you close it (unless you use tmux/screen, see [Section 9](#9-keeping-the-app-running-after-you-disconnect)).

---

## 5. Opening the App in Your Browser

Since the server usually doesn't have a monitor, you need to access it from your personal computer.

### Method A: VS Code Remote SSH (Easiest)

1. Install the **Remote-SSH** extension in VS Code on your PC
2. Connect to the server via Remote-SSH
3. Open the project folder
4. Open a terminal inside VS Code and run `streamlit run app.py`
5. VS Code will show a notification — click **"Open in Browser"**
6. The app opens at `http://localhost:8501`

### Method B: SSH Tunnel

If you're not using VS Code:

**On your PC** (PowerShell, CMD, or Mac Terminal):
```bash
ssh -L 8501:localhost:8501 your_username@server-address
```

This opens an SSH session. In that session, start the app:
```bash
source ~/mmf_gpu_env/bin/activate
cd /path/to/T_IL-Cellomics-5-C_streamlit
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Method C: Direct Access (if on same network)

```bash
# On the server:
streamlit run app.py --server.address 0.0.0.0
```

Then open `http://<server-ip>:8501` in your browser.

---

## 6. Configuring the Pipeline

When the app loads, you'll see the **Pipeline Configuration** section at the top.

### 6a. Experiment ID

Type your experiment number (e.g. `008`). This controls the naming of output files like `Embedding008.json`.

### 6b. Path to your data file

Enter the **full path** to your input data file on the server. For example:
```
/home/username/T_IL-Cellomics-5-C_streamlit/cell_data/MergedAndFilteredExperiment008.csv
```

This can be a `.csv` or `.xlsx` file. A green checkmark ✅ will appear if the file is found.

### 6c. Treatment labels

Enter your treatment group names, separated by commas. For example:
```
CON0, BRCACON1, BRCACON2, BRCAOLAPARIB, BRCATALAZOPARIB
```

These labels are used for grouping cells in the clustering and statistical analysis steps.

### 6d. Dose CSV (optional)

If you have dose-response data, enter the path to the dose summary CSV. Leave blank to auto-detect in the `cell_data/` subdirectory. This CSV is generated by Step 0 (Dose Extraction).

### 6e. Control channel name

Enter the 4-letter channel code used as the control channel (default: `NNIR`). This channel is **excluded from dose-category labels** so that clustering only differentiates by treatment channels, not the control.

For example, if the experiment ID is `AM001100425CHR2D02293TGABYNNIRNOCONNN0NNN0WH00`, the channels are GABY (channel 1) and NNIR (channel 2). With control channel set to `NNIR`, the DoseLabel will only include GABY's category (e.g. `Cha1:High`) and skip NNIR's.

### 6f. Sidebar — Enable/Disable Steps

On the **left sidebar**, you'll see checkboxes for each pipeline step (including Dose Extraction). All are enabled by default. Uncheck any step you want to skip.

---

## 7. Running the Pipeline Steps

The main area shows **13 tabs** — one for each step. Work through them **from left to right**.

### Step 0: Dose Extraction

Builds a per-cell dose-dependency summary from normalised Excel exports. This step must run **before** clustering if you want dose-category analysis.

1. Click the **💊 Dose Summary** tab
2. Expand **⚙️ Dose Summary Parameters** to configure:
   - **Directory containing dose Excel files** — full path to the folder with your normalised Excel exports (e.g. `/data/exports/`). Leave blank to search in the current working directory
   - **Excel glob pattern** (default: `Gab_Normalized_Combined_*.xlsx`) — filename pattern resolved inside the directory above
   - **Sheet name** (default: `Area`) — the sheet to read in each workbook
   - **Filename prefix** (default: `Gab_Normalized_Combined_`) — prefix stripped from filename to derive well name
   - **Well-map JSON path** (optional) — path to a JSON file that maps short well names (like `B2`, `C3`) to full experiment IDs. Example contents:
     ```json
     {
       "B2": "AM001100425CHR2B02293TNNIRNOCONNN0NNN0NNN0WH00",
       "C2": "AM001100425CHR2C02293TMETRNNIRNOCONNN0NNN0WH00"
     }
     ```
     Leave blank to use the built-in default mapping.
3. Click **💊 Run Dose Extraction**
4. Output: `cell_data/dose_dependency_summary_all_wells.csv`

### Step 1: Data Preparation

1. Click the **📁 Data Preparation** tab
2. (Optional) Expand **⚙️ Parameters** to change:
   - **Min frames per cell** (default: 25) — cells with fewer time points are removed
   - **Max gap** (default: 5) — largest allowed gap in a cell's time series
3. Click **🚀 Run Data Preparation**
4. Wait for it to finish — you'll see a green "Completed" message
5. Output: `raw_all_cells.csv` in the `cell_data/` folder

### Step 2: Feature Selection

1. Click the **🎯 Feature Selection** tab
2. (Optional) Adjust parameters like PCA variance threshold, correlation cutoff, etc.
3. Click **🎯 Run Feature Selection** button
4. Output: `cell_data/selected_features.txt` and figures showing which features were selected

### Step 3: Forecasting

This step uses AI models to evaluate how well they can predict cell behavior. It **does not** change your data — it only determines which AI model works best for each feature.

1. Click the **📈 Forecasting** tab
2. Choose **Single GPU** or **Multi-GPU Parallel** mode:
   - Single GPU: Simpler, runs on one GPU
   - Multi-GPU: Faster with multiple GPUs — choose how many GPUs to use
3. (Optional) Change **Max cells** (default: 500). This subsamples cells for faster evaluation. Use 0 for all cells (slower but slightly more precise model ranking).
4. Click **📈 Run Forecasting**
5. ⏳ This is the **longest step** — can take 1–4 hours depending on data size and GPU count
6. Output: Three JSON files mapping each feature to its best-performing model:
   - `forecasting/best_model_per_feature.json` — best model overall (any family)
   - `forecasting/best_chronos_model_per_feature.json` — best Chronos model (T5 or Bolt)
   - `forecasting/best_t5_model_per_feature.json` — best T5 model (used by the Embedding step)

### Step 4: Embedding

This step uses the best T5 models (from Step 3) to convert each cell's time-series data into a compact numerical representation (embedding), then reduces dimensions with UMAP.

1. Click the **🧬 Embedding** tab
2. Choose **Single GPU** or **Multi-GPU Parallel** mode
3. (Optional) Change **UMAP dimensions** (default: 3)
4. Click the Run button
5. Output: `Embedding008.json` (or your experiment ID) in the `embeddings/` folder

### Step 5: Curve Fitting

Fits 12 mathematical models (linear, exponential, logistic, etc.) to each cell's trajectory for each feature.

1. Click the **📐 Curve Fitting** tab
2. (Optional) Adjust **maxfev** (max iterations for curve fitting) and **p-value threshold**
3. Click **📐 Run Curve Fitting**
4. Output: Multiple CSV and JSON files in the project root, plus trajectory plots in `figures/`

### Step 6: Clustering

Groups cells into clusters based on their embeddings (and optionally fitting parameters). Dose-category analysis is included automatically if a dose CSV exists.

There are **two clustering tabs**:

- **🔮 Clustering** — uses embeddings only
- **🔗 Emb+Fit Clustering** — uses embeddings combined with curve fitting parameters (richer analysis)

For each:
1. Click the tab
2. (Optional) Adjust k-range (number of clusters to try), PCA components, etc.
3. Click the Run button
4. Output: Cluster assignments, PCA scatter plots (by cluster, treatment, and dose category), contingency tables, heatmaps

**Dose-category plots:** When dose data is available, the clustering step generates PCA scatter plots and contingency tables split by dose category. The control channel (set in Pipeline Configuration) is automatically excluded from these dose labels so plots only show treatment-relevant channel categories.

### Step 7: ANOVA

Tests whether the clusters are statistically different for each feature.

- **📊 ANOVA** tab — for embedding-only clusters
- **📊 Emb+Fit ANOVA** tab — for combined clusters

Output: Excel files with p-values highlighted in blue where significant.

### Step 8: Descriptive Statistics

Summarizes each cluster with mean, standard deviation, standard error, and confidence intervals.

- **📋 Descriptive Stats** tab — for embedding-only clusters
- **📋 Emb+Fit Descriptive** tab — for combined clusters

Output: Excel files with descriptive statistics per cluster.

### Step 9: Baseline Comparison

Compares the best MMF transformer model against four baseline models (SimpleLSTM, SimpleGRU, SimpleDLinear, SimpleAutoformer) to quantify how much better the MMF models are. The baselines are trained per-cell on the same data — they are **not** part of the MMF forecasting pipeline, only used for comparison.

1. Click the **🧠 Baselines vs MMF** tab
2. Choose **Number of GPUs** (1–4) — features are split across GPUs for parallel training
3. Click **🧠 Run Baseline Comparison**
4. ⏳ This step trains all four baseline models per cell per feature — speed depends on GPU count and cell count
5. Output:
   - `baseline/baseline_comparison.csv` — per-feature comparison table with MSE, MAE, RMSE for both MMF and all baselines, plus percentage improvement columns
   - `baseline/baseline_comparison.json` — same data as JSON
   - `baseline/baseline_comparison_summary.txt` — headline summary (e.g. "+21% lower MSE, +17% lower MAE vs. SimpleLSTM")
   - `results/<feature>/SimpleLSTM_metrics.csv` — per-cell LSTM metrics
   - `results/<feature>/SimpleGRU_metrics.csv` — per-cell GRU metrics
   - `results/<feature>/SimpleDLinear_metrics.csv` — per-cell DLinear metrics
   - `results/<feature>/SimpleAutoformer_metrics.csv` — per-cell Autoformer metrics

**Cell matching:** The comparison script automatically discovers which cells the best MMF model was evaluated on (from the existing metrics CSVs) and trains on exactly those cells. This guarantees a fair apples-to-apples comparison regardless of MAX_CELLS settings. If no MMF results exist yet, it falls back to MAX_CELLS sub-sampling with seed=42.

**Multi-GPU mode:** When using 2–4 GPUs, features are split round-robin across GPUs. Each GPU trains all four baselines on its shard of features. Per-GPU logs are saved to `logs/baseline_gpu{i}.txt` and shard results are automatically merged at the end.

**Baseline model ladder:**
| Model | Architecture | Purpose |
|---|---|---|
| SimpleLSTM | Single-layer LSTM (hidden_size=64) | Classic RNN baseline |
| SimpleGRU | Single-layer GRU (hidden_size=64) | Classic RNN baseline |
| SimpleDLinear | Linear trend/residual decomposition | Strong linear baseline |
| SimpleAutoformer | FFT auto-correlation + series decomposition | Lightweight transformer baseline |

### Run All Stages

Above the tabs, there's a **🚀 Run All Enabled Stages** button that runs all enabled steps in sequence. The execution order is: Dose Extraction → Data Preparation → Feature Selection → Forecasting → Embedding → Curve Fitting → Clustering → ANOVA → Descriptive Stats (then the Emb+Fit branch).

It has a **⏭️ Skip stages whose outputs already exist** checkbox (on by default) — if outputs from a step already exist, that step is skipped automatically. There's also a **🛑 Stop** button to cancel during execution.

The execution order is: Dose Extraction → Data Preparation → Feature Selection → Forecasting → Embedding → Curve Fitting → Clustering → ANOVA → Descriptive Stats (both branches) → Baseline Comparison.

---

## 8. Understanding the Results

### Where are the output files?

All outputs are saved in the project directory on the server:

| Step | Output Files | Location |
|------|-------------|----------|
| Dose Extraction | `dose_dependency_summary_all_wells.csv` | `cell_data/` |
| Data Prep | `raw_all_cells.csv` | `cell_data/` |
| Feature Selection | `selected_features.txt`, `normalized_all_cells.csv`, heatmaps/plots | `cell_data/` + `figures/` |
| Forecasting | `best_model_per_feature.json`, `best_t5_model_per_feature.json`, etc. | `forecasting/` |
| Embedding | `Embedding{ID}.json` | `embeddings/` |
| Curve Fitting | `fitting_*.csv`, `fitting_*.json`, trajectory plots | `fitting/` + `figures/` |
| Clustering | Cluster CSVs, PCA plots, dose tables, heatmaps | `clustering/` |
| ANOVA | `ANOVA - OneWay.xlsx` | `clustering/` |
| Descriptive Stats | `Descriptive_Table_By_Cluster_UniqueCells.xlsx` | `clustering/` |
| Emb+Fit Clustering | Combined cluster CSVs, PCA plots, descriptive tables | `fitting/` |
| Emb+Fit ANOVA | `embedding_fitting_ANOVA - OneWay.xlsx` | `fitting/` |
| Emb+Fit Descriptive | `embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells.xlsx` | `fitting/` |
| Baseline Comparison | `baseline_comparison.csv`, `baseline_comparison.json`, `baseline_comparison_summary.txt` | `baseline/` |
| Baseline Per-Cell Metrics | `SimpleLSTM_metrics.csv`, `SimpleGRU_metrics.csv`, `SimpleDLinear_metrics.csv`, `SimpleAutoformer_metrics.csv` | `results/<feature>/` |

### Viewing results in the app

After each step finishes, the app shows:
- **Output images** — plots and figures are displayed directly in the browser
- **Completion status** — green success messages with timing information
- **Script output** — expand the output section to see detailed logs

### Downloading results

To get the files onto your personal computer:
- **VS Code**: Right-click the file in the Explorer sidebar → Download
- **SCP**: `scp username@server:/path/to/file.xlsx ./`
- **FileZilla**: Navigate to the folder and drag files to your PC

---

## 9. Keeping the App Running After You Disconnect

By default, closing your terminal or SSH connection stops the app. To keep it running:

### Using tmux (recommended)

```bash
# Start a named session
tmux new -s streamlit

# Inside tmux, run the app
source ~/mmf_gpu_env/bin/activate
cd /path/to/T_IL-Cellomics-5-C_streamlit
streamlit run app.py --server.headless true

# Detach: press Ctrl+B, then D
# The app keeps running in the background

# Reconnect later:
tmux attach -t streamlit
```

### Using screen

```bash
screen -S streamlit
source ~/mmf_gpu_env/bin/activate
cd /path/to/T_IL-Cellomics-5-C_streamlit
streamlit run app.py --server.headless true
# Detach: press Ctrl+A, then D
# Reconnect: screen -r streamlit
```

### Using nohup

```bash
nohup streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
```

---

## 10. Troubleshooting

### "File not found" errors

- Make sure the path you entered in Pipeline Configuration is the **full path** (starting with `/`)
- Check that the file actually exists: `ls -la /path/to/your/file.csv`

### "No files matched the glob pattern" (Dose Extraction)

- Make sure the **Directory containing dose Excel files** is set to the correct folder path
- Check that the glob pattern matches your file names (e.g. `Gab_Normalized_Combined_*.xlsx`)
- Verify files exist: `ls /path/to/dose/dir/Gab_Normalized_Combined_*.xlsx`

### The app shows "Step X requires outputs from Step Y"

- Steps depend on each other — you must run them in order
- Dose Extraction should run before Clustering if you want dose analysis
- Check the "Pipeline Intermediate Files" expander to see which files exist

### `\r: command not found` or strange errors after copying files from Windows

Run this once to fix line endings:
```bash
sed -i 's/\r$//' *.py run_4gpus.sh
```

### GPU not detected

```bash
# Check GPU is visible
nvidia-smi

# Check PyTorch can see it
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, your PyTorch installation may not match your CUDA version.

### "No module named ..." errors

Make sure you activated the right Python environment:
```bash
source ~/mmf_gpu_env/bin/activate
```

### Forecasting takes too long

- Use **Multi-GPU** mode if you have multiple GPUs
- Reduce **Max cells** (e.g. from 500 to 200) for faster model evaluation
- The model ranking is usually stable even with fewer cells

### Out of GPU memory

- Reduce the number of active models in `my_models_conf.yaml` (remove larger models like `ChronosT5Large`, `MoiraiLarge`)
- Reduce **Max cells** in the Forecasting parameters

### Matplotlib / display errors

Already handled — all scripts run in headless mode. If you still see display errors, set:
```bash
export MPLBACKEND=Agg
```

### The app is slow to respond / shows "Running..."

This is normal during long computations. The app processes data in the background. Do not refresh the page — wait for the step to finish.

### Mixed-type column warnings

If you see `DtypeWarning: Columns have mixed types`, this is handled automatically — the pipeline reads data with `low_memory=False` and filters to numeric columns only for analysis.

---

## Quick Reference: Running the Pipeline Start to Finish

```bash
# 1. Connect to server
ssh username@server

# 2. Activate environment
source ~/mmf_gpu_env/bin/activate

# 3. Go to project folder
cd /path/to/T_IL-Cellomics-5-C_streamlit

# 4. Launch the app
streamlit run app.py

# 5. Open in browser (on your PC)
#    → http://localhost:8501

# 6. In the app:
#    a. Enter your data file path
#    b. Enter treatment labels
#    c. Set control channel name (default: NNIR)
#    d. (Optional) Configure dose extraction directory
#    e. Click through tabs 0–8, or use "Run All"
#    f. Download results
```

---

## Pipeline Files Reference

### Active pipeline scripts

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit GUI — runs all steps |
| `build_dose_summary.py` | Step 0: Dose extraction from Excel exports |
| `make_raw_all_cells_from_pybatch.py` | Step 1: Data preparation |
| `feature_selection.py` | Step 2: Feature selection |
| `TSA_analysis.py` | Step 3: Forecasting (single GPU) |
| `TSA_analysis_4gpu.py` | Step 3: Forecasting (multi-GPU) |
| `Embedding.py` | Step 4: Embedding (single GPU) |
| `Embedding_multi_gpu.py` | Step 4: Embedding (multi-GPU) |
| `fit_cell_trajectory.py` | Step 5: Curve fitting |
| `embedding_unsupervised_clustering.py` | Step 6: Clustering (embedding only) |
| `embedding_fitting_unsupervised_clustering.py` | Step 6b: Clustering (embedding + fitting) |
| `ANOVA.py` | Step 7: ANOVA |
| `embedding_fitting_anova.py` | Step 7b: ANOVA (embedding + fitting) |
| `descriptive_table_by_cluster.py` | Step 8: Descriptive statistics |
| `embedding_fitting_descriptive_table.py` | Step 8b: Descriptive statistics (embedding + fitting) |
| `baseline_comparison.py` | Step 9: Baseline comparison (single GPU) |
| `run_baseline_multi_gpu.sh` | Step 9: Baseline comparison launcher (1–4 GPUs) |

### Baseline model files

| File | Purpose |
|------|---------|
| `many-model-forecasting/mmf_sa/models/rnnforecast/RNNPipeline.py` | All four baseline model classes (LSTM, GRU, DLinear, Autoformer) |
| `many-model-forecasting/mmf_sa/models/rnnforecast/__init__.py` | Package init |

### Configuration files

| File | Purpose |
|------|---------|
| `my_models_conf.yaml` | AI model configuration for forecasting |
| `well_map.json` | Maps short well names to full experiment IDs (for dose extraction) |
| `streamlit_requirements.txt` | Python package dependencies |
| `run_4gpus.sh` | Shell helper for multi-GPU forecasting |
| `run_baseline_multi_gpu.sh` | Shell helper for multi-GPU baseline comparison |

---

## 11. Changelog

### February 20–21, 2026

#### New: Step 9 — Baseline Comparison

Added a new pipeline stage that trains four baseline models (SimpleLSTM, SimpleGRU, SimpleDLinear, SimpleAutoformer) and compares their forecasting performance against the best MMF transformer model per feature. This provides a graduated comparison ladder: simple RNNs → linear decomposition → lightweight transformer → foundation models.

**New files:**

- **`baseline_comparison.py`** — Standalone comparison script. Reads existing MMF metrics CSVs, trains all four baselines per cell per feature on the same data, computes MSE/MAE/RMSE, and outputs percentage-improvement tables.
- **`run_baseline_multi_gpu.sh`** — Shell launcher for 1–4 GPU parallel execution. Shards features across GPUs, monitors progress, and merges shard results automatically.
- **`many-model-forecasting/mmf_sa/models/rnnforecast/RNNPipeline.py`** — All four baseline model classes: SimpleLSTM, SimpleGRU (single-layer RNNs with hidden_size=64), SimpleDLinear (linear trend/residual decomposition), and SimpleAutoformer (FFT-based auto-correlation + series decomposition). All share per-series z-score normalization, sliding-window training, auto-regressive forecasting, and early stopping. Uses `torch.backends.cudnn.enabled = False` to avoid cuDNN errors on some GPUs (e.g. NVIDIA TITAN Xp).
- **`many-model-forecasting/mmf_sa/models/rnnforecast/__init__.py`** — Package init.

**Modified files:**

- **`app.py`** — Added "🧠 Baselines vs MMF" tab (tab 12) with:
  - GPU selector (1–4 GPUs)
  - Comparison summary display, full data table, percentage-improvement metric cards
  - Per-GPU log viewer for multi-GPU runs
  - Sidebar enable/disable checkbox
  - Integration into "Run All" pipeline as the last stage
- **`many-model-forecasting/mmf_sa/models/models_conf.yaml`** — Added SimpleLSTM, SimpleGRU, SimpleDLinear, and SimpleAutoformer model definitions (not in active_models — used only by the standalone comparison script).

**Key design decisions:**

- Baseline models are **not** part of the MMF forecasting pipeline — they run independently and are only used for comparison.
- The comparison script **auto-discovers cells** from the best MMF model's metrics CSV, ensuring an apples-to-apples comparison regardless of MAX_CELLS settings across different runs.
- Each baseline predicts the **last 5 time points** per cell (matching the MMF prediction_length=5), with the same train/val split as the MMF pipeline.
- cuDNN is disabled at module level in RNNPipeline.py because the RNN models are tiny (single-layer, hidden_size=64) and cuDNN kernel overhead provides no benefit while causing crashes on some GPU architectures.
