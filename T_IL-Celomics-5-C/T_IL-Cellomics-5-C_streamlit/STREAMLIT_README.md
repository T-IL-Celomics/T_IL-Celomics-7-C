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

---

## 1. What Does This Pipeline Do?

This pipeline analyzes the movement and shape (morphokinetics) of cells over time. It takes raw cell tracking data and runs it through a series of analysis steps:

```
Step 1: Data Preparation        → Clean and filter the raw tracking data
Step 2: Feature Selection       → Pick the most informative cell measurements
Step 3: Forecasting             → AI models predict cell behavior to find patterns
Step 4: Embedding               → Convert time-series data into compact representations
Step 5: Curve Fitting           → Fit mathematical models to each cell's trajectory
Step 6: Clustering              → Group similar cells together
Step 7: ANOVA                   → Test if clusters are statistically different
Step 8: Descriptive Statistics  → Summarize each cluster's characteristics
```

Steps 6–8 run in **two parallel branches**:
- **Embedding-only branch** — clusters cells using only the AI embeddings
- **Embedding + Fitting branch** — clusters cells using both AI embeddings and mathematical curve fits (richer information)

---

## 2. What You Need Before Starting

| Requirement | Details |
|-------------|---------|
| **Remote Linux server** | With GPU(s) — needed for AI models in Steps 3 and 4 |
| **Python 3.11** | Already installed on most servers |
| **CUDA** | GPU drivers for PyTorch — check with `nvidia-smi` |
| **Java 17** | Needed for PySpark (used internally by the forecasting framework) |
| **Input data file** | Your pybatch output — either a `.csv` or `.xlsx` file |

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

If you have dose-response data, enter the path to the dose summary CSV. Leave blank if not applicable.

### 6e. Sidebar — Enable/Disable Steps

On the **left sidebar**, you'll see checkboxes for each pipeline step. All are enabled by default. Uncheck any step you want to skip.

---

## 7. Running the Pipeline Steps

The main area shows **tabs** — one for each step. Work through them **from left to right**.

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
4. Output: `selected_features.txt` and figures showing which features were selected

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
   - `best_model_per_feature.json` — best model overall (any family)
   - `best_chronos_model_per_feature.json` — best Chronos model (T5 or Bolt)
   - `best_t5_model_per_feature.json` — best T5 model (used by the Embedding step)

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
3. Click **📐 Run Curve Fitting** (button label: "📐 Fit")
4. Output: Multiple CSV and JSON files in the project root, plus trajectory plots in `figures/`

### Step 6: Clustering

Groups cells into clusters based on their embeddings (and optionally fitting parameters).

There are **two clustering tabs**:

- **🔮 Clustering** — uses embeddings only
- **🔗 Emb+Fit Clustering** — uses embeddings combined with curve fitting parameters (richer analysis)

For each:
1. Click the tab
2. (Optional) Adjust k-range (number of clusters to try), PCA components, etc.
3. Click the Run button
4. Output: Cluster assignments, PCA scatter plots colored by cluster and treatment, dose tables

### Step 7: ANOVA

Tests whether the clusters are statistically different for each feature.

- **📊 ANOVA** tab — for embedding-only clusters
- **📊 Emb+Fit ANOVA** tab — for combined clusters

Output: Excel files with p-values highlighted in red where significant.

### Step 8: Descriptive Statistics

Summarizes each cluster with mean, standard deviation, standard error, and confidence intervals.

- **📋 Descriptive Stats** tab — for embedding-only clusters
- **📋 Emb+Fit Descriptive** tab — for combined clusters

Output: Excel files with descriptive statistics per cluster.

### Run All Stages

Above the tabs, there's a **🚀 Run All Enabled Stages** button that runs all enabled steps in sequence. It has a **⏭️ Skip stages whose outputs already exist** checkbox (on by default) — if outputs from a step already exist, that step is skipped automatically. There's also a **🛑 Stop** button to cancel during execution.

---

## 8. Understanding the Results

### Where are the output files?

All outputs are saved in the project directory on the server:

| Step | Output Files | Location |
|------|-------------|----------|
| Data Prep | `raw_all_cells.csv` | `cell_data/` |
| Feature Selection | `selected_features.txt`, heatmaps/plots | Root + `figures/` |
| Forecasting | `best_model_per_feature.json`, `best_t5_model_per_feature.json` | Root |
| Embedding | `Embedding{ID}.json` | `embeddings/` |
| Curve Fitting | `fitting_*.csv`, `fitting_*.json`, trajectory plots | Root + `figures/` |
| Clustering | Cluster CSVs, PCA plots, dose tables | `clustering/` |
| Emb+Fit Clustering | Combined cluster CSVs, PCA plots, descriptive tables | `fitting/` |
| ANOVA | `ANOVA - OneWay.xlsx` | Root |
| Emb+Fit ANOVA | `embedding_fitting_ANOVA - OneWay.xlsx` | Root |
| Descriptive Stats | `Descriptive_Table_By_Cluster_UniqueCells.xlsx` | Root |
| Emb+Fit Descriptive | `embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells.xlsx` | Root |

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

### The app shows "Step X requires outputs from Step Y"

- Steps depend on each other — you must run them in order (1 → 2 → 3 → 4 → 5 → 6 → 7 → 8)
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
#    c. Click through tabs 1–8, or use "Run All"
#    d. Download results
```
