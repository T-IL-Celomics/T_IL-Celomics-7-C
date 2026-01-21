# Cellomics-5-C Streamlit Analysis Pipeline

A Streamlit GUI that **runs the existing Python scripts** for analyzing morphokinetic behavior of BRCA1-knockout breast cancer cells.

## 🚀 Quick Start

### On Remote Server (with existing environment)

```bash
# Activate your conda environment
conda activate mmf_gpu_env

# Run the app
streamlit run app.py
```

The app will automatically use all packages from your activated environment.

### Fresh Installation

1. **Install dependencies:**
   ```bash
   pip install -r streamlit_requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

## 🌐 Accessing GUI from Personal Computer (Remote Server)

Since the remote server has no GUI, you need to use **SSH port forwarding** to access the Streamlit app from your local browser.

### Option 1: SSH Tunnel (Recommended)

**Step 1:** On the remote server, run Streamlit:
```bash
conda activate mmf_gpu_env
streamlit run app.py --server.port 8501 --server.headless true
```

**Step 2:** On your **local computer**, open a terminal and create an SSH tunnel:
```bash
# Windows (PowerShell or CMD):
ssh -L 8501:localhost:8501 your_username@ai4vi.tau.ac.il
```

**Step 3:** Open your local browser and go to:
```
http://localhost:8501
```

### Option 2: VS Code Remote SSH

If you're using VS Code with Remote SSH extension:

1. Connect to the remote server via VS Code Remote SSH
2. Run `streamlit run app.py` in the VS Code integrated terminal
3. VS Code will automatically detect the port and show a popup to open in browser
4. Click "Open in Browser" or manually go to `http://localhost:8501`

### Option 3: Direct Network Access

If the server is on your network and firewall allows:
```bash
# On remote server, bind to all interfaces:
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then access from local browser:
```
http://remote-server-ip:8501
```

### Keeping the App Running

To keep Streamlit running after closing SSH:
```bash
# Option A: Use nohup
nohup streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &

# Option B: Use screen
screen -S streamlit
streamlit run app.py --server.port 8501 --server.headless true
# Press Ctrl+A, then D to detach

# Option C: Use tmux
tmux new -s streamlit
streamlit run app.py --server.port 8501 --server.headless true
# Press Ctrl+B, then D to detach
```

## 📋 Pipeline Scripts

The app runs these **existing Python scripts** (not re-implementations):

| Step | Script | Description |
|------|--------|-------------|
| 1 | `make_raw_all_cells_from_pybatch.py` | Data preparation & filtering |
| 2 | `feature_selection.py` | PCA-based feature selection |
| 3 | `TSA_analysis.py` | Time series forecasting (single GPU) |
| 3b | `TSA_analysis_4gpu.py` + `run_4gpus.sh` | **4-GPU parallel forecasting** |
| 4 | `Embedding.py` | Chronos embeddings + UMAP (single GPU) |
| 4b | `Embedding_multi_gpu.py` | **Multi-GPU parallel embedding** |
| 5 | `fit_cell_trajectory.py` | Curve fitting (12 models) |
| 6 | `Unsupervised Clustering - Embedding - K=3.py` | K-Means clustering |
| 7 | `ANOVA.py` | One-way ANOVA by cluster |
| 8 | `descriptive table by cluster.py` | Descriptive statistics |

## 🖥️ Multi-GPU Support

### 4-GPU Forecasting
```bash
# Runs automatically via the app, or manually:
bash run_4gpus.sh
```
- Distributes features across 4 GPUs using `CUDA_VISIBLE_DEVICES` and `SHARD_IDX`
- Logs saved to `logs/gpu0.txt`, `logs/gpu1.txt`, etc.

### Multi-GPU Embedding
```bash
# Via app, or manually:
python Embedding_multi_gpu.py --num_gpus=4 --dim=3 --verbose
```
- Uses Python multiprocessing to distribute across GPUs
- Configurable number of GPUs and UMAP dimensions

## 📁 Expected Data Format

### Required Columns:
- `Experiment` - Experiment identifier
- `Parent` - Cell/track identifier  
- `TimeIndex` - Time point index

### Optional Metadata:
- `Treatment` - Treatment group label
- `dt` - Time interval
- `unique_id` - Created by data prep script

### Feature Columns:
All other numeric columns are treated as features.

## 📊 Output Files

| Script | Outputs |
|--------|---------|
| Data Prep | `raw_all_cells.csv` |
| Feature Selection | `selected_features.txt`, `normalized_all_cells.csv` |
| Forecasting | `best_model_per_feature.json`, `best_chronos_model_per_feature.json` |
| Embedding | `Embedding008.json` |
| Curve Fitting | `fitting_all_models.csv`, `fitting_best_with_nrmse.csv` |
| Clustering | `cluster_assignments_k3.csv`, `pca_kmeans_k3_clusters.png` |
| ANOVA | `ANOVA - OneWay.xlsx` |
| Descriptive | `Descriptive_Table_By_Cluster_UniqueCells.xlsx` |

## 🔧 Configuration Files

- `my_models_conf.yaml` - Forecasting model configuration
- `best_chronos_model_per_feature.json` - Best model per feature (from TSA analysis)

## 📊 Workflow

1. **Data Preparation** → Load and filter pybatch data
2. **Feature Selection** → PCA-based selection with correlation filtering
3. **Forecasting** (optional) → Run foundation models (Chronos, TimesFM, Moirai)
4. **Embedding** (optional) → Generate Chronos embeddings + UMAP
5. **Curve Fitting** → Fit 12 mathematical models to trajectories
6. **Clustering** → K-Means on embeddings with PCA visualization
7. **ANOVA** → Statistical analysis by cluster
8. **Descriptive Stats** → Mean, Std, SE, CI per cluster

## 🛠️ Troubleshooting

### Common Issues:

1. **Script not found**
   - Ensure all Python scripts are in the same directory as `app.py`

2. **GPU not available**
   - For forecasting/embedding, ensure CUDA is properly configured
   - Check with `python -c "import torch; print(torch.cuda.is_available())"`

3. **Script fails**
   - Check the output expander for error messages
   - Verify input files exist (e.g., `raw_all_cells.csv`, `selected_features.txt`)

### Performance Tips:

- Use 4-GPU mode for forecasting to speed up processing
- Multi-GPU embedding distributes features across GPUs automatically
- Each step can be enabled/disabled in the sidebar

## 📦 Key Dependencies

- `streamlit` - Web app framework
- `chronos-forecasting==1.4.1` - Foundation model for time series
- `timesfm==1.2.7` - Google's TimesFM model
- `uni2ts==1.2.0` - Unified time series interface
- `torch==2.4.1+cu118` - PyTorch with CUDA 11.8
- `umap-learn==0.5.5` - Dimensionality reduction
- `scikit-learn==1.4.2` - Machine learning

See `streamlit_requirements.txt` for full list.
