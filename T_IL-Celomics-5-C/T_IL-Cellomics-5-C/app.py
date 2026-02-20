"""
Cellomics-5-C Analysis Pipeline - Streamlit GUI
================================================
A Streamlit application that RUNS THE EXISTING Python scripts.

This app executes the following scripts from the directory:
- make_raw_all_cells_from_pybatch.py: Data preparation
- feature_selection.py: PCA-based feature selection
- TSA_analysis.py: Time series forecasting
- Embedding.py: Chronos embeddings + UMAP
- fit_cell_trajectory.py: Curve fitting with 12 models
- Unsupervised Clustering - Embedding - K=3.py: K-Means clustering
- ANOVA.py: Statistical analysis
- descriptive table by cluster.py: Descriptive statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

# Get the script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)  # Change to script directory for relative paths

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Cellomics-5-C Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== DEFINE AVAILABLE SCRIPTS =====
SCRIPTS = {
    "data_prep": {
        "file": "make_raw_all_cells_from_pybatch.py",
        "name": "Data Preparation",
        "description": "Load pybatch data, create unique_id, filter cells by frame count and gaps",
        "outputs": ["raw_all_cells.csv"]
    },
    "feature_selection": {
        "file": "feature_selection.py",
        "name": "Feature Selection",
        "description": "PCA-based feature selection with correlation filtering and silhouette evaluation",
        "outputs": ["selected_features.txt", "normalized_all_cells.csv"]
    },
    "forecasting": {
        "file": "TSA_analysis.py",
        "name": "Time Series Forecasting",
        "description": "Forecast with Chronos, TimesFM, Moirai foundation models via MMF framework (single GPU)",
        "outputs": ["best_model_per_feature.json", "best_chronos_model_per_feature.json"]
    },
    "forecasting_4gpu": {
        "file": "TSA_analysis_4gpu.py",
        "name": "Time Series Forecasting (4-GPU)",
        "description": "Parallel forecasting across 4 GPUs - shards features across GPUs for faster processing",
        "outputs": ["best_model_per_feature.json", "best_chronos_model_per_feature.json"],
        "multi_gpu": True,
        "shell_script": "run_4gpus.sh"
    },
    "embedding": {
        "file": "Embedding.py",
        "name": "Embedding Generation",
        "description": "Generate embeddings using Chronos models + UMAP dimensionality reduction (single GPU)",
        "outputs": ["Embedding008.json"]
    },
    "embedding_multi_gpu": {
        "file": "Embedding_multi_gpu.py",
        "name": "Embedding Generation (Multi-GPU)",
        "description": "Parallel embedding generation across multiple GPUs with feature sharding",
        "outputs": ["Embedding008.json"],
        "multi_gpu": True
    },
    "fitting": {
        "file": "fit_cell_trajectory.py",
        "name": "Curve Fitting",
        "description": "Fit 12 mathematical models (linear, exponential, logistic, etc.) to cell trajectories",
        "outputs": ["fitting_all_models.csv", "fitting_best_with_nrmse.csv", "fitting_top3_with_nrmse.csv"]
    },
    "clustering": {
        "file": "Unsupervised Clustering - Embedding - K=3.py",
        "name": "Unsupervised Clustering",
        "description": "K-Means clustering on embeddings with PCA visualization",
        "outputs": ["cluster_assignments_k3.csv", "pca_kmeans_k3_clusters.png"]
    },
    "anova": {
        "file": "ANOVA.py",
        "name": "ANOVA Analysis",
        "description": "One-way ANOVA by cluster with multi-level headers",
        "outputs": ["ANOVA - OneWay.xlsx"]
    },
    "descriptive": {
        "file": "descriptive table by cluster.py",
        "name": "Descriptive Statistics",
        "description": "Mean, Std, SE, CI per cluster for all features",
        "outputs": ["Descriptive_Table_By_Cluster_UniqueCells.xlsx"]
    }
}

# ===== HELPER FUNCTION: Run Script =====
def run_script(script_name: str):
    """Run a Python script and capture output."""
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        return -1, "", f"Script not found: {script_path}"
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        timeout=3600  # 1 hour timeout
    )
    return result.returncode, result.stdout, result.stderr

# ===== HELPER: Check if script exists =====
def check_script_exists(script_key: str) -> tuple:
    """Check if script file exists and return status icon."""
    script_info = SCRIPTS[script_key]
    script_path = SCRIPT_DIR / script_info["file"]
    exists = script_path.exists()
    return "✅" if exists else "❌", exists

# ===== HELPER: Check output files =====
def check_outputs(script_key: str) -> list:
    """Check which output files exist."""
    outputs = SCRIPTS[script_key].get("outputs", [])
    existing = []
    for out in outputs:
        if (SCRIPT_DIR / out).exists():
            existing.append(out)
    return existing

# ===== SESSION STATE =====
if 'script_outputs' not in st.session_state:
    st.session_state.script_outputs = {}

# ===== SIDEBAR =====
st.sidebar.title("🔬 Pipeline Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("Enable/Disable Steps")
enable_data_prep = st.sidebar.checkbox("📁 Data Preparation", value=True)
enable_feature_selection = st.sidebar.checkbox("🎯 Feature Selection", value=True)
enable_forecasting = st.sidebar.checkbox("📈 Forecasting", value=False, help="Requires GPU & MMF")
enable_embedding = st.sidebar.checkbox("🧬 Embedding", value=False, help="Requires Chronos")
enable_fitting = st.sidebar.checkbox("📐 Curve Fitting", value=True)
enable_clustering = st.sidebar.checkbox("🔮 Clustering", value=True)
enable_anova = st.sidebar.checkbox("📊 ANOVA", value=True)
enable_descriptive = st.sidebar.checkbox("📋 Descriptive Stats", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Script Status")

for key, info in SCRIPTS.items():
    icon, _ = check_script_exists(key)
    st.sidebar.text(f"{icon} {info['file'][:30]}...")

# ===== MAIN TITLE =====
st.title("🔬 Cellomics-5-C Analysis Pipeline")
st.markdown("*Runs the **existing Python scripts** with interactive controls*")
st.markdown("---")

# ===== TAB LAYOUT =====
tabs = st.tabs([
    "📁 Data Prep", 
    "🎯 Features", 
    "📈 Forecast", 
    "🧬 Embed", 
    "📐 Fit", 
    "🔮 Cluster", 
    "📊 ANOVA",
    "📋 Descriptive"
])

# ===== TAB 1: DATA PREPARATION =====
with tabs[0]:
    st.header("📁 Data Preparation")
    info = SCRIPTS['data_prep']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_data_prep:
        st.warning("⚠️ Disabled. Enable in sidebar.")
    else:
        icon, exists = check_script_exists('data_prep')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            # Show script content
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            # Show existing outputs
            existing_outputs = check_outputs('data_prep')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🚀 Run Data Preparation", type="primary", key="run_data_prep"):
                    with st.spinner(f"Running {info['file']}..."):
                        try:
                            returncode, stdout, stderr = run_script(info['file'])
                            
                            if returncode == 0:
                                st.success("✅ Completed successfully!")
                            else:
                                st.error(f"❌ Failed (code {returncode})")
                            
                            with st.expander("📤 Output", expanded=True):
                                if stdout:
                                    st.text(stdout)
                                if stderr:
                                    st.error(stderr)
                        except subprocess.TimeoutExpired:
                            st.error("⏱️ Script timed out")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            # Preview data
            st.subheader("Data Preview")
            csv_path = SCRIPT_DIR / "raw_all_cells.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, nrows=100)
                st.dataframe(df, use_container_width=True)
                st.metric("Rows (showing 100)", f"{len(df)}")

# ===== TAB 2: FEATURE SELECTION =====
with tabs[1]:
    st.header("🎯 Feature Selection")
    info = SCRIPTS['feature_selection']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_feature_selection:
        st.warning("⚠️ Disabled. Enable in sidebar.")
    else:
        icon, exists = check_script_exists('feature_selection')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            existing_outputs = check_outputs('feature_selection')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                
                # Show selected features if exists
                feat_file = SCRIPT_DIR / "selected_features.txt"
                if feat_file.exists():
                    with open(feat_file, 'r') as f:
                        features = f.read().splitlines()
                    with st.expander(f"📋 Selected Features ({len(features)})"):
                        st.write(features)
            
            if st.button("🎯 Run Feature Selection", type="primary", key="run_features"):
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'])
                        
                        if returncode == 0:
                            st.success("✅ Completed!")
                        else:
                            st.error(f"❌ Failed (code {returncode})")
                        
                        with st.expander("📤 Output", expanded=True):
                            if stdout:
                                st.text(stdout)
                            if stderr:
                                st.error(stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            # Show figures if exist
            fig_path = SCRIPT_DIR / "figures"
            if fig_path.exists():
                st.subheader("Generated Figures")
                for img in fig_path.glob("*.png"):
                    with st.expander(f"📊 {img.name}"):
                        st.image(str(img))

# ===== TAB 3: FORECASTING =====
with tabs[2]:
    st.header("📈 Time Series Forecasting")
    
    if not enable_forecasting:
        st.warning("⚠️ Disabled. Enable in sidebar.")
        st.info("💡 Requires: GPU, MMF framework, Chronos/TimesFM/Moirai models")
    else:
        # Choose between single GPU and 4-GPU version
        gpu_mode = st.radio(
            "Select GPU Mode",
            ["Single GPU", "4-GPU Parallel"],
            horizontal=True,
            help="4-GPU mode shards features across 4 GPUs for faster processing"
        )
        
        if gpu_mode == "Single GPU":
            info = SCRIPTS['forecasting']
            script_key = 'forecasting'
        else:
            info = SCRIPTS['forecasting_4gpu']
            script_key = 'forecasting_4gpu'
        
        st.markdown(f"**Script:** `{info['file']}`")
        st.info(info['description'])
        
        icon, exists = check_script_exists(script_key)
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            # Show shell script for 4-GPU mode
            if gpu_mode == "4-GPU Parallel":
                shell_script = SCRIPT_DIR / "run_4gpus.sh"
                if shell_script.exists():
                    with st.expander("🖥️ View run_4gpus.sh (shell script)"):
                        with open(shell_script, 'r') as f:
                            st.code(f.read(), language='bash')
                
                st.info("💡 `run_4gpus.sh` automatically distributes work across all 4 GPUs in parallel.")
            
            # Show config if exists
            conf_file = SCRIPT_DIR / "my_models_conf.yaml"
            if conf_file.exists():
                with st.expander("⚙️ View my_models_conf.yaml"):
                    with open(conf_file, 'r') as f:
                        st.code(f.read(), language='yaml')
            
            existing_outputs = check_outputs(script_key)
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                
                for out in existing_outputs:
                    if out.endswith('.json'):
                        out_path = SCRIPT_DIR / out
                        if out_path.exists():
                            with st.expander(f"📋 {out}"):
                                with open(out_path, 'r') as f:
                                    st.json(json.load(f))
            
            st.warning("⚠️ This may take hours to run!")
            
            # Run button - different behavior for 4-GPU mode
            if gpu_mode == "4-GPU Parallel":
                st.markdown("**4-GPU mode runs `bash run_4gpus.sh` which launches 4 parallel processes:**")
                st.code("CUDA_VISIBLE_DEVICES=0,1,2,3 with SHARD_IDX=0,1,2,3", language="text")
                
                if st.button("🚀 Run 4-GPU Forecasting", type="primary", key="run_forecast"):
                    with st.spinner("Running run_4gpus.sh... (distributing across 4 GPUs)"):
                        try:
                            import subprocess
                            shell_script = SCRIPT_DIR / "run_4gpus.sh"
                            result = subprocess.run(
                                ["bash", str(shell_script)],
                                cwd=str(SCRIPT_DIR),
                                capture_output=True,
                                text=True,
                                timeout=14400  # 4 hours timeout
                            )
                            returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
                            
                            if returncode == 0:
                                st.success("✅ All 4 GPU processes completed!")
                            else:
                                st.error(f"❌ Failed (code {returncode})")
                            
                            with st.expander("📤 Output", expanded=True):
                                if stdout:
                                    st.text(stdout)
                                if stderr:
                                    st.error(stderr)
                            
                            # Show logs from each GPU
                            logs_dir = SCRIPT_DIR / "logs"
                            if logs_dir.exists():
                                st.subheader("GPU Logs")
                                for i in range(4):
                                    log_file = logs_dir / f"gpu{i}.txt"
                                    if log_file.exists():
                                        with st.expander(f"📋 GPU {i} Log"):
                                            with open(log_file, 'r') as f:
                                                st.text(f.read()[-5000:])  # Last 5000 chars
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                if st.button("📈 Run Forecasting", type="primary", key="run_forecast"):
                    with st.spinner(f"Running {info['file']}... (this may take a long time)"):
                        try:
                            returncode, stdout, stderr = run_script(info['file'])
                            
                            if returncode == 0:
                                st.success("✅ Completed!")
                            else:
                                st.error(f"❌ Failed (code {returncode})")
                            
                            with st.expander("📤 Output", expanded=True):
                                if stdout:
                                    st.text(stdout)
                                if stderr:
                                    st.error(stderr)
                        except Exception as e:
                            st.error(f"Error: {e}")

# ===== TAB 4: EMBEDDING =====
with tabs[3]:
    st.header("🧬 Embedding Generation")
    
    if not enable_embedding:
        st.warning("⚠️ Disabled. Enable in sidebar.")
        st.info("💡 Requires: GPU, Chronos models, huggingface_hub")
    else:
        # Choose between single GPU and multi-GPU version
        gpu_mode_embed = st.radio(
            "Select GPU Mode",
            ["Single GPU", "Multi-GPU Parallel"],
            horizontal=True,
            key="embed_gpu_mode",
            help="Multi-GPU mode distributes embedding generation across multiple GPUs"
        )
        
        if gpu_mode_embed == "Single GPU":
            info = SCRIPTS['embedding']
            script_key = 'embedding'
        else:
            info = SCRIPTS['embedding_multi_gpu']
            script_key = 'embedding_multi_gpu'
        
        st.markdown(f"**Script:** `{info['file']}`")
        st.info(info['description'])
        
        icon, exists = check_script_exists(script_key)
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            st.subheader("Supported Models")
            models_df = pd.DataFrame({
                "Name": ["ChronosT5Tiny", "ChronosT5Mini", "ChronosT5Small", "ChronosT5Base", "ChronosT5Large"],
                "HuggingFace ID": ["amazon/chronos-t5-tiny", "amazon/chronos-t5-mini", "amazon/chronos-t5-small", "amazon/chronos-t5-base", "amazon/chronos-t5-large"]
            })
            st.dataframe(models_df, use_container_width=True, hide_index=True)
            
            existing_outputs = check_outputs(script_key)
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
            
            st.warning("⚠️ This may take a long time to run!")
            
            # Multi-GPU specific options
            if gpu_mode_embed == "Multi-GPU Parallel":
                st.subheader("Multi-GPU Configuration")
                st.info("💡 The script automatically distributes features across GPUs using multiprocessing")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_gpus = st.number_input("Number of GPUs", value=4, min_value=1, max_value=8, key="embed_num_gpus")
                with col2:
                    umap_dim = st.number_input("UMAP Dimensions", value=3, min_value=2, max_value=10, key="embed_umap_dim")
                
                if st.button("🧬 Run Multi-GPU Embedding", type="primary", key="run_embed"):
                    with st.spinner(f"Running {info['file']} with --num_gpus={num_gpus}..."):
                        try:
                            import subprocess
                            result = subprocess.run(
                                [sys.executable, str(SCRIPT_DIR / info['file']), 
                                 f"--num_gpus={num_gpus}", f"--dim={umap_dim}", "--verbose"],
                                cwd=str(SCRIPT_DIR),
                                capture_output=True,
                                text=True,
                                timeout=14400  # 4 hours
                            )
                            returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
                            
                            if returncode == 0:
                                st.success("✅ Completed!")
                            else:
                                st.error(f"❌ Failed (code {returncode})")
                            
                            with st.expander("📤 Output", expanded=True):
                                if stdout:
                                    st.text(stdout)
                                if stderr:
                                    st.error(stderr)
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                if st.button("🧬 Run Embedding", type="primary", key="run_embed"):
                    with st.spinner(f"Running {info['file']}..."):
                        try:
                            returncode, stdout, stderr = run_script(info['file'])
                            
                            if returncode == 0:
                                st.success("✅ Completed!")
                            else:
                                st.error(f"❌ Failed (code {returncode})")
                            
                            with st.expander("📤 Output", expanded=True):
                                if stdout:
                                    st.text(stdout)
                                if stderr:
                                    st.error(stderr)
                        except Exception as e:
                            st.error(f"Error: {e}")

# ===== TAB 5: CURVE FITTING =====
with tabs[4]:
    st.header("📐 Curve Fitting")
    info = SCRIPTS['fitting']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_fitting:
        st.warning("⚠️ Disabled. Enable in sidebar.")
    else:
        icon, exists = check_script_exists('fitting')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code (first 200 lines)"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:200]
                    st.code(''.join(lines) + '\n... (truncated)', language='python', line_numbers=True)
            
            st.subheader("Available Models")
            models_df = pd.DataFrame({
                "Model": ["linear", "exponential", "logarithmic", "logistic", "power", "inverse", 
                         "quadratic", "sigmoid", "gompertz", "weibull", "poly3", "poly4"],
                "Formula": ["ax + b", "a·exp(bx)", "a·log(bx+1)", "c/(1+a·exp(-bx))", "a·x^b", "a/(x+b)",
                           "ax² + bx + c", "d+(a-d)/(1+(x/c)^b)", "a·exp(-b·exp(-cx))", "a-b·exp(-cx²)",
                           "ax³+bx²+cx+d", "ax⁴+bx³+cx²+dx+e"]
            })
            st.dataframe(models_df, use_container_width=True, hide_index=True)
            
            existing_outputs = check_outputs('fitting')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                
                for out in existing_outputs:
                    if out.endswith('.csv'):
                        csv_path = SCRIPT_DIR / out
                        with st.expander(f"📊 {out}"):
                            df = pd.read_csv(csv_path, nrows=100)
                            st.dataframe(df, use_container_width=True)
            
            if st.button("📐 Run Curve Fitting", type="primary", key="run_fit"):
                with st.spinner(f"Running {info['file']}... (this may take a while)"):
                    try:
                        returncode, stdout, stderr = run_script(info['file'])
                        
                        if returncode == 0:
                            st.success("✅ Completed!")
                        else:
                            st.error(f"❌ Failed (code {returncode})")
                        
                        with st.expander("📤 Output", expanded=True):
                            if stdout:
                                st.text(stdout)
                            if stderr:
                                st.error(stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ===== TAB 6: CLUSTERING =====
with tabs[5]:
    st.header("🔮 Unsupervised Clustering")
    info = SCRIPTS['clustering']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_clustering:
        st.warning("⚠️ Disabled. Enable in sidebar.")
    else:
        icon, exists = check_script_exists('clustering')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            existing_outputs = check_outputs('clustering')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                
                # Show cluster plot if exists
                img_path = SCRIPT_DIR / "pca_kmeans_k3_clusters.png"
                if img_path.exists():
                    st.image(str(img_path), caption="PCA K-Means Clusters")
                
                # Show cluster assignments
                csv_path = SCRIPT_DIR / "cluster_assignments_k3.csv"
                if csv_path.exists():
                    with st.expander("📊 Cluster Assignments"):
                        df = pd.read_csv(csv_path)
                        st.dataframe(df.head(100), use_container_width=True)
                        
                        # Cluster distribution
                        fig, ax = plt.subplots()
                        df['Cluster'].value_counts().plot(kind='bar', ax=ax)
                        ax.set_xlabel('Cluster')
                        ax.set_ylabel('Count')
                        ax.set_title('Cluster Distribution')
                        st.pyplot(fig)
            
            if st.button("🔮 Run Clustering", type="primary", key="run_cluster"):
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'])
                        
                        if returncode == 0:
                            st.success("✅ Completed!")
                            st.rerun()  # Refresh to show new outputs
                        else:
                            st.error(f"❌ Failed (code {returncode})")
                        
                        with st.expander("📤 Output", expanded=True):
                            if stdout:
                                st.text(stdout)
                            if stderr:
                                st.error(stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ===== TAB 7: ANOVA =====
with tabs[6]:
    st.header("📊 ANOVA Analysis")
    info = SCRIPTS['anova']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_anova:
        st.warning("⚠️ Disabled. Enable in sidebar.")
    else:
        icon, exists = check_script_exists('anova')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            existing_outputs = check_outputs('anova')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                
                xlsx_path = SCRIPT_DIR / "ANOVA - OneWay.xlsx"
                if xlsx_path.exists():
                    try:
                        df = pd.read_excel(xlsx_path, header=[0,1])
                        st.dataframe(df, use_container_width=True)
                    except:
                        df = pd.read_excel(xlsx_path)
                        st.dataframe(df, use_container_width=True)
            
            if st.button("📊 Run ANOVA", type="primary", key="run_anova"):
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'])
                        
                        if returncode == 0:
                            st.success("✅ Completed!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed (code {returncode})")
                        
                        with st.expander("📤 Output", expanded=True):
                            if stdout:
                                st.text(stdout)
                            if stderr:
                                st.error(stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ===== TAB 8: DESCRIPTIVE STATS =====
with tabs[7]:
    st.header("📋 Descriptive Statistics")
    info = SCRIPTS['descriptive']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_descriptive:
        st.warning("⚠️ Disabled. Enable in sidebar.")
    else:
        icon, exists = check_script_exists('descriptive')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            existing_outputs = check_outputs('descriptive')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                
                xlsx_path = SCRIPT_DIR / "Descriptive_Table_By_Cluster_UniqueCells.xlsx"
                if xlsx_path.exists():
                    df = pd.read_excel(xlsx_path)
                    st.dataframe(df, use_container_width=True)
            
            if st.button("📋 Run Descriptive Stats", type="primary", key="run_desc"):
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'])
                        
                        if returncode == 0:
                            st.success("✅ Completed!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed (code {returncode})")
                        
                        with st.expander("📤 Output", expanded=True):
                            if stdout:
                                st.text(stdout)
                            if stderr:
                                st.error(stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
### 📚 Pipeline Overview

This Streamlit app **runs the existing Python scripts** in the directory:

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

**To run:** `streamlit run app.py`

**For 4-GPU forecasting:** `bash run_4gpus.sh`

*Built with Streamlit • Cellomics-5-C Analysis Pipeline*
""")
