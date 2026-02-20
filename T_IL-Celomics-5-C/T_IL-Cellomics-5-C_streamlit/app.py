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
- embedding_unsupervised_clustering.py: K-Means clustering
- ANOVA.py: Statistical analysis
- descriptive_table_by_cluster.py: Descriptive statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import subprocess
import signal
import threading
import time
import atexit
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # headless-safe backend (no display needed)
import matplotlib.pyplot as plt

# Get the script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)  # Change to script directory for relative paths


def _find_file(name: str) -> Path:
    """Find *name* in SCRIPT_DIR or common subdirs (cell_data/, embeddings/, fitting/).

    Returns the resolved Path if found, otherwise SCRIPT_DIR / name (so the
    caller gets the default expected location).
    """
    for candidate in [
        SCRIPT_DIR / name,
        SCRIPT_DIR / "cell_data" / name,
        SCRIPT_DIR / "embeddings" / name,
        SCRIPT_DIR / "fitting" / name,
    ]:
        if candidate.exists():
            return candidate
    return SCRIPT_DIR / name          # default (may not exist)


def _exp_id() -> str:
    """Return the current experiment ID from session state (default '008')."""
    return st.session_state.get("experiment_id", "008")


def _merged_csv_name() -> str:
    """Dynamic merged CSV filename based on experiment ID."""
    return f"MergedAndFilteredExperiment{_exp_id()}.csv"


def _embedding_json_name() -> str:
    """Dynamic embedding JSON filename based on experiment ID."""
    return f"Embedding{_exp_id()}.json"


def _find_best_k_file(subdir: str, prefix: str, ext: str = ".csv") -> str:
    """Find the first available output file with a k-suffix in *subdir*.

    The clustering scripts write files like
      clustering/Merged_Clusters_PCA_k3.csv
      fitting/embedding_fitting_Merged_Clusters_PCA_k5.csv
    This helper returns the path (relative to SCRIPT_DIR) of the first
    match, or an empty string if nothing is found.
    """
    search_dir = SCRIPT_DIR / subdir
    if search_dir.exists():
        for f in sorted(search_dir.glob(f"{prefix}_k*{ext}")):
            return str(f)
    return ""

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
        "outputs": ["best_model_per_feature.json", "best_chronos_model_per_feature.json", "best_t5_model_per_feature.json"]
    },
    "forecasting_4gpu": {
        "file": "TSA_analysis_4gpu.py",
        "name": "Time Series Forecasting (4-GPU)",
        "description": "Parallel forecasting across 4 GPUs - shards features across GPUs for faster processing",
        "outputs": ["best_model_per_feature.json", "best_chronos_model_per_feature.json", "best_t5_model_per_feature.json"],
        "multi_gpu": True,
        "shell_script": "run_4gpus.sh"
    },
    "embedding": {
        "file": "Embedding.py",
        "name": "Embedding Generation",
        "description": "Generate embeddings using Chronos models + UMAP dimensionality reduction (single GPU)",
        "outputs": ["Embedding{exp_id}.json"]
    },
    "embedding_multi_gpu": {
        "file": "Embedding_multi_gpu.py",
        "name": "Embedding Generation (Multi-GPU)",
        "description": "Parallel embedding generation across multiple GPUs with feature sharding",
        "outputs": ["Embedding{exp_id}.json"],
        "multi_gpu": True
    },
    "fitting": {
        "file": "fit_cell_trajectory.py",
        "name": "Curve Fitting",
        "description": "Fit 12 mathematical models (linear, exponential, logistic, etc.) to cell trajectories",
        "outputs": ["fitting_all_models.csv", "fitting_best_with_nrmse.csv", "fitting_best_model_log_scaled.json"]
    },
    "clustering": {
        "file": "embedding_unsupervised_clustering.py",
        "name": "Unsupervised Clustering",
        "description": "K-Means clustering on embeddings with silhouette sweep (k=2–10), dose analysis & PCA visualization",
        "outputs": ["clustering/cluster_assignments_k*.csv", "clustering/pca_kmeans_k*_clusters.png"]
    },
    "anova": {
        "file": "ANOVA.py",
        "name": "ANOVA Analysis",
        "description": "One-way ANOVA by cluster with multi-level headers",
        "outputs": ["ANOVA - OneWay.xlsx"]
    },
    "descriptive": {
        "file": "descriptive_table_by_cluster.py",
        "name": "Descriptive Statistics",
        "description": "Mean, Std, SE, CI per cluster for all features",
        "outputs": ["Descriptive_Table_By_Cluster_UniqueCells.xlsx"]
    },
    "embfit_clustering": {
        "file": "embedding_fitting_unsupervised_clustering.py",
        "name": "Emb+Fit Clustering",
        "description": "Combine embedding vectors + fitting parameters, then PCA + K-Means with silhouette sweep & dose analysis",
        "outputs": ["fitting/embedding_fitting_combined_by_feature_scaled.json",
                    "fitting/embedding_fitting_PCA_KMeans_k*.png",
                    "fitting/embedding_fitting_PCA_KMeans_by_Treatment_k*.png"]
    },
    "embfit_anova": {
        "file": "embedding_fitting_anova.py",
        "name": "Emb+Fit ANOVA",
        "description": "One-way ANOVA by cluster on embedding+fitting merged data",
        "outputs": ["embedding_fitting_ANOVA - OneWay.xlsx"]
    },
    "embfit_descriptive": {
        "file": "embedding_fitting_descriptive_table.py",
        "name": "Emb+Fit Descriptive Stats",
        "description": "Mean, Std, SE, CI per cluster for embedding+fitting merged data",
        "outputs": ["embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells.xlsx"]
    },
    "dose_summary": {
        "file": "build_dose_summary.py",
        "name": "Dose Extraction Summary",
        "description": "Build per-cell dose-dependency summary from normalised Excel exports (aggregates _Norm means & _Category majority per Experiment/Parent)",
        "outputs": ["dose_dependency_summary_all_wells.csv"]
    }
}

# ===== HELPER: kill a subprocess tree =====
def _kill_proc(pid: int):
    """Kill a process and all its children (SIGTERM then SIGKILL)."""
    try:
        if sys.platform == "win32":
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            # Give processes a moment to exit gracefully, then force-kill
            time.sleep(1)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass  # already dead — good
    except (ProcessLookupError, OSError, PermissionError):
        pass


def _kill_gpu_orphans():
    """Find and kill any orphaned GPU processes (TSA_analysis_4gpu.py and
    Embedding_multi_gpu.py workers). Called before starting a new multi-GPU
    run, on stop, and at exit."""
    if sys.platform == "win32":
        return
    # Use a single pgrep regex to match all GPU scripts at once.
    # The pattern requires 'python' before the script name to avoid matching
    # editors, grep, or other tools that happen to mention the filename.
    # NOTE: pgrep uses POSIX ERE — no (?:) allowed; use plain (|) grouping.
    _pattern = r"python.*(TSA_analysis_4gpu\.py|Embedding_multi_gpu\.py)"
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", _pattern],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p) for p in result.stdout.strip().split() if p.strip()]
        pids = [p for p in pids if p != my_pid]
        if not pids:
            return
        # SIGTERM first
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1)
        # Force-kill survivors
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        print(f"🧹 Killed {len(pids)} orphaned GPU process(es): {pids}", flush=True)
    except Exception:
        pass


# Clean up GPU processes when the Streamlit app exits
atexit.register(_kill_gpu_orphans)


# ===== HELPER FUNCTION: Run Script =====
def run_script(script_name: str, env_vars: dict = None):
    """Run a Python script, stream output to terminal, and capture it for the UI.
    Checks for stop_requested every 0.5s so the Stop button is responsive."""
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        return -1, "", f"Script not found: {script_path}"

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    print(f"\n{'='*60}")
    print(f"▶ RUNNING: {script_name}")
    print(f"{'='*60}", flush=True)

    kwargs = dict(
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    # On POSIX create a process group so we can kill the whole tree
    if sys.platform != "win32":
        kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],
        **kwargs,
    )

    # Store PID so the Stop button can kill it on rerun
    st.session_state.running_pid = proc.pid
    st.session_state.running_stage = script_name

    # Read stdout in a background thread so the main thread can poll stop_requested
    captured_lines: list[str] = []

    def _reader():
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                captured_lines.append(line)
        except ValueError:
            pass  # pipe closed after kill

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # Poll until process exits or user requests stop
    while proc.poll() is None:
        if st.session_state.stop_requested:
            _kill_proc(proc.pid)
            reader_thread.join(timeout=2)
            st.session_state.running_pid = None
            st.session_state.running_stage = None
            st.session_state.stop_requested = False
            combined = "".join(captured_lines)
            print(f"\n🛑 STOPPED by user: {script_name}", flush=True)
            return -2, combined, "Stopped by user"
        time.sleep(0.5)

    reader_thread.join(timeout=5)

    # Clear PID tracking
    st.session_state.running_pid = None
    st.session_state.running_stage = None

    combined = "".join(captured_lines)
    rc = proc.returncode
    status = "✅ SUCCESS" if rc == 0 else f"❌ FAILED (exit code {rc})"
    print(f"{'─'*60}")
    print(f"{status}: {script_name}")
    print(f"{'='*60}\n", flush=True)

    return rc, combined, ""

# ===== PIPELINE HELPERS =====

# Maps each stage to the key output files it produces
def _stage_outputs() -> dict:
    emb_json = _embedding_json_name()
    return {
        "data_prep":          ["raw_all_cells.csv"],
        "feature_selection":  ["selected_features.txt", "normalized_all_cells.csv"],
        "forecasting":        ["best_model_per_feature.json", "best_chronos_model_per_feature.json", "best_t5_model_per_feature.json"],
        "forecasting_4gpu":   ["best_model_per_feature.json", "best_chronos_model_per_feature.json", "best_t5_model_per_feature.json"],
        "embedding":          [emb_json],
        "embedding_multi_gpu":[emb_json],
        "fitting":            ["fitting_all_models.csv", "fitting_best_with_nrmse.csv", "fitting_best_model_log_scaled.json"],
        "clustering":         ["clustering/Merged_Clusters_PCA_k*"],
        "anova":              ["ANOVA - OneWay.xlsx"],
        "descriptive":        ["Descriptive_Table_By_Cluster_UniqueCells.xlsx"],
        "embfit_clustering":  ["fitting/embedding_fitting_Merged_Clusters_PCA_k*"],
        "embfit_anova":       ["embedding_fitting_ANOVA - OneWay.xlsx"],
        "embfit_descriptive": ["embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells.xlsx"],
        "dose_summary":     ["dose_dependency_summary_all_wells.csv"],
    }

# Maps each stage to the stages that consume its outputs
DOWNSTREAM_MAP = {
    "data_prep":          ["feature_selection", "forecasting", "fitting"],
    "feature_selection":  ["forecasting"],
    "forecasting":        ["embedding"],
    "forecasting_4gpu":   ["embedding"],
    "embedding":          ["clustering", "embfit_clustering"],
    "embedding_multi_gpu":["clustering", "embfit_clustering"],
    "fitting":            ["embfit_clustering"],
    "clustering":         ["anova", "descriptive"],
    "embfit_clustering":  ["embfit_anova", "embfit_descriptive"],
    "anova":              [],
    "descriptive":        [],
    "embfit_anova":       [],
    "embfit_descriptive": [],
    "dose_summary":       ["clustering", "embfit_clustering"],
}

# Human-readable step labels
STAGE_LABELS = {
    "data_prep":          "Step 1 · Data Preparation",
    "feature_selection":  "Step 2 · Feature Selection",
    "forecasting":        "Step 3 · Forecasting",
    "forecasting_4gpu":   "Step 3 · Forecasting (4-GPU)",
    "embedding":          "Step 4 · Embedding",
    "embedding_multi_gpu":"Step 4 · Embedding (Multi-GPU)",
    "fitting":            "Step 5 · Curve Fitting",
    "clustering":         "Step 6 · Clustering",
    "anova":              "Step 7 · ANOVA",
    "descriptive":        "Step 8 · Descriptive Stats",
    "embfit_clustering":  "Step 6b · Emb+Fit Clustering",
    "embfit_anova":       "Step 7b · Emb+Fit ANOVA",
    "embfit_descriptive": "Step 8b · Emb+Fit Descriptive",
    "dose_summary":       "Step 0 · Dose Extraction",
}

# Maps each pipeline stage to glob patterns for its output figures
STAGE_FIGURE_PATTERNS: dict[str, list[str]] = {
    "data_prep":          [],
    "feature_selection":  [
        "figures/heatmap_final_features.png",
        "figures/autocorr_heatmap*.png",
        "figures/silhouette_plot.png",
    ],
    "forecasting":        [],
    "forecasting_4gpu":   [],
    "embedding":          [],
    "embedding_multi_gpu":[],
    "fitting":            [
        "figures/dist_clipped_*.png",
        "figures/stacked_*.png",
        "figures/boxplot_*.png",
        "figures/avg_*.png",
        "figures/heatmap_best_model_*.png",
        "figures/sunburst_*.png",
        "figures/sunburst_*.html",
    ],
    "clustering":         [
        "clustering/*.png",
    ],
    "anova":              [],
    "descriptive":        [],
    "embfit_clustering":  [
        "fitting/embedding_fitting_PCA_KMeans*.png",
        "fitting/cluster_vs_dose_heatmap*.png",
    ],
    "embfit_anova":       [],
    "embfit_descriptive": [],
    "dose_summary":       [],
}


def display_stage_figures(stage_key: str, use_expanders: bool = True):
    """Find and display all output figures/images produced by a pipeline stage.

    When *use_expanders* is True (default) each image gets its own expander
    with a header.  Set to False for a compact inline display (e.g. inside
    the Run-All results container).
    """
    patterns = STAGE_FIGURE_PATTERNS.get(stage_key, [])
    if not patterns:
        return

    images: list[Path] = []
    html_files: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for f in sorted(SCRIPT_DIR.glob(pat)):
            key = str(f)
            if key not in seen:
                seen.add(key)
                if f.suffix.lower() == ".html":
                    html_files.append(f)
                elif f.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".gif"):
                    images.append(f)

    if not images and not html_files:
        return

    if use_expanders:
        st.subheader("📊 Generated Figures")

    for img in images:
        if use_expanders:
            with st.expander(f"🖼️ {img.relative_to(SCRIPT_DIR)}", expanded=True):
                st.image(str(img))
        else:
            st.image(str(img), caption=str(img.relative_to(SCRIPT_DIR)))

    for html_file in html_files:
        import streamlit.components.v1 as components
        if use_expanders:
            with st.expander(f"📈 {html_file.relative_to(SCRIPT_DIR)}", expanded=True):
                html_content = html_file.read_text(encoding="utf-8")
                components.html(html_content, height=600, scrolling=True)
        else:
            st.caption(str(html_file.relative_to(SCRIPT_DIR)))
            html_content = html_file.read_text(encoding="utf-8")
            components.html(html_content, height=600, scrolling=True)


def get_pipeline_env(script_key: str) -> dict:
    """Build environment variables dict for a given pipeline step."""
    env = {}
    data_path = st.session_state.get("input_data_path", "")

    if script_key == "data_prep":
        env["PIPELINE_INPUT_EXCEL"] = data_path
        env["PIPELINE_OUTPUT_CSV"]  = str(SCRIPT_DIR / "raw_all_cells.csv")
        env["PIPELINE_MIN_FRAMES"]  = str(st.session_state.get("param_min_frames", 25))
        env["PIPELINE_MAX_GAP"]     = str(st.session_state.get("param_max_gap", 5))
    elif script_key == "feature_selection":
        env["PIPELINE_RAW_CSV"]       = str(_find_file("raw_all_cells.csv"))
        env["PIPELINE_NORM_CSV"]      = str(SCRIPT_DIR / "normalized_all_cells.csv")
        env["PIPELINE_FEATURES_FILE"] = str(_find_file("selected_features.txt"))
        env["PIPELINE_EXPERIMENT_ID"] = _exp_id()
        env["PIPELINE_PCA_VARIANCE"]        = str(st.session_state.get("param_pca_variance", 0.95))
        env["PIPELINE_CORR_THRESHOLD"]      = str(st.session_state.get("param_corr_threshold", 0.8))
        env["PIPELINE_CONST_CELL_THRESH"]   = str(st.session_state.get("param_const_cell_thresh", 0.99))
        env["PIPELINE_STD_KEEP_THRESH"]     = str(st.session_state.get("param_std_keep_thresh", 0.2))
        env["PIPELINE_MAX_LAG"]             = str(st.session_state.get("param_max_lag", 25))
        env["PIPELINE_MIN_FEATURES"]        = str(st.session_state.get("param_min_features", 5))
        env["PIPELINE_FS_CLUSTERS"]         = str(st.session_state.get("param_fs_clusters", 3))
        env["PIPELINE_NORMALIZE_BEFORE_VARIANCE"] = str(st.session_state.get("param_normalize_before_variance", True))
    elif script_key in ("forecasting", "forecasting_4gpu"):
        env["PIPELINE_RAW_CSV"]       = str(_find_file("raw_all_cells.csv"))
        env["PIPELINE_FEATURES_FILE"] = str(_find_file("selected_features.txt"))
        env["PIPELINE_MAX_CELLS"]     = str(st.session_state.get("param_max_cells", 500))
    elif script_key in ("embedding", "embedding_multi_gpu"):
        env["PIPELINE_MERGED_CSV"]           = data_path or str(_find_file(_merged_csv_name()))
        env["PIPELINE_CHRONOS_MODEL_DICT"]   = str(_find_file("best_t5_model_per_feature.json"))
        env["PIPELINE_EMBEDDING_JSON"]       = str(_find_file(_embedding_json_name()))
        env["PIPELINE_EXPERIMENT_ID"]        = _exp_id()
        env["PIPELINE_UMAP_DIM"]             = str(st.session_state.get("param_umap_dim", 3))
    elif script_key == "fitting":
        env["PIPELINE_RAW_CSV"]          = str(_find_file("raw_all_cells.csv"))
        env["PIPELINE_MAXFEV"]           = str(st.session_state.get("param_maxfev", 10000))
        env["PIPELINE_PVAL_THRESHOLD"]   = str(st.session_state.get("param_pval_threshold", 0.05))
    elif script_key == "clustering":
        env["PIPELINE_MERGED_CSV"]     = data_path or str(_find_file(_merged_csv_name()))
        env["PIPELINE_EMBEDDING_JSON"] = str(_find_file(_embedding_json_name()))
        env["PIPELINE_EXPERIMENT_ID"]  = _exp_id()
        treatments_str = st.session_state.get("treatments_list", "")
        if treatments_str.strip():
            env["PIPELINE_TREATMENTS"] = treatments_str.strip()
        dose_path = st.session_state.get("dose_csv_path", "")
        if dose_path.strip():
            env["PIPELINE_DOSE_CSV"] = dose_path.strip()
        else:
            _auto_dose = _find_file("dose_dependency_summary_all_wells.csv")
            if _auto_dose.exists():
                env["PIPELINE_DOSE_CSV"] = str(_auto_dose)
        control_ch = st.session_state.get("param_control_channel", "")
        if control_ch.strip():
            env["PIPELINE_CONTROL_CHANNEL"] = control_ch.strip()
        env["PIPELINE_K_MIN"]           = str(st.session_state.get("param_k_min", 2))
        env["PIPELINE_K_MAX"]           = str(st.session_state.get("param_k_max", 10))
        env["PIPELINE_KMEANS_N_INIT"]   = str(st.session_state.get("param_kmeans_n_init", 10))
        env["PIPELINE_KMEANS_SEED"]     = str(st.session_state.get("param_kmeans_seed", 42))
        env["PIPELINE_PCA_COMPONENTS"]  = str(st.session_state.get("param_pca_components", 2))
        env["PIPELINE_NUM_BEST_K"]      = str(st.session_state.get("param_num_best_k", 2))
    elif script_key == "anova":
        _csv = _find_best_k_file("clustering", "Merged_Clusters_PCA")
        env["PIPELINE_CLUSTERS_CSV"] = _csv or str(SCRIPT_DIR / "clustering" / "Merged_Clusters_PCA.csv")
        env["PIPELINE_ANOVA_PVAL"]   = str(st.session_state.get("param_anova_pval", 0.05))
    elif script_key == "descriptive":
        _csv = _find_best_k_file("clustering", "Merged_Clusters_PCA")
        env["PIPELINE_CLUSTERS_CSV"] = _csv or str(SCRIPT_DIR / "clustering" / "Merged_Clusters_PCA.csv")
        env["PIPELINE_CI_ZSCORE"]    = str(st.session_state.get("param_ci_zscore", 1.96))
    elif script_key == "embfit_clustering":
        env["PIPELINE_MERGED_CSV"]     = data_path or str(_find_file(_merged_csv_name()))
        env["PIPELINE_EMBEDDING_JSON"] = str(_find_file(_embedding_json_name()))
        env["PIPELINE_FITTING_JSON"]   = str(_find_file("fitting_best_model_log_scaled.json"))
        env["PIPELINE_EXPERIMENT_ID"]  = _exp_id()
        treatments_str = st.session_state.get("treatments_list", "")
        if treatments_str.strip():
            env["PIPELINE_TREATMENTS"] = treatments_str.strip()
        dose_path = st.session_state.get("dose_csv_path", "")
        if dose_path.strip():
            env["PIPELINE_DOSE_CSV"] = dose_path.strip()
        else:
            _auto_dose = _find_file("dose_dependency_summary_all_wells.csv")
            if _auto_dose.exists():
                env["PIPELINE_DOSE_CSV"] = str(_auto_dose)
        control_ch = st.session_state.get("param_control_channel", "")
        if control_ch.strip():
            env["PIPELINE_CONTROL_CHANNEL"] = control_ch.strip()
        env["PIPELINE_K_MIN"]           = str(st.session_state.get("param_k_min", 2))
        env["PIPELINE_K_MAX"]           = str(st.session_state.get("param_k_max", 10))
        env["PIPELINE_KMEANS_N_INIT"]   = str(st.session_state.get("param_kmeans_n_init", 10))
        env["PIPELINE_KMEANS_SEED"]     = str(st.session_state.get("param_kmeans_seed", 42))
        env["PIPELINE_PCA_COMPONENTS"]  = str(st.session_state.get("param_pca_components", 2))
        env["PIPELINE_NUM_BEST_K"]      = str(st.session_state.get("param_num_best_k", 2))
        env["PIPELINE_CI_ZSCORE"]       = str(st.session_state.get("param_ci_zscore", 1.96))
    elif script_key == "embfit_anova":
        _csv = _find_best_k_file("fitting", "embedding_fitting_Merged_Clusters_PCA")
        env["PIPELINE_EMBFIT_CLUSTERS_CSV"] = _csv or str(SCRIPT_DIR / "fitting" / "embedding_fitting_Merged_Clusters_PCA.csv")
        env["PIPELINE_ANOVA_PVAL"]   = str(st.session_state.get("param_anova_pval", 0.05))
    elif script_key == "embfit_descriptive":
        _csv = _find_best_k_file("fitting", "embedding_fitting_Merged_Clusters_PCA")
        env["PIPELINE_EMBFIT_CLUSTERS_CSV"] = _csv or str(SCRIPT_DIR / "fitting" / "embedding_fitting_Merged_Clusters_PCA.csv")
        env["PIPELINE_CI_ZSCORE"]    = str(st.session_state.get("param_ci_zscore", 1.96))
    elif script_key == "dose_summary":
        dose_dir = st.session_state.get("param_dose_dir", "").strip()
        if dose_dir:
            env["PIPELINE_DOSE_DIR"] = dose_dir
        env["PIPELINE_DOSE_EXCEL_GLOB"] = st.session_state.get("param_dose_excel_glob", "Gab_Normalized_Combined_*.xlsx")
        env["PIPELINE_DOSE_SHEET"]      = st.session_state.get("param_dose_sheet", "Area")
        env["PIPELINE_DOSE_PREFIX"]     = st.session_state.get("param_dose_prefix", "Gab_Normalized_Combined_")
        env["PIPELINE_DOSE_OUTPUT"]     = str(SCRIPT_DIR / "dose_dependency_summary_all_wells.csv")
        well_map = st.session_state.get("param_dose_well_map", "")
        if well_map.strip():
            env["PIPELINE_DOSE_WELL_MAP"] = well_map.strip()
    return env


def check_pipeline_deps(script_key: str) -> tuple:
    """Return (all_ok: bool, missing: list[str]) for a pipeline step."""
    missing = []
    data_path = st.session_state.get("input_data_path", "")

    if script_key == "data_prep":
        if not data_path or not os.path.exists(data_path):
            missing.append("Summary table — provide the path in Pipeline Configuration above")
    elif script_key in ("feature_selection", "fitting"):
        if not _find_file("raw_all_cells.csv").exists():
            missing.append("raw_all_cells.csv — run **Step 1 · Data Preparation** first")
    elif script_key in ("forecasting", "forecasting_4gpu"):
        if not _find_file("raw_all_cells.csv").exists():
            missing.append("raw_all_cells.csv — run **Step 1 · Data Preparation** first")
        if not _find_file("selected_features.txt").exists():
            missing.append("selected_features.txt — run **Step 2 · Feature Selection** first")
    elif script_key in ("embedding", "embedding_multi_gpu"):
        _data_found = (data_path and os.path.exists(data_path)) or _find_file(_merged_csv_name()).exists()
        if not _data_found:
            missing.append(f"Summary table — provide the path in Pipeline Configuration")
        if not _find_file("best_t5_model_per_feature.json").exists():
            missing.append("best_t5_model_per_feature.json — run **Step 3 · Forecasting** first")
    elif script_key == "clustering":
        _data_found = (data_path and os.path.exists(data_path)) or _find_file(_merged_csv_name()).exists()
        if not _data_found:
            missing.append(f"Summary table — provide the path in Pipeline Configuration")
        if not _find_file(_embedding_json_name()).exists():
            missing.append(f"{_embedding_json_name()} — run **Step 4 · Embedding** first")
    elif script_key in ("anova", "descriptive"):
        if not _find_best_k_file("clustering", "Merged_Clusters_PCA"):
            missing.append("clustering/Merged_Clusters_PCA_k*.csv — run **Step 6 · Clustering** first")
    elif script_key == "embfit_clustering":
        _data_found = (data_path and os.path.exists(data_path)) or _find_file(_merged_csv_name()).exists()
        if not _data_found:
            missing.append(f"Summary table — provide the path in Pipeline Configuration")
        if not _find_file(_embedding_json_name()).exists():
            missing.append(f"{_embedding_json_name()} — run **Step 4 · Embedding** first")
        if not _find_file("fitting_best_model_log_scaled.json").exists():
            missing.append("fitting_best_model_log_scaled.json — run **Step 5 · Curve Fitting** first")
    elif script_key in ("embfit_anova", "embfit_descriptive"):
        if not _find_best_k_file("fitting", "embedding_fitting_Merged_Clusters_PCA"):
            missing.append("fitting/embedding_fitting_Merged_Clusters_PCA_k*.csv — run **Step 6b · Emb+Fit Clustering** first")
    elif script_key == "dose_summary":
        # Dose summary only needs the input Excel files — validated at runtime
        pass
    return len(missing) == 0, missing


def stage_outputs_exist(script_key: str) -> bool:
    """Return True if ALL key output files of this stage exist on disk.
    Supports glob patterns (e.g. 'clustering/Merged_Clusters_PCA_k*')."""
    for fname in _stage_outputs().get(script_key, []):
        if "*" in fname:
            if not list(SCRIPT_DIR.glob(fname)):
                return False
        else:
            if not _find_file(fname).exists():
                return False
    return True


def get_skip_enable_key(script_key: str) -> str:
    """Map a script_key to its sidebar enable flag name."""
    mapping = {
        "data_prep": "enable_data_prep",
        "feature_selection": "enable_feature_selection",
        "forecasting": "enable_forecasting",
        "forecasting_4gpu": "enable_forecasting",
        "embedding": "enable_embedding",
        "embedding_multi_gpu": "enable_embedding",
        "fitting": "enable_fitting",
        "clustering": "enable_clustering",
        "anova": "enable_anova",
        "descriptive": "enable_descriptive",
        "embfit_clustering": "enable_embfit_clustering",
        "embfit_anova": "enable_embfit_anova",
        "embfit_descriptive": "enable_embfit_descriptive",
        "dose_summary": "enable_dose_summary",
    }
    return mapping.get(script_key, "")


def show_skip_status(script_key: str):
    """When a stage is skipped, check if its outputs exist.
    If not, and downstream stages need them, show a blocking warning."""
    outputs_ok = stage_outputs_exist(script_key)
    downstream = DOWNSTREAM_MAP.get(script_key, [])
    needed_by = [STAGE_LABELS[d] for d in downstream if d in STAGE_LABELS]

    if outputs_ok:
        st.info(f"⏭️ Skipped — but outputs already exist on disk, downstream stages can proceed.")
    elif needed_by:
        st.error(
            f"🚫 Skipped, and outputs are **missing**. "
            f"The following stages need this step's output and **cannot run** until you either "
            f"enable and run this step, or provide the files manually:\n\n"
            + "\n".join(f"- {n}" for n in needed_by)
        )
    else:
        st.warning("⏭️ Skipped. No downstream stages depend on this step.")


def show_dependency_status(script_key: str) -> bool:
    """Display dependency warnings and return True if all deps are met (run is allowed)."""
    ok, missing = check_pipeline_deps(script_key)
    if not ok:
        for m in missing:
            st.error(f"🚫 Missing input: {m}")
        st.warning("⛔ Cannot run — resolve the missing inputs above first.")
    return ok

# ===== HELPER: Check if script exists =====
def check_script_exists(script_key: str) -> tuple:
    """Check if script file exists and return status icon."""
    script_info = SCRIPTS[script_key]
    script_path = SCRIPT_DIR / script_info["file"]
    exists = script_path.exists()
    return "✅" if exists else "❌", exists

# ===== HELPER: Check output files =====
def check_outputs(script_key: str) -> list:
    """Check which output files exist.  Supports subdir and glob patterns."""
    outputs = SCRIPTS[script_key].get("outputs", [])
    existing = []
    for out in outputs:
        # Resolve {exp_id} placeholder
        out = out.replace("{exp_id}", _exp_id())
        if "*" in out:
            matches = list(SCRIPT_DIR.glob(out))
            existing.extend(str(m.relative_to(SCRIPT_DIR)) for m in matches)
        elif (SCRIPT_DIR / out).exists():
            existing.append(out)
    return existing

# ===== SESSION STATE =====
if 'script_outputs' not in st.session_state:
    st.session_state.script_outputs = {}
if 'experiment_id' not in st.session_state:
    st.session_state.experiment_id = "008"
if 'input_data_path' not in st.session_state:
    st.session_state.input_data_path = ""
if 'treatments_list' not in st.session_state:
    st.session_state.treatments_list = "CON0, BRCACON1, BRCACON2, BRCACON3, BRCACON4, BRCACON5"
if 'dose_csv_path' not in st.session_state:
    st.session_state.dose_csv_path = ""
if 'running_pid' not in st.session_state:
    st.session_state.running_pid = None
if 'running_stage' not in st.session_state:
    st.session_state.running_stage = None
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
# Non-blocking GPU process tracking
for _init_key in ('_4gpu_proc', '_4gpu_thread', '_4gpu_lines', '_4gpu_start',
                  '_mgpu_embed_proc', '_mgpu_embed_thread', '_mgpu_embed_lines', '_mgpu_embed_start'):
    if _init_key not in st.session_state:
        st.session_state[_init_key] = None

# --- Stage parameter defaults ---
_PARAM_DEFAULTS = {
    # Data Preparation
    "param_min_frames": 25,
    "param_max_gap": 5,
    # Feature Selection
    "param_pca_variance": 0.95,
    "param_corr_threshold": 0.80,
    "param_const_cell_thresh": 0.99,
    "param_std_keep_thresh": 0.20,
    "param_max_lag": 25,
    "param_min_features": 5,
    "param_fs_clusters": 3,
    "param_normalize_before_variance": True,
    # Forecasting
    "param_max_cells": 500,
    # Embedding
    "param_umap_dim": 3,
    # Fitting
    "param_maxfev": 10000,
    "param_pval_threshold": 0.05,
    # Clustering (shared by both clustering stages)
    "param_k_min": 2,
    "param_k_max": 10,
    "param_kmeans_n_init": 10,
    "param_kmeans_seed": 42,
    "param_pca_components": 2,
    "param_num_best_k": 2,
    # ANOVA
    "param_anova_pval": 0.05,
    # Descriptive
    "param_ci_zscore": 1.96,
    # Control channel
    "param_control_channel": "NNIR",
    # Dose Extraction Summary
    "param_dose_dir": "",
    "param_dose_excel_glob": "Gab_Normalized_Combined_*.xlsx",
    "param_dose_sheet": "Area",
    "param_dose_prefix": "Gab_Normalized_Combined_",
    "param_dose_well_map": "",
}
for _pk, _pv in _PARAM_DEFAULTS.items():
    if _pk not in st.session_state:
        st.session_state[_pk] = _pv

# ===== STOP HANDLER — kill orphaned subprocess from previous rerun =====
if st.session_state.stop_requested and st.session_state.running_pid:
    _pid = st.session_state.running_pid
    _stage = st.session_state.running_stage or "unknown"
    _kill_proc(_pid)
    # Also kill any orphaned GPU processes (4-GPU mode spawns children that
    # may survive the parent bash process being killed)
    _kill_gpu_orphans()
    print(f"\n\U0001f6d1 STOPPED by user: {_stage} (PID {_pid})", flush=True)
    st.session_state.running_pid = None
    st.session_state.running_stage = None
    st.session_state.stop_requested = False
    st.warning(f"\U0001f6d1 **{_stage}** was stopped by user.")
elif st.session_state.stop_requested:
    # No PID — run_script already handled it or nothing was running
    _kill_gpu_orphans()  # still clean up any orphans
    st.session_state.stop_requested = False


def _request_stop():
    """Callback for Stop buttons — sets flag so cleanup handler fires on rerun."""
    st.session_state.stop_requested = True

# ===== SIDEBAR =====
st.sidebar.title("🔬 Pipeline Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("Enable/Disable Steps")
enable_data_prep = st.sidebar.checkbox("📁 Data Preparation", value=True)
enable_feature_selection = st.sidebar.checkbox("🎯 Feature Selection", value=True)
enable_forecasting = st.sidebar.checkbox("📈 Forecasting", value=True, help="Requires GPU & MMF")
enable_embedding = st.sidebar.checkbox("🧬 Embedding", value=True, help="Requires Chronos")
enable_fitting = st.sidebar.checkbox("📐 Curve Fitting", value=True)
enable_clustering = st.sidebar.checkbox("🔮 Clustering", value=True)
enable_anova = st.sidebar.checkbox("📊 ANOVA", value=True)
enable_descriptive = st.sidebar.checkbox("📋 Descriptive Stats", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Emb+Fit Branch")
enable_embfit_clustering = st.sidebar.checkbox("🔗 Emb+Fit Clustering", value=True)
enable_embfit_anova = st.sidebar.checkbox("📊 Emb+Fit ANOVA", value=True)
enable_embfit_descriptive = st.sidebar.checkbox("📋 Emb+Fit Descriptive", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Dose Extraction")
enable_dose_summary = st.sidebar.checkbox("💊 Dose Extraction Summary", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Script Status")

for key, info in SCRIPTS.items():
    icon, _ = check_script_exists(key)
    st.sidebar.text(f"{icon} {info['file'][:30]}...")

# ===== MAIN TITLE =====
st.title("🔬 Cellomics-5-C Analysis Pipeline")
st.markdown("*Runs the **existing Python scripts** with interactive controls*")
st.markdown("---")

# ===== PIPELINE CONFIGURATION =====
st.header("⚙️ Pipeline Configuration")
st.markdown("Provide the input file paths so every downstream step receives the correct data.")

# --- Experiment ID ---
_exp_id_input = st.text_input(
    "🔢 Experiment ID",
    value=st.session_state.experiment_id,
    help="Numeric experiment identifier (e.g. 008). Used to construct filenames like Embedding{ID}.json.",
    key="experiment_id_input"
)
st.session_state.experiment_id = _exp_id_input

# Derived filenames based on experiment ID
_MERGED_CSV_NAME = f"MergedAndFilteredExperiment{_exp_id_input}.csv"
_EMBEDDING_JSON_NAME = f"Embedding{_exp_id_input}.json"

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    _data_input = st.text_input(
        "📂 Path to summary table  *(csv or xlsx)*",
        value=st.session_state.input_data_path,
        help="Full path to the summary table produced by pybatch (e.g. summary_table.xlsx or MergedAndFilteredExperiment008.csv). Used by all pipeline stages.",
        key="input_data_path_input"
    )
with col_cfg2:
    st.markdown("")  # spacer

# --- Treatments, Dose & Control Channel ---
col_cfg3, col_cfg4, col_cfg5 = st.columns(3)
with col_cfg3:
    _treatments_input = st.text_input(
        "🏷️ Treatment labels  *(comma-separated)*",
        value=st.session_state.treatments_list,
        help="Treatment substrings to match against Experiment names (e.g. CON0, BRCACON1, …). Order matters for display.",
        key="treatments_input"
    )
with col_cfg4:
    _dose_input = st.text_input(
        "📂 Path to dose_dependency_summary CSV  *(optional)*",
        value=st.session_state.dose_csv_path,
        help="Full path to dose_dependency_summary_all_wells.csv. Leave blank to auto-detect in cell_data/ subdirectory.",
        key="dose_csv_input"
    )
with col_cfg5:
    _control_ch_input = st.text_input(
        "🔬 Control channel name",
        value=st.session_state.param_control_channel,
        help="4-letter channel code used as control (e.g. NNIR). This channel is excluded from dose-category labels so clustering only differentiates by treatment channels.",
        key="control_channel_input"
    )

# Persist into session_state
st.session_state.input_data_path    = _data_input
st.session_state.treatments_list    = _treatments_input
st.session_state.dose_csv_path      = _dose_input
st.session_state.param_control_channel = _control_ch_input

# Validate & show status
if _data_input:
    if os.path.exists(_data_input):
        st.success(f"✅ Input data file found: {os.path.basename(_data_input)}")
    else:
        st.error(f"❌ File not found: {_data_input}")
else:
    st.info("ℹ️ Enter the path to your summary table (csv or xlsx) to start the pipeline")

# --- Pipeline intermediate file status ---
with st.expander("📊 Pipeline Intermediate Files"):
    _pipeline_files = {
        "raw_all_cells.csv":                       "Step 1 → Data Preparation",
        "normalized_all_cells.csv":                 "Step 2 → Feature Selection",
        "selected_features.txt":                    "Step 2 → Feature Selection",
        "best_chronos_model_per_feature.json":      "Step 3 → Forecasting",
        "best_t5_model_per_feature.json":              "Step 3 → Forecasting",
        _EMBEDDING_JSON_NAME:                        "Step 4 → Embedding",
        "fitting_all_models.csv":                   "Step 5 → Curve Fitting",
        "clustering/Merged_Clusters_PCA_k*":        "Step 6 → Clustering",
        "ANOVA - OneWay.xlsx":                      "Step 7 → ANOVA",
        "Descriptive_Table_By_Cluster_UniqueCells.xlsx": "Step 8 → Descriptive Stats",
        "fitting/embedding_fitting_Merged_Clusters_PCA_k*": "Step 6b → Emb+Fit Clustering",
        "embedding_fitting_ANOVA - OneWay.xlsx":       "Step 7b → Emb+Fit ANOVA",
        "embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells.xlsx": "Step 8b → Emb+Fit Descriptive",
        "dose_dependency_summary_all_wells.csv":                              "Step 0 → Dose Extraction",
    }
    _cols = st.columns(3)
    for _i, (_fname, _step) in enumerate(_pipeline_files.items()):
        with _cols[_i % 3]:
            if "*" in _fname:
                if list(SCRIPT_DIR.glob(_fname)):
                    st.success(f"✅ {_fname}")
                else:
                    st.caption(f"⏳ {_fname}  \n*{_step}*")
            elif (SCRIPT_DIR / _fname).exists():
                st.success(f"✅ {_fname}")
            else:
                st.caption(f"⏳ {_fname}  \n*{_step}*")

st.markdown("---")

# ===== RUN ALL STAGES =====
# Define the ordered pipeline (main branch first, then emb+fit branch)
_PIPELINE_ORDER = [
    ("dose_summary",        enable_dose_summary),
    ("data_prep",          enable_data_prep),
    ("feature_selection",  enable_feature_selection),
    ("forecasting",        enable_forecasting),
    ("embedding",          enable_embedding),
    ("fitting",            enable_fitting),
    ("clustering",         enable_clustering),
    ("anova",              enable_anova),
    ("descriptive",        enable_descriptive),
    ("embfit_clustering",  enable_embfit_clustering),
    ("embfit_anova",       enable_embfit_anova),
    ("embfit_descriptive", enable_embfit_descriptive),
]

st.header("🚀 Run All Stages")
st.markdown("Run all **enabled** pipeline stages sequentially. Stages with existing outputs are skipped unless re-run is forced.")

col_run_all1, col_run_all2, col_run_all_stop = st.columns([2, 2, 1])
with col_run_all1:
    skip_existing = st.checkbox("⏭️ Skip stages whose outputs already exist", value=True, key="skip_existing")
with col_run_all2:
    run_all_clicked = st.button("🚀 Run All Enabled Stages", type="primary", key="run_all_stages")
with col_run_all_stop:
    st.button("🛑 Stop", on_click=_request_stop, key="stop_all")

if run_all_clicked:
    enabled_stages = [(key, SCRIPTS[key]) for key, enabled in _PIPELINE_ORDER if enabled and key in SCRIPTS]

    if not enabled_stages:
        st.warning("No stages are enabled. Enable at least one stage in the sidebar.")
    else:
        progress_bar = st.progress(0, text="Starting pipeline...")
        results_container = st.container()
        total = len(enabled_stages)
        passed = 0
        failed = 0

        for i, (key, info) in enumerate(enabled_stages):
            stage_label = STAGE_LABELS.get(key, info["name"])
            progress_bar.progress((i) / total, text=f"Running {stage_label}...")

            # Skip if outputs exist and user chose to skip
            if skip_existing and stage_outputs_exist(key):
                with results_container:
                    st.info(f"⏭️ **{stage_label}** — outputs already exist, skipped.")
                passed += 1
                continue

            # Check dependencies
            deps_ok, missing = check_pipeline_deps(key)
            if not deps_ok:
                with results_container:
                    st.error(f"🚫 **{stage_label}** — missing dependencies: {', '.join(missing)}")
                failed += 1
                continue

            # Run the script
            with results_container:
                with st.spinner(f"Running {stage_label}..."):
                    try:
                        returncode, stdout, stderr = run_script(
                            info["file"], env_vars=get_pipeline_env(key)
                        )
                        if returncode == 0:
                            st.success(f"✅ **{stage_label}** — completed.")
                            # Show generated figures for this stage
                            if STAGE_FIGURE_PATTERNS.get(key):
                                with st.expander(f"📊 Figures: {stage_label}", expanded=False):
                                    display_stage_figures(key, use_expanders=False)
                            passed += 1
                        else:
                            st.error(f"❌ **{stage_label}** — failed (code {returncode}).")
                            with st.expander(f"Output: {stage_label}"):
                                if stdout:
                                    st.text(stdout[-2000:])  # last 2000 chars
                                if stderr:
                                    st.text(stderr[-2000:])
                            failed += 1
                    except Exception as e:
                        st.error(f"💥 **{stage_label}** — error: {e}")
                        failed += 1

        progress_bar.progress(1.0, text="Pipeline complete!")
        with results_container:
            st.markdown("---")
            if failed == 0:
                st.success(f"🎉 Pipeline finished: **{passed}/{total}** stages completed successfully.")
            else:
                st.warning(f"Pipeline finished: **{passed}** passed, **{failed}** failed out of **{total}** stages.")

st.markdown("---")

# ===== TAB LAYOUT =====
tabs = st.tabs([
    "💊 Dose Summary",
    "📁 Data Prep", 
    "🎯 Features", 
    "📈 Forecast", 
    "🧬 Embed", 
    "📐 Fit", 
    "🔮 Cluster", 
    "📊 ANOVA",
    "📋 Descriptive",
    "🔗 Emb+Fit Cluster",
    "📊 Emb+Fit ANOVA",
    "📋 Emb+Fit Desc"
])

# ===== TAB 0: DOSE EXTRACTION SUMMARY =====
with tabs[0]:
    st.header("💊 Dose Extraction Summary")
    info = SCRIPTS['dose_summary']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])

    if not enable_dose_summary:
        show_skip_status('dose_summary')
    else:
        icon, exists = check_script_exists('dose_summary')

        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)

            existing_outputs = check_outputs('dose_summary')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")

            # Stage parameters
            with st.expander("⚙️ Dose Summary Parameters", expanded=False):
                st.session_state.param_dose_dir = st.text_input(
                    "📂 Directory containing dose Excel files",
                    value=st.session_state.param_dose_dir,
                    key="dose_dir",
                    help="Absolute or relative path to the folder with the normalised Excel exports. "
                         "Leave blank to search in the current working directory.")
                _c1, _c2 = st.columns(2)
                with _c1:
                    st.session_state.param_dose_excel_glob = st.text_input(
                        "Excel glob pattern",
                        value=st.session_state.param_dose_excel_glob,
                        key="dose_glob",
                        help="Filename pattern for input Excel files (e.g. Gab_Normalized_Combined_*.xlsx). "
                             "This pattern is resolved inside the directory above.")
                    st.session_state.param_dose_sheet = st.text_input(
                        "Sheet name",
                        value=st.session_state.param_dose_sheet,
                        key="dose_sheet",
                        help="Name of the sheet to read in each workbook")
                with _c2:
                    st.session_state.param_dose_prefix = st.text_input(
                        "Filename prefix",
                        value=st.session_state.param_dose_prefix,
                        key="dose_prefix",
                        help="Prefix used to derive Experiment name from the filename")
                    st.session_state.param_dose_well_map = st.text_input(
                        "Well-map JSON path",
                        value=st.session_state.param_dose_well_map,
                        key="dose_well_map",
                        help='Path to a JSON file mapping short well names to full experiment IDs. '
                             'Example contents: {"B2": "AM001100425CHR2B02293T...", "C2": "AM001100425CHR2C02293T..."}'
                    )

            # Pipeline dependency check
            deps_ok = show_dependency_status('dose_summary')

            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("💊 Run Dose Extraction", type="primary", key="run_dose_summary", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_dose_summary")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('dose_summary'))

                        if returncode == 0:
                            st.success("✅ Completed successfully!")
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

            # Preview output data
            st.subheader("Data Preview")
            csv_path = _find_file("dose_dependency_summary_all_wells.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path, nrows=100)
                st.dataframe(df, use_container_width=True)
                st.metric("Rows (showing 100)", f"{len(df)}")

# ===== TAB 1: DATA PREPARATION =====
with tabs[1]:
    st.header("📁 Data Preparation")
    info = SCRIPTS['data_prep']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_data_prep:
        show_skip_status('data_prep')
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
            
            # Stage parameters
            with st.expander("⚙️ Data Preparation Parameters", expanded=False):
                _c1, _c2 = st.columns(2)
                with _c1:
                    st.session_state.param_min_frames = st.number_input(
                        "Min frames per cell", value=st.session_state.param_min_frames,
                        min_value=1, max_value=200, step=1, key="dp_min_frames",
                        help="Cells with fewer time-frames are discarded")
                with _c2:
                    st.session_state.param_max_gap = st.number_input(
                        "Max gap in TimeIndex", value=st.session_state.param_max_gap,
                        min_value=1, max_value=50, step=1, key="dp_max_gap",
                        help="Max allowed gap between consecutive TimeIndex values")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('data_prep')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("🚀 Run Data Preparation", type="primary", key="run_data_prep", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_data_prep")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('data_prep'))
                        
                        if returncode == 0:
                            st.success("✅ Completed successfully!")
                        else:
                            st.error(f"❌ Failed (code {returncode})")
                        
                        with st.expander("📤 Output", expanded=True):
                            if stdout:
                                st.text(stdout)
                            if stderr:
                                st.error(stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            # Preview data
            st.subheader("Data Preview")
            csv_path = _find_file("raw_all_cells.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path, nrows=100)
                st.dataframe(df, width='stretch')
                st.metric("Rows (showing 100)", f"{len(df)}")

# ===== TAB 2: FEATURE SELECTION =====
with tabs[2]:
    st.header("🎯 Feature Selection")
    info = SCRIPTS['feature_selection']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_feature_selection:
        show_skip_status('feature_selection')
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
            
            # Stage parameters
            with st.expander("⚙️ Feature Selection Parameters", expanded=False):
                _c1, _c2, _c3 = st.columns(3)
                with _c1:
                    st.session_state.param_pca_variance = st.number_input(
                        "PCA explained variance", value=st.session_state.param_pca_variance,
                        min_value=0.50, max_value=1.00, step=0.01, format="%.2f", key="fs_pca_var",
                        help="Cumulative explained variance threshold for PCA")
                    st.session_state.param_corr_threshold = st.number_input(
                        "Correlation threshold", value=st.session_state.param_corr_threshold,
                        min_value=0.10, max_value=1.00, step=0.05, format="%.2f", key="fs_corr",
                        help="Spearman correlation cutoff for removing redundant features")
                with _c2:
                    st.session_state.param_const_cell_thresh = st.number_input(
                        "Constant-cell threshold", value=st.session_state.param_const_cell_thresh,
                        min_value=0.50, max_value=1.00, step=0.01, format="%.2f", key="fs_const",
                        help="Fraction of cells with zero std to consider feature constant")
                    st.session_state.param_std_keep_thresh = st.number_input(
                        "Std keep threshold", value=st.session_state.param_std_keep_thresh,
                        min_value=0.01, max_value=2.00, step=0.05, format="%.2f", key="fs_std",
                        help="Min average std to keep a feature")
                with _c3:
                    st.session_state.param_max_lag = st.number_input(
                        "Max autocorrelation lag", value=st.session_state.param_max_lag,
                        min_value=1, max_value=100, step=1, key="fs_lag",
                        help="Maximum lag for autocorrelation analysis")
                    st.session_state.param_min_features = st.number_input(
                        "Min features", value=st.session_state.param_min_features,
                        min_value=2, max_value=50, step=1, key="fs_min_feat",
                        help="Minimum number of features to try in silhouette sweep")
                    st.session_state.param_fs_clusters = st.number_input(
                        "Evaluation clusters (k)", value=st.session_state.param_fs_clusters,
                        min_value=2, max_value=20, step=1, key="fs_clusters",
                        help="K for KMeans during silhouette evaluation")
                st.session_state.param_normalize_before_variance = st.checkbox(
                    "Normalize before variance filter",
                    value=st.session_state.param_normalize_before_variance,
                    key="fs_norm_before_var",
                    help="If checked, use normalized_all_cells.csv for variance filtering")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('feature_selection')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("🎯 Run Feature Selection", type="primary", key="run_features", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_features")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('feature_selection'))
                        
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
            
            # Show generated figures for this stage
            display_stage_figures('feature_selection')

# ===== TAB 3: FORECASTING =====
with tabs[3]:
    st.header("📈 Time Series Forecasting")
    
    if not enable_forecasting:
        show_skip_status('forecasting')
    else:
        # Choose between single GPU and 4-GPU version
        gpu_mode = st.radio(
            "Select GPU Mode",
            ["4-GPU Parallel", "Single GPU"],
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
            
            # ── Forecasting Configuration GUI ──
            conf_file = SCRIPT_DIR / "my_models_conf.yaml"
            if conf_file.exists():
                from omegaconf import OmegaConf
                _conf = OmegaConf.load(str(conf_file))
                fc = _conf.forecasting

                with st.expander("⚙️ Forecasting Configuration", expanded=False):
                    st.markdown("##### General Settings")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        new_pred_len = st.number_input(
                            "Prediction length", value=int(fc.prediction_length),
                            min_value=1, max_value=50, key="conf_pred_len")
                    with col_b:
                        new_backtest_len = st.number_input(
                            "Backtest length", value=int(fc.backtest_length),
                            min_value=1, max_value=50, key="conf_bt_len")
                    with col_c:
                        new_stride = st.number_input(
                            "Stride", value=int(fc.stride),
                            min_value=1, max_value=20, key="conf_stride")

                    col_d, col_e = st.columns(2)
                    with col_d:
                        new_metric = st.selectbox(
                            "Metric", ["rmse", "mae", "mape", "smape"],
                            index=["rmse", "mae", "mape", "smape"].index(fc.metric)
                                if fc.metric in ["rmse", "mae", "mape", "smape"] else 0,
                            key="conf_metric")
                    with col_e:
                        new_limit = st.number_input(
                            "Max series (0 = all)", value=int(fc.limit_num_series),
                            min_value=0, max_value=100000, step=100, key="conf_limit")

                    st.markdown("##### Active Models")
                    st.caption("Toggle which foundation models to include in forecasting.")

                    # Group models by framework
                    all_model_names = list(_conf.models.keys())
                    frameworks = {}
                    for m in all_model_names:
                        fw = _conf.models[m].get("framework", "Other")
                        frameworks.setdefault(fw, []).append(m)

                    current_active = list(fc.active_models) if fc.active_models else []
                    new_active = []

                    for fw_name, fw_models in frameworks.items():
                        st.markdown(f"**{fw_name}**")
                        cols = st.columns(min(len(fw_models), 4))
                        for idx, model_name in enumerate(fw_models):
                            with cols[idx % len(cols)]:
                                on = st.checkbox(
                                    model_name,
                                    value=(model_name in current_active),
                                    key=f"model_{model_name}")
                                if on:
                                    new_active.append(model_name)

                    if not new_active:
                        st.error("⚠️ Select at least one model!")

                    # Show raw YAML for reference
                    with st.expander("📄 Raw YAML (read-only)"):
                        with open(conf_file, 'r') as f:
                            st.code(f.read(), language='yaml')

                    # Save button
                    config_changed = (
                        new_pred_len != int(fc.prediction_length) or
                        new_backtest_len != int(fc.backtest_length) or
                        new_stride != int(fc.stride) or
                        new_metric != fc.metric or
                        new_limit != int(fc.limit_num_series) or
                        sorted(new_active) != sorted(current_active)
                    )

                    if config_changed:
                        st.info("🔄 Configuration has been modified.")
                    if st.button("💾 Save Configuration", key="save_conf",
                                 disabled=(not config_changed or not new_active)):
                        _conf.forecasting.prediction_length = new_pred_len
                        _conf.forecasting.backtest_length = new_backtest_len
                        _conf.forecasting.stride = new_stride
                        _conf.forecasting.metric = new_metric
                        _conf.forecasting.limit_num_series = new_limit
                        _conf.forecasting.active_models = new_active
                        with open(conf_file, 'w') as f:
                            f.write(OmegaConf.to_yaml(_conf))
                        st.success("✅ Saved to my_models_conf.yaml")
                        st.rerun()
            
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
            
            # Stage parameters
            with st.expander("⚙️ Forecasting Parameters", expanded=False):
                st.session_state.param_max_cells = st.number_input(
                    "Max cells to subsample (0 = all)", value=st.session_state.param_max_cells,
                    min_value=0, max_value=100000, step=100, key="tsa_max_cells",
                    help="Number of unique cells to subsample for forecasting. Set 0 to use all cells.")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status(script_key)
            
            # Run button - different behavior for 4-GPU mode
            if gpu_mode == "4-GPU Parallel":
                st.markdown("**4-GPU mode runs `bash run_4gpus.sh` which launches 4 parallel processes:**")
                st.code("CUDA_VISIBLE_DEVICES=0,1,2,3 with SHARD_IDX=0,1,2,3", language="text")

                # --- Non-blocking 4-GPU process monitoring ---
                _4gpu_proc = st.session_state.get('_4gpu_proc')

                if _4gpu_proc is not None and _4gpu_proc.poll() is None:
                    # ── RUNNING: show live monitoring UI ──
                    _elapsed = int(time.time() - (st.session_state.get('_4gpu_start') or time.time()))
                    _h, _rem = divmod(_elapsed, 3600)
                    _m, _s = divmod(_rem, 60)
                    st.info(f"⏳ 4-GPU forecasting in progress — {_h}h {_m:02d}m {_s:02d}s elapsed")

                    _log_dir = SCRIPT_DIR / "logs"
                    _gcols = st.columns(4)
                    for _gi in range(4):
                        with _gcols[_gi]:
                            _lf = _log_dir / f"gpu{_gi}.txt"
                            if _lf.exists():
                                try:
                                    _raw = _lf.read_text()
                                    _last = _raw.replace('\r', '\n').rstrip().split('\n')[-1][:100]
                                    st.caption(f"**GPU {_gi}:** {_last}")
                                except Exception:
                                    st.caption(f"**GPU {_gi}:** (reading…)")
                            else:
                                st.caption(f"**GPU {_gi}:** (no log yet)")

                    _bc1, _bc2 = st.columns([1, 1])
                    with _bc1:
                        if st.button("🔄 Refresh Status", key="refresh_4gpu"):
                            st.rerun()
                    with _bc2:
                        if st.button("🛑 Stop 4-GPU Forecasting", key="stop_4gpu_live", type="secondary"):
                            _kill_proc(_4gpu_proc.pid)
                            _kill_gpu_orphans()
                            _4t = st.session_state.get('_4gpu_thread')
                            if _4t:
                                _4t.join(timeout=3)
                            for _sk in ('_4gpu_proc', '_4gpu_thread', '_4gpu_lines', '_4gpu_start'):
                                st.session_state[_sk] = None
                            st.session_state.running_pid = None
                            st.session_state.running_stage = None
                            st.warning("🛑 **4-GPU forecasting** was stopped by user.")
                            st.rerun()

                elif _4gpu_proc is not None:
                    # ── FINISHED: show results ──
                    _4t = st.session_state.get('_4gpu_thread')
                    if _4t:
                        _4t.join(timeout=5)
                    _4rc = _4gpu_proc.returncode
                    _4lines = st.session_state.get('_4gpu_lines') or []
                    _4stdout = "".join(_4lines)

                    for _sk in ('_4gpu_proc', '_4gpu_thread', '_4gpu_lines', '_4gpu_start'):
                        st.session_state[_sk] = None
                    st.session_state.running_pid = None
                    st.session_state.running_stage = None

                    if _4rc == 0:
                        st.success("✅ All 4 GPU processes completed!")
                    else:
                        st.error(f"❌ Failed (code {_4rc})")

                    with st.expander("📤 Output", expanded=True):
                        if _4stdout:
                            st.text(_4stdout[-10000:])

                    logs_dir = SCRIPT_DIR / "logs"
                    if logs_dir.exists():
                        st.subheader("GPU Logs")
                        for i in range(4):
                            log_file = logs_dir / f"gpu{i}.txt"
                            if log_file.exists():
                                with st.expander(f"📋 GPU {i} Log"):
                                    with open(log_file, 'r') as f:
                                        st.text(f.read()[-5000:])

                else:
                    # ── NOT RUNNING: show Run button ──
                    if st.button("🚀 Run 4-GPU Forecasting", type="primary", key="run_forecast", disabled=not deps_ok):
                        _kill_gpu_orphans()
                        try:
                            shell_script = SCRIPT_DIR / "run_4gpus.sh"
                            _env = os.environ.copy()
                            _env.update(get_pipeline_env(script_key))
                            _kwargs = dict(
                                cwd=str(SCRIPT_DIR),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                env=_env,
                            )
                            if sys.platform != "win32":
                                _kwargs["preexec_fn"] = os.setsid
                            proc = subprocess.Popen(
                                ["bash", str(shell_script)],
                                **_kwargs,
                            )
                            _lines: list[str] = []

                            def _r4():
                                try:
                                    for line in proc.stdout:
                                        print(line, end="", flush=True)
                                        _lines.append(line)
                                except ValueError:
                                    pass

                            _t4 = threading.Thread(target=_r4, daemon=True)
                            _t4.start()
                            st.session_state['_4gpu_proc'] = proc
                            st.session_state['_4gpu_thread'] = _t4
                            st.session_state['_4gpu_lines'] = _lines
                            st.session_state['_4gpu_start'] = time.time()
                            st.session_state.running_pid = proc.pid
                            st.session_state.running_stage = "run_4gpus.sh"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error starting 4-GPU forecasting: {e}")
            else:
                _rc, _sc = st.columns([3, 1])
                with _rc:
                    _clicked = st.button("📈 Run Forecasting", type="primary", key="run_forecast", disabled=not deps_ok)
                with _sc:
                    st.button("🛑 Stop", on_click=_request_stop, key="stop_forecast")
                if _clicked:
                    with st.spinner(f"Running {info['file']}... (this may take a long time)"):
                        try:
                            returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env(script_key))
                            
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
with tabs[4]:
    st.header("🧬 Embedding Generation")
    
    if not enable_embedding:
        show_skip_status('embedding')
    else:
        # Choose between single GPU and multi-GPU version
        gpu_mode_embed = st.radio(
            "Select GPU Mode",
            ["Multi-GPU Parallel", "Single GPU"],
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
            st.dataframe(models_df, width='stretch', hide_index=True)
            
            existing_outputs = check_outputs(script_key)
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
            
            st.warning("⚠️ This may take a long time to run!")
            
            # Stage parameters
            with st.expander("⚙️ Embedding Parameters", expanded=False):
                st.session_state.param_umap_dim = st.number_input(
                    "UMAP dimensions", value=st.session_state.param_umap_dim,
                    min_value=2, max_value=10, step=1, key="emb_umap_dim",
                    help="Number of UMAP output dimensions for embeddings")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status(script_key)
            
            # Multi-GPU specific options
            if gpu_mode_embed == "Multi-GPU Parallel":
                st.subheader("Multi-GPU Configuration")
                st.info("💡 The script automatically distributes features across GPUs using multiprocessing")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_gpus = st.number_input("Number of GPUs", value=4, min_value=1, max_value=8, key="embed_num_gpus")
                with col2:
                    umap_dim = st.number_input("UMAP Dimensions", value=st.session_state.param_umap_dim, min_value=2, max_value=10, key="embed_umap_dim")
                
                # --- Non-blocking multi-GPU embedding monitoring ---
                _mgpu_proc = st.session_state.get('_mgpu_embed_proc')

                if _mgpu_proc is not None and _mgpu_proc.poll() is None:
                    # ── RUNNING: show live monitoring UI ──
                    _elapsed = int(time.time() - (st.session_state.get('_mgpu_embed_start') or time.time()))
                    _h, _rem = divmod(_elapsed, 3600)
                    _m, _s = divmod(_rem, 60)
                    st.info(f"⏳ Multi-GPU embedding in progress — {_h}h {_m:02d}m {_s:02d}s elapsed")

                    _em_lines = st.session_state.get('_mgpu_embed_lines') or []
                    if _em_lines:
                        st.caption(f"Last output: {_em_lines[-1].strip()[:120]}")

                    _bc1, _bc2 = st.columns([1, 1])
                    with _bc1:
                        if st.button("🔄 Refresh Status", key="refresh_mgpu_embed"):
                            st.rerun()
                    with _bc2:
                        if st.button("🛑 Stop Multi-GPU Embedding", key="stop_mgpu_embed_live", type="secondary"):
                            _kill_proc(_mgpu_proc.pid)
                            _kill_gpu_orphans()
                            _et = st.session_state.get('_mgpu_embed_thread')
                            if _et:
                                _et.join(timeout=3)
                            for _sk in ('_mgpu_embed_proc', '_mgpu_embed_thread', '_mgpu_embed_lines', '_mgpu_embed_start'):
                                st.session_state[_sk] = None
                            st.session_state.running_pid = None
                            st.session_state.running_stage = None
                            st.warning(f"🛑 **{info['file']}** was stopped by user.")
                            st.rerun()

                elif _mgpu_proc is not None:
                    # ── FINISHED: show results ──
                    _et = st.session_state.get('_mgpu_embed_thread')
                    if _et:
                        _et.join(timeout=5)
                    _erc = _mgpu_proc.returncode
                    _em_lines = st.session_state.get('_mgpu_embed_lines') or []
                    _estdout = "".join(_em_lines)

                    for _sk in ('_mgpu_embed_proc', '_mgpu_embed_thread', '_mgpu_embed_lines', '_mgpu_embed_start'):
                        st.session_state[_sk] = None
                    st.session_state.running_pid = None
                    st.session_state.running_stage = None

                    if _erc == 0:
                        st.success("✅ Completed!")
                    else:
                        st.error(f"❌ Failed (code {_erc})")

                    with st.expander("📤 Output", expanded=True):
                        if _estdout:
                            st.text(_estdout[-10000:])

                else:
                    # ── NOT RUNNING: show Run button ──
                    if st.button("🧬 Run Multi-GPU Embedding", type="primary", key="run_embed", disabled=not deps_ok):
                        _kill_gpu_orphans()
                        try:
                            _env = os.environ.copy()
                            _env.update(get_pipeline_env(script_key))
                            _kwargs = dict(
                                cwd=str(SCRIPT_DIR),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                env=_env,
                            )
                            if sys.platform != "win32":
                                _kwargs["preexec_fn"] = os.setsid
                            proc = subprocess.Popen(
                                [sys.executable, "-u", str(SCRIPT_DIR / info['file']),
                                 f"--num_gpus={num_gpus}", f"--dim={umap_dim}", "--verbose"],
                                **_kwargs,
                            )
                            _lines: list[str] = []

                            def _rem():
                                try:
                                    for line in proc.stdout:
                                        print(line, end="", flush=True)
                                        _lines.append(line)
                                except ValueError:
                                    pass

                            _tem = threading.Thread(target=_rem, daemon=True)
                            _tem.start()
                            st.session_state['_mgpu_embed_proc'] = proc
                            st.session_state['_mgpu_embed_thread'] = _tem
                            st.session_state['_mgpu_embed_lines'] = _lines
                            st.session_state['_mgpu_embed_start'] = time.time()
                            st.session_state.running_pid = proc.pid
                            st.session_state.running_stage = info['file']
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error starting multi-GPU embedding: {e}")
            else:
                _rc, _sc = st.columns([3, 1])
                with _rc:
                    _clicked = st.button("🧬 Run Embedding", type="primary", key="run_embed", disabled=not deps_ok)
                with _sc:
                    st.button("🛑 Stop", on_click=_request_stop, key="stop_embed")
                if _clicked:
                    with st.spinner(f"Running {info['file']}..."):
                        try:
                            returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env(script_key))
                            
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
with tabs[5]:
    st.header("📐 Curve Fitting")
    info = SCRIPTS['fitting']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_fitting:
        show_skip_status('fitting')
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
            st.dataframe(models_df, width='stretch', hide_index=True)
            
            existing_outputs = check_outputs('fitting')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                if "fitting_all_models.csv" in existing_outputs:
                    st.warning(
                        "⚠️ **fitting_all_models.csv** already exists. "
                        "The script will **skip** the heavy curve-fitting step and only "
                        "regenerate the JSON exports. To re-run fitting from scratch, "
                        "delete `fitting_all_models.csv` first."
                    )
                
                for out in existing_outputs:
                    if out.endswith('.csv'):
                        csv_path = SCRIPT_DIR / out
                        with st.expander(f"📊 {out}"):
                            df = pd.read_csv(csv_path, nrows=100)
                            st.dataframe(df, width='stretch')
            
            # Stage parameters
            with st.expander("⚙️ Curve Fitting Parameters", expanded=False):
                _c1, _c2 = st.columns(2)
                with _c1:
                    st.session_state.param_maxfev = st.number_input(
                        "Max function evaluations (maxfev)", value=st.session_state.param_maxfev,
                        min_value=1000, max_value=500000, step=1000, key="fit_maxfev",
                        help="Maximum iterations for scipy curve_fit per model")
                with _c2:
                    st.session_state.param_pval_threshold = st.number_input(
                        "P-value significance threshold", value=st.session_state.param_pval_threshold,
                        min_value=0.001, max_value=0.50, step=0.01, format="%.3f", key="fit_pval",
                        help="Fits with p-value above this are discarded")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('fitting')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("📐 Run Curve Fitting", type="primary", key="run_fit", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_fit")
            if _clicked:
                with st.spinner(f"Running {info['file']}... (this may take a while)"):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('fitting'))
                        
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

            # Show generated figures for this stage
            display_stage_figures('fitting')

# ===== TAB 6: CLUSTERING =====
with tabs[6]:
    st.header("🔮 Unsupervised Clustering")
    info = SCRIPTS['clustering']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_clustering:
        show_skip_status('clustering')
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

                # Show cluster assignments CSVs
                clustering_dir = SCRIPT_DIR / "clustering"
                if clustering_dir.exists():
                    for csv_file in sorted(clustering_dir.glob("cluster_assignments_k*.csv")):
                        with st.expander(f"📊 {csv_file.name}"):
                            df = pd.read_csv(csv_file)
                            st.dataframe(df.head(100), width='stretch')

                            # Cluster distribution
                            fig, ax = plt.subplots()
                            df['Cluster'].value_counts().plot(kind='bar', ax=ax)
                            ax.set_xlabel('Cluster')
                            ax.set_ylabel('Count')
                            ax.set_title('Cluster Distribution')
                            st.pyplot(fig)
            
            # Stage parameters
            with st.expander("⚙️ Clustering Parameters", expanded=False):
                _c1, _c2, _c3 = st.columns(3)
                with _c1:
                    st.session_state.param_k_min = st.number_input(
                        "K min", value=st.session_state.param_k_min,
                        min_value=2, max_value=20, step=1, key="cl_k_min",
                        help="Minimum k for silhouette sweep")
                    st.session_state.param_k_max = st.number_input(
                        "K max", value=st.session_state.param_k_max,
                        min_value=2, max_value=50, step=1, key="cl_k_max",
                        help="Maximum k for silhouette sweep")
                with _c2:
                    st.session_state.param_kmeans_n_init = st.number_input(
                        "KMeans n_init", value=st.session_state.param_kmeans_n_init,
                        min_value=1, max_value=100, step=1, key="cl_n_init",
                        help="Number of KMeans initializations")
                    st.session_state.param_kmeans_seed = st.number_input(
                        "Random seed", value=st.session_state.param_kmeans_seed,
                        min_value=0, max_value=99999, step=1, key="cl_seed",
                        help="Random state for reproducibility")
                with _c3:
                    st.session_state.param_pca_components = st.number_input(
                        "PCA components (vis)", value=st.session_state.param_pca_components,
                        min_value=2, max_value=10, step=1, key="cl_pca",
                        help="Number of PCA components for visualization")
                    st.session_state.param_num_best_k = st.number_input(
                        "Best k values to keep", value=st.session_state.param_num_best_k,
                        min_value=1, max_value=5, step=1, key="cl_best_k",
                        help="Number of top-k values to process")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('clustering')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("🔮 Run Clustering", type="primary", key="run_cluster", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_cluster")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('clustering'))
                        
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

            # Show all generated figures for this stage
            display_stage_figures('clustering')

# ===== TAB 7: ANOVA =====
with tabs[7]:
    st.header("📊 ANOVA Analysis")
    info = SCRIPTS['anova']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_anova:
        show_skip_status('anova')
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
                        st.dataframe(df, width='stretch')
                    except:
                        df = pd.read_excel(xlsx_path)
                        st.dataframe(df, width='stretch')
            
            # Stage parameters
            with st.expander("⚙️ ANOVA Parameters", expanded=False):
                st.session_state.param_anova_pval = st.number_input(
                    "P-value highlight threshold", value=st.session_state.param_anova_pval,
                    min_value=0.001, max_value=0.50, step=0.01, format="%.3f", key="anova_pval",
                    help="P-values below this threshold are highlighted in blue in the Excel output")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('anova')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("📊 Run ANOVA", type="primary", key="run_anova", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_anova")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('anova'))
                        
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
with tabs[8]:
    st.header("📋 Descriptive Statistics")
    info = SCRIPTS['descriptive']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_descriptive:
        show_skip_status('descriptive')
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
                    st.dataframe(df, width='stretch')
            
            # Stage parameters
            with st.expander("⚙️ Descriptive Stats Parameters", expanded=False):
                st.session_state.param_ci_zscore = st.number_input(
                    "CI z-score", value=st.session_state.param_ci_zscore,
                    min_value=1.00, max_value=4.00, step=0.01, format="%.2f", key="desc_ci",
                    help="z-score multiplier for confidence interval (1.96 = 95% CI, 2.576 = 99% CI)")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('descriptive')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("📋 Run Descriptive Stats", type="primary", key="run_desc", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_desc")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('descriptive'))
                        
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

# ===== TAB 9: EMB+FIT CLUSTERING =====
with tabs[9]:
    st.header("🔗 Embedding + Fitting — Clustering")
    info = SCRIPTS['embfit_clustering']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_embfit_clustering:
        show_skip_status('embfit_clustering')
    else:
        icon, exists = check_script_exists('embfit_clustering')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            existing_outputs = check_outputs('embfit_clustering')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")

                # Show cluster assignments CSVs
                fitting_dir = SCRIPT_DIR / "fitting"
                if fitting_dir.exists():
                    for csv_file in sorted(fitting_dir.glob("embedding_fitting_Merged_Clusters_PCA_k*.csv")):
                        with st.expander(f"📊 {csv_file.name}"):
                            df = pd.read_csv(csv_file, nrows=100)
                            st.dataframe(df, width='stretch')
            
            # Stage parameters (shared with Clustering tab)
            with st.expander("⚙️ Emb+Fit Clustering Parameters", expanded=False):
                st.caption("These settings are shared with the Clustering tab. Changing them here applies to both.")
                _c1, _c2, _c3 = st.columns(3)
                with _c1:
                    st.session_state.param_k_min = st.number_input(
                        "K min", value=st.session_state.param_k_min,
                        min_value=2, max_value=20, step=1, key="ef_k_min",
                        help="Minimum k for silhouette sweep")
                    st.session_state.param_k_max = st.number_input(
                        "K max", value=st.session_state.param_k_max,
                        min_value=2, max_value=50, step=1, key="ef_k_max",
                        help="Maximum k for silhouette sweep")
                with _c2:
                    st.session_state.param_kmeans_n_init = st.number_input(
                        "KMeans n_init", value=st.session_state.param_kmeans_n_init,
                        min_value=1, max_value=100, step=1, key="ef_n_init",
                        help="Number of KMeans initializations")
                    st.session_state.param_kmeans_seed = st.number_input(
                        "Random seed", value=st.session_state.param_kmeans_seed,
                        min_value=0, max_value=99999, step=1, key="ef_seed",
                        help="Random state for reproducibility")
                with _c3:
                    st.session_state.param_pca_components = st.number_input(
                        "PCA components (vis)", value=st.session_state.param_pca_components,
                        min_value=2, max_value=10, step=1, key="ef_pca",
                        help="Number of PCA components for visualization")
                    st.session_state.param_num_best_k = st.number_input(
                        "Best k values to keep", value=st.session_state.param_num_best_k,
                        min_value=1, max_value=5, step=1, key="ef_best_k",
                        help="Number of top-k values to process")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('embfit_clustering')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("🔗 Run Emb+Fit Clustering", type="primary", key="run_embfit_cluster", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_embfit_cluster")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('embfit_clustering'))
                        
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

            # Show all generated figures for this stage
            display_stage_figures('embfit_clustering')

# ===== TAB 10: EMB+FIT ANOVA =====
with tabs[10]:
    st.header("📊 Embedding + Fitting — ANOVA")
    info = SCRIPTS['embfit_anova']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_embfit_anova:
        show_skip_status('embfit_anova')
    else:
        icon, exists = check_script_exists('embfit_anova')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            existing_outputs = check_outputs('embfit_anova')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                
                xlsx_path = SCRIPT_DIR / "embedding_fitting_ANOVA - OneWay.xlsx"
                if xlsx_path.exists():
                    try:
                        df = pd.read_excel(xlsx_path, header=[0,1])
                        st.dataframe(df, width='stretch')
                    except:
                        df = pd.read_excel(xlsx_path)
                        st.dataframe(df, width='stretch')
            
            # Stage parameters
            with st.expander("⚙️ Emb+Fit ANOVA Parameters", expanded=False):
                st.session_state.param_anova_pval = st.number_input(
                    "P-value highlight threshold", value=st.session_state.param_anova_pval,
                    min_value=0.001, max_value=0.50, step=0.01, format="%.3f", key="ef_anova_pval",
                    help="P-values below this threshold are highlighted in blue")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('embfit_anova')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("📊 Run Emb+Fit ANOVA", type="primary", key="run_embfit_anova", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_embfit_anova")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('embfit_anova'))
                        
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

# ===== TAB 11: EMB+FIT DESCRIPTIVE =====
with tabs[11]:
    st.header("📋 Embedding + Fitting — Descriptive Statistics")
    info = SCRIPTS['embfit_descriptive']
    st.markdown(f"**Script:** `{info['file']}`")
    st.info(info['description'])
    
    if not enable_embfit_descriptive:
        show_skip_status('embfit_descriptive')
    else:
        icon, exists = check_script_exists('embfit_descriptive')
        
        if not exists:
            st.error(f"❌ Script not found: {info['file']}")
        else:
            with st.expander("📜 View Script Source Code"):
                with open(SCRIPT_DIR / info['file'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='python', line_numbers=True)
            
            existing_outputs = check_outputs('embfit_descriptive')
            if existing_outputs:
                st.success(f"✅ Existing outputs: {', '.join(existing_outputs)}")
                
                xlsx_path = SCRIPT_DIR / "embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells.xlsx"
                if xlsx_path.exists():
                    df = pd.read_excel(xlsx_path)
                    st.dataframe(df, width='stretch')
            
            # Stage parameters
            with st.expander("⚙️ Emb+Fit Descriptive Parameters", expanded=False):
                st.session_state.param_ci_zscore = st.number_input(
                    "CI z-score", value=st.session_state.param_ci_zscore,
                    min_value=1.00, max_value=4.00, step=0.01, format="%.2f", key="ef_desc_ci",
                    help="z-score multiplier for confidence interval (1.96 = 95% CI, 2.576 = 99% CI)")

            # Pipeline dependency check — blocks run if deps missing
            deps_ok = show_dependency_status('embfit_descriptive')
            
            _rc, _sc = st.columns([3, 1])
            with _rc:
                _clicked = st.button("📋 Run Emb+Fit Descriptive Stats", type="primary", key="run_embfit_desc", disabled=not deps_ok)
            with _sc:
                st.button("🛑 Stop", on_click=_request_stop, key="stop_embfit_desc")
            if _clicked:
                with st.spinner(f"Running {info['file']}..."):
                    try:
                        returncode, stdout, stderr = run_script(info['file'], env_vars=get_pipeline_env('embfit_descriptive'))
                        
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
| 0 | `build_dose_summary.py` | Dose extraction summary from Excel exports |
| 1 | `make_raw_all_cells_from_pybatch.py` | Data preparation & filtering |
| 2 | `feature_selection.py` | PCA-based feature selection |
| 3 | `TSA_analysis.py` | Time series forecasting (single GPU) |
| 3b | `TSA_analysis_4gpu.py` + `run_4gpus.sh` | **4-GPU parallel forecasting** |
| 4 | `Embedding.py` | Chronos embeddings + UMAP (single GPU) |
| 4b | `Embedding_multi_gpu.py` | **Multi-GPU parallel embedding** |
| 5 | `fit_cell_trajectory.py` | Curve fitting (12 models) |
| 6 | `embedding_unsupervised_clustering.py` | K-Means clustering (embedding only) |
| 6b | `embedding_fitting_unsupervised_clustering.py` | K-Means clustering (embedding + fitting) |
| 7 | `ANOVA.py` | One-way ANOVA by cluster |
| 7b | `embedding_fitting_anova.py` | One-way ANOVA (embedding + fitting) |
| 8 | `descriptive_table_by_cluster.py` | Descriptive statistics |
| 8b | `embedding_fitting_descriptive_table.py` | Descriptive stats (embedding + fitting) |

**To run:** `streamlit run app.py`

*Built with Streamlit • Cellomics-5-C Analysis Pipeline*
""")
