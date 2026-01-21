# T_IL-Cellomics-5-C

## Investigating Dynamic Features of Cell Motility and Morphology in BRCA1-Knockout Cells

This project analyzes the **morphokinetic behavior of BRCA1-knockout breast cancer cells** using live-cell imaging data. We aim to identify patterns in cell movement and morphology over time that could serve as dynamic markers for metastasis or treatment response.

Our goal is to explore whether modeling the **temporal dynamics** of morphokinetic features can uncover patterns that are not evident from static summary statistics. To this end, we first select the best-fitting model for each feature based on its forecasting performance (measured using RMSE), and then use its forecast as input for downstream embedding and clustering.

---

## Data Overview

The core input dataset used throughout this project is `raw_all_cells.csv`, which contains all cells from the Experiment 008 entries listed in `Hala_reports.xlsx` after preprocessing into a single-cell time series file.

Each row represents a single frame of a tracked cell, and includes:

- `unique_id` – Concatenation of `Parent` and `Experiment` (serves as unique cell identifier)
- `ds` – Frame timestamp in datetime format, derived from `TimeIndex` and `dt`
- Metadata columns:
  - `Experiment`, `Parent`, `TimeIndex`, `dt`
- 49 morpho-kinetic features (e.g., `Velocity_X`, `Directional_Change`, `Displacement2`, etc.)

Before generating this file, we apply the following quality filters:
- Each cell must have at least **25 valid frames**
- No frame gaps larger than **5 consecutive timepoints** are allowed

This file is used as input across all major pipeline components, including feature selection, forecasting, embedding, and function fitting.

---

## Feature Preselection for Forecasting

Not all features are suitable for time-series modeling.  
To focus the forecasting process on meaningful temporal patterns, we apply a **feature selection step** prior to modeling.

Main script: `feature_selection.py`

### Input

- `MergedAndFilteredExperiment008.csv` – Time-series dataset after initial filtering for minimum cell length and maximum frame gap, containing 49 features and metadata per timepoint.

### Functionality

1. **Load** the input data
2. **For each feature**:
   - Check if it has sufficient variance across time and cells
   - Filter out features that are constant, nearly constant, or have insufficient temporal dynamics
3. **Export** the list of selected features for use in forecasting

### Output

- `raw_all_cells.csv` – Reformatted version of the input file, with added columns `ds` (datetime) and `unique_id` (cell ID), used as the standardized input across all downstream modeling and analysis components.
- `selected_features.txt` – A text file listing the names of features that passed filtering

These features are then used as input to the forecasting pipeline (`TSA_analysis.py`).

---

## Forecasting with Foundation Models (MMF)

We use a modified version of the [Many Model Forecasting (MMF)](https://github.com/databricks-industry-solutions/many-model-forecasting) framework to forecast selected morphokinetic features over time using multiple foundation models.

Main script: `TSA_analysis.py`

### Input

- `raw_all_cells.csv` – Unified time-series dataset (see output of `feature_selection.py`)
- `selected_features.txt` – List of informative features (see output of `feature_selection.py`)

### Functionality

The following process was applied on a subset of 500 individual cells:

1. **Load** the input data and selected feature list
2. **For each selected feature**:
   - Extract its time series per cell
   - Run forecasts using multiple foundation models (TimesFM, Moirai, Chronos, etc.)
   - Evaluate performance using **RMSE** on the last 5 timepoints
3. **Select** the best-performing model per feature based on lowest RMSE
4. **Store** model-specific outputs for later use (e.g. embeddings, clustering)

### Output

- Forecasts: `results/{model_name}_forecast.csv`
- Evaluation metrics: `results/{model_name}_metrics.csv`
- Final summary of best model per feature: `best_model_per_feature.csv`
- Full JSON results of best model per feature: `best_model_per_feature.json`
- Best Chronos model per feature (used for embeddings): `best_chronos_model_per_feature.json`

---

## Embedding Pipeline
This repository provides a complete pipeline for generating low-dimensional embeddings of time-series features extracted from single-cell tracking experiments. The pipeline is designed to support time-series models, including Amazon's Chronos, Salesforce's Moirai, and Google's TimesFM.

Main script: `Embedding.py`

### Input
- A CSV file with columns:
`Experiment`, `Parent`, `TimeIndex`, `dt`, `[feature_1, feature_2, ..., feature_n]`
- A JSON file:  
  `best_chronos_model_per_feature.json` – the mapping between each feature and its best Chronos model (see output of `TSA_analysis.py`)
 
  > We originally intended to use `best_model_per_feature.json`, which includes the overall best model (**Chronos**, **Moirai**, or **TimesFM**) per feature based on forecasting accuracy. However, due to restricted embedding access for **Moirai** and **TimesFM**, we currently use the Chronos-only version.

  
### Functionality
1. **Load preprocessed time-series input** (e.g. `MergedAndFilteredExperiment008.csv` or `raw_all_cells.csv`)
2. **For each selected feature in `best_chronos_model_per_feature.json`:**
   - Extract its full time series per cell
   - Generate embeddings using a pretrained model (currently supports Chronos only)
   - Apply UMAP to reduce embedding dimensionality (default: 3)
3. **For all other features not in the dictionary:**
   - Compute the mean value across time per cell
   - Use the scalar mean as a 1D embedding
4. **Export all cell embeddings into a single structured JSON file**

### Output
- JSON file (e.g. `Embedding008.json`) structured as:
{
  `"Experiment"`: "EXPERIMENT_ID",
  `"Parent"`: "CELL_ID",
  `"Acceleration"`: [1.23, -0.44, 2.88],
  `"Sphericity"`: [0.77, 1.01, -0.65],
  ...}
- Each entry contains one embedding vector per feature per cell

### Note
Only Chronos models are currently supported. Embedding access for Moirai and TimesFM models is not yet available via HuggingFace inference APIs.

---

## Unsupervised Clustering – Embedding-Based PCA + KMeans

This script performs unsupervised clustering of single-cell time-series data using precomputed feature embeddings. The goal is to reduce the dimensionality of the embeddings, apply clustering (KMeans), and merge the cluster assignments back to the original dataset for further downstream analysis.

Main script: `Unsupervised Clustering - Embedding - K=3.py`

### Input

- **Embedding JSON file**
- **Original experiment CSV file** (e.g. `MergedAndFilteredExperiment008.csv`)  
  Contains repeated rows per cell across time, with fields such as: `Experiment`, `Parent`, `TimeIndex`, `dt`, feature values...

### Functionality

1. **Load and parse embedding vectors** per cell from the JSON.
2. **Assign treatment labels** based on `Experiment` name (e.g., `CON0`, `BRCACON1`, etc.).
3. **Standardize** the embedding matrix using `StandardScaler`.
4. **Reduce dimensionality** to 2D using PCA.
5. **Apply KMeans clustering** with `k=3`.  
   - *Note*: `k=3` was selected based on the highest silhouette score in prior analyses.
6. **Visualize**:
   - Scatter plot of PCA embeddings colored by cluster
   - 6 subplots showing PCA clusters split by treatment
7. **Save** the clustering results (`PC1`, `PC2`, `cluster label`, `treatment`) to `cluster_assignments_k3.csv`.
8. **Merge** the cluster assignments back into the original time-series dataset by `Experiment` and `Parent`.
9. **Export** the merged dataframe as `Merged_Clusters_PCA.csv`.

### Output

- `cluster_assignments_k3.csv` – one row per cell, with PCA coordinates and cluster label  
- `pca_kmeans_k3_clusters.png` – PCA plot showing all clusters  
- `pca_kmeans_k3_by_treatment.png` – PCA plots separated by treatment  
- `Merged_Clusters_PCA.csv` – original dataset with appended `PC1`, `PC2`, `Cluster`, and `Treatment`

---

## Descriptive Table by Cluster

This module prepares descriptive statistics for each cluster as a **preliminary step toward classifying clusters based on their feature profiles**. After unsupervised clustering of cell embeddings, this step summarizes each cluster’s biological characteristics to help identify discriminative features.

Main script: `Descriptive Table by Cluster.py`

### Input
- `Merged_Clusters_PCA.csv`: A merged file containing:
  - Original time-series features
  - Cluster assignments (`Cluster`)
  - PCA components (`PC1`, `PC2`)
  - Treatment labels (e.g. `BRCACON1`, `CON0`)

### Functionality
1. **Filter unique cells**: Averages features across all timepoints for each unique cell (identified by `Experiment` and `Parent`).
2. **Group by cluster** to compute descriptive statistics for each feature:
   - Mean
   - Standard Deviation (Std)
   - Standard Error (SE)
   - 95% Confidence Interval (CI)
3. **Outputs a summary table** with one row per cluster, and statistics for all features.

### Output
- `Descriptive_Table_By_Cluster_UniqueCells.xlsx`: A spreadsheet summarizing feature statistics per cluster.

---

## ANOVA – OneWay

This script performs **one-way ANOVA** on time-series-derived features to identify which features significantly differ between clusters. It uses cluster labels from prior unsupervised clustering (e.g. KMeans on PCA embeddings) to group cells and test for statistical differences.

Main script: `ANOVA.py`

### Input

`Merged_Clusters_PCA.csv`: A merged dataset containing original features, PCA components, and cluster assignments per cell.

### Functionality
1. **Aggregate per unique cell** by averaging all timepoint rows (based on `Experiment`, `Parent`, and `Cluster`).
2. **Select features** for analysis (excluding metadata such as `PC1`, `Treatment`, etc.).
3. **Perform one-way ANOVA**:
   - Compares the means of each feature across clusters.
   - Calculates:
     - Sum of Squares (Between / Within / Total)
     - Degrees of Freedom
     - Mean Squares
     - **F-statistic** and **p-value**
4. **Style the results**:
   - P-values below 0.05 are highlighted in **blue**.
   - Output uses **multi-level headers** for clarity (e.g. "Between Groups", "Within Groups").
5. **Export** to a formatted Excel file.

### Output
- `ANOVA - OneWay.xlsx`: Multi-level Excel table with:
  - Rows = Features  
  - Columns = ANOVA statistics (organized by source of variance)

### Example Columns
| Feature | Between Groups | | | | Within Groups | | | Total | |
|---------|----------------|--|--|--|----------------|--|--|--------|--|
|         | Sum of Squares | df | Mean Square | F statistic | Sum of Squares | df | Mean Square | Sum of Squares | df |

P-values < 0.05 are **automatically styled in blue**.

---

## Two-Way ANOVA and Feature Interpretation for KMeans Clusters

This analysis module compares the **current project’s clusters** to those from **a previous project based on static (mean-based) feature profiles**, using two-way ANOVA and interpretive visualizations to identify features with consistent or divergent behavior between the two strategies.

Main scripts:
- `K3_analysis.py`
- `K3_graphs.py`

### Input

- `cluster_assignments_k3.csv` – Cluster labels and PCA coordinates (see output of `Unsupervised Clustering – Embedding-Based PCA + KMeans`)
- `Descriptive_Table_By_Cluster_UniqueCells.xlsx` – Summary table with per-cluster feature statistics (mean, std, CI) (see output of `Descriptive Table by Cluster`)
- `ANOVA - OneWay.xlsx` – One-way ANOVA results for each feature across clusters (see output of `ANOVA – OneWay`)
- `rawdatagraph.csv` – Cluster-level features from the previous project (mean-based profiles)
- `Merged_Clusters_PCA.csv` – Cluster-level features from the current project
- `linessplit_element0.csv` – Cluster assignments for the previous project 

### Functionality

#### `K3_analysis.py`

1. **Aggregate cluster-level means** from both current and previous projects
2. **Perform two-way ANOVA** per feature (`Analysis × Cluster`)
3. **Run Tukey HSD post-hoc tests** to compare group-level means
4. **Export statistical tables** with FDR-adjusted p-values
5. **Visualize p-values** as a feature × cluster heatmap

#### `K3_graphs.py`

1. **Normalize feature means** for each group relative to Group 0 (G0)
2. **Merge with ANOVA results** and categorize each feature:
   - Morphological vs. Kinetic
3. **Generate visual summaries**:
   - Barplots of normalized feature levels per group
   - Z-score heatmaps for all, morphological, and kinetic features

### Output

- `anova_two_way.csv` – Two-way ANOVA summary (per feature)
- `tukey_two_way.csv` – Tukey HSD post-hoc results (per feature × cluster)
- `old_populations_means.csv`, `new_populations_means.csv` – Aggregated means for previous vs. current project
- `tukey_pvalue_heatmap.png` – Heatmap of Tukey-adjusted p-values
- `Normalized_Feature_Ratios.csv` – Feature values normalized to Group 0
- `Cell_Distribution_Per_Treatment.png` – Barplot of group assignments by treatment
- `Heatmap_All_Features_ZScore.png` – Z-score heatmap for all features
- `Heatmap_Morphological_Features_ZScore.png` – Z-score heatmap for morphological features
- `Heatmap_Kinetic_Features_ZScore.png` – Z-score heatmap for kinetic features
- `Normalized_Kinetic_Part_*.png`, `Normalized_Morph_Part_*.png` – Multi-part barplots by feature category

---

## Function Fitting for Feature Characterization

In addition to forecasting-based modeling, a complementary approach was used to characterize the temporal behavior of each feature by fitting predefined mathematical functions — including linear, exponential, sigmoid, and others — to its time series per cell. This method provides a structured and parameter-based alternative to deep learning-derived embeddings, offering compact representations of temporal patterns.

Main script: `fit_cell_trajectory.py`

### Input

- `raw_all_cells.csv` – Unified time-series dataset (see output of `feature_selection.py`)

### Functionality

1. For each cell-feature:
   - Fit 12 predefined functions
   - Evaluate model quality using RMSE and p-value
   - Retain only statistically significant fits (p < 0.05)
2. For each `(Experiment, Parent, Feature)`:
   - Select top 3 models based on normalized RMSE (NRMSE)
   - Select the best-fitting model with the lowest NRMSE
3. For each cell:
   - Represent each feature using the parameters of the best-fitting model
   - Apply log transformation and normalization to the resulting vectors
4. Generate parameter-based vectors and integrate with MMF-based representations
5. Apply full analysis pipeline (dimensionality reduction, clustering, statistical testing) to the combined vectors

### Output

- `fitting_all_models.csv` – Full results for all models and features  
- `fitting_significant_models.csv` – Statistically significant fits only  
- `fitting_top3_with_nrmse.csv` – Top 3 models per cell-feature (by NRMSE)  
- `fitting_best_with_nrmse.csv` – Best model per cell-feature (by NRMSE)  
- `fitting_best_model_log.json` – Embedding vectors from best model (log-transformed)  
- `fitting_top3_models_log.json` – Embedding vectors from top 3 models (log-transformed)  
- `fitting_best_model_log_scaled.json` – Embeddings from best model (log + normalized)  
- `fitting_top3_models_log_scaled.json` – Embeddings from top 3 models (log + normalized)

This integrated strategy enabled a more comprehensive representation of each cell’s morphokinetic profile and allowed assessment of whether combining multiple temporal modeling approaches improves the identification of biologically meaningful subgroups.


All downstream steps in the analysis pipeline (dimensionality reduction, clustering, descriptive statistics, and ANOVA) were repeated using the combined embedding + fitting vectors.  
**Scripts**:  
- `Embedding and Fitting - Unsupervised Clustering.py`  
- `Embedding and Fitting - descriptive table by cluster.py`  
- `Embedding and Fitting - ANOVA.py`  
- `2way_ANOVA_emb_vs_comb.py`

---

---

## Classification Performance – Embedding vs. Embedding+Fitting

To assess whether combining **embedding vectors** with **function fitting vectors** improves downstream classification, we trained a supervised model to predict cluster assignment using both representations.

Main script: `compare_embedding_vs_fitting.py`

### Input

- `Embedding008.json` – Per-feature embedding vectors from Chronos (output of `Embedding.py`)
- `embedding_fitting_combined_by_feature_scaled.json` – Combined embedding + fitting vectors per feature
- `cluster_assignments_k3.csv` – Cluster labels for each cell (from unsupervised clustering)
- `embedding_fitting_Merged_Clusters_PCA.csv` – Extended version of the original dataset with PCA and cluster annotations

### Functionality

1. **Parse and flatten feature vectors** from both sources:
   - Chronos-based embeddings only
   - Embedding + fitting concatenated vectors
2. **Merge with cluster labels** based on cell ID
3. **Train Random Forest classifiers** to predict cluster labels:
   - Using **embedding only**
   - Using **embedding + fitting**
4. **Evaluate performance** using:
   - Accuracy
   - Confusion matrix
   - Precision / Recall / F1 score (per cluster)

### Output

- `classification_report_Embedding_Only.xlsx`  
- `classification_report_Embedding_+_Fitting.xlsx`  
- `confusion_matrix_Embedding_Only.png`  
- `confusion_matrix_Embedding_+_Fitting.png`

This comparison helps quantify the added value of incorporating functional fitting parameters into cell representations for classification tasks.

---

## File Summary

| File | Description |
|------|-------------|
| `raw_all_cells.csv` | Main input: filtered time series per cell, with `ds` and `unique_id` columns |
| `selected_features.txt` | List of informative features selected for forecasting |
| `TSA_analysis.py` | Forecasting pipeline using MMF |
| `many-model-forecasting/` | Adapted MMF repo (see its own README) |
| `best_model_per_feature.csv` | Best forecast model per feature |
| `Embedding.py` | Embedding pipeline for time-series features |
| `Unsupervised Clustering - Embedding - K=3.py` | PCA + KMeans clustering of embedding vectors (k=3) |
| `cluster_assignments_k3.csv` | Cluster labels and PCA coordinates (one row per cell) |
| `Merged_Clusters_PCA.csv` | Original dataset merged with cluster and PCA labels |
| `Descriptive_Table_By_Cluster.py` | Computes descriptive statistics per cluster |
| `Descriptive_Table_By_Cluster_UniqueCells.xlsx` | Summary statistics (mean, std, CI) per cluster |
| `ANOVA.py` | Performs one-way ANOVA per feature to test for significant differences |
| `ANOVA - OneWay.xlsx` | One-way ANOVA output with multi-level headers |
| `K3_analysis.py`, `K3_graphs.py` | Two-way ANOVA and visualization of old vs. new clusters |
| `anova_two_way.csv` | Two-way ANOVA summary (per feature) |
| `tukey_two_way.csv` | Tukey HSD post-hoc results |
| `old_populations_means.csv`, `new_populations_means.csv` | Aggregated cluster-level means for each project |
| `tukey_pvalue_heatmap.png` | Heatmap of Tukey-adjusted p-values |
| `Normalized_Feature_Ratios.csv` | Feature values normalized to Group 0 |
| `Cell_Distribution_Per_Treatment.png` | Barplot showing cell group distribution by treatment |
| `Heatmap_All_Features_ZScore.png` | Z-score heatmap of all features by cluster |
| `Heatmap_Morphological_Features_ZScore.png` | Z-score heatmap for morphological features |
| `Heatmap_Kinetic_Features_ZScore.png` | Z-score heatmap for kinetic features |
| `Normalized_Kinetic_Part_*.png` | Barplots for subsets of kinetic features |
| `Normalized_Morph_Part_*.png` | Barplots for subsets of morphological features |
| `fit_cell_trajectory.py` | Curve-fitting pipeline for functional modeling |
| `fitting_all_models.csv` | Full results of function fitting per cell-feature |
| `fitting_significant_models.csv` | Statistically significant function fits |
| `fitting_top3_with_nrmse.csv` | Top 3 fitted models per cell-feature |
| `fitting_best_with_nrmse.csv` | Best fitted model per cell-feature |
| `fitting_best_model_log.json` | Embedding vectors from best model (log-transformed) |
| `fitting_top3_models_log.json` | Embedding vectors from top 3 models (log-transformed) |
| `fitting_best_model_log_scaled.json` | Embeddings from best model (log + normalized) |
| `fitting_top3_models_log_scaled.json` | Embeddings from top 3 models (log + normalized) |
| `requirements.txt` | Python dependencies |
| `my_models_conf.yaml` | Forecast model configuration |
| `Embedding and Fitting - Unsupervised Clustering.py` | Combines embedding and fitting vectors, performs PCA + KMeans clustering |
| `Embedding and Fitting - descriptive table by cluster.py` | Descriptive statistics per cluster using combined vectors |
| `Embedding and Fitting - ANOVA.py` | One-way ANOVA on combined embedding + fitting data |
| `2way_ANOVA_emb_vs_comb.py` | Two-way ANOVA comparing embedding vs. embedding+fitting representations |
| `compare_embedding_vs_fitting.py` | Classification of cluster labels using embedding vs. embedding+fitting vectors |
| `embedding_fitting_Merged_Clusters_PCA.csv` | Combined dataset of embedding + fitting vectors with PCA and cluster labels |
| `embedding_fitting_combined_by_feature_scaled.json` | JSON file with concatenated embedding + fitting vectors per feature |
| `embedding_fitting_Descriptive_Table_By_Cluster_UniqueCells.xlsx` | Cluster-level summary statistics based on combined vectors |
| `embedding_fitting_ANOVA - OneWay.xlsx` | One-way ANOVA output for embedding + fitting clusters |
| `embedding_fitting_anova_two_way.csv` | Two-way ANOVA results comparing embedding vs. embedding+fitting |
| `embedding_fitting_tukey_two_way.csv` | Tukey HSD post-hoc results for each feature × cluster |
| `embedding_fitting_tukey_pvalue_heatmap.png` | Heatmap visualization of Tukey-adjusted p-values |
| `classification_report_Embedding_Only.xlsx` | Classification report (Random Forest) using Chronos embedding only |
| `classification_report_Embedding_+_Fitting.xlsx` | Classification report using combined embedding + fitting |
| `confusion_matrix_Embedding_Only.png` | Confusion matrix for embedding-only classification |
| `confusion_matrix_Embedding_+_Fitting.png` | Confusion matrix for combined embedding + fitting classification |

---

### Required Model Files (Moirai)

Due to file size limitations, the following model folders **are not included in the repository**.  
To use the Moirai models, please **manually download** them from [Hugging Face](https://huggingface.co/ibm/moirai/tree/main) and place them inside:

```
many-model-forecasting/mmf_sa/models/moiraiforecast/
```

Required folders:

- `moirai-1.0-R-base`
- `moirai-1.0-R-large`
- `moirai-1.0-R-small`
- `moirai-moe-1.0-R-base`
- `moirai-moe-1.0-R-small`


## Notes

- Only **foundation models** are evaluated during forecasting.
- MMF pipeline is run **locally with Spark on CPU** (GPU support in progress).
- The **function fitting step is fully standalone**, and does not depend on MMF.
- Both types of representations (forecast-based and function-based) will be tested for impact on **embedding and classification** quality.

To install required packages:

```
pip install -r requirements.txt
```

➡ For details about the forecasting engine, see:  
[`many-model-forecasting/README.md`](many-model-forecasting/README.md)


