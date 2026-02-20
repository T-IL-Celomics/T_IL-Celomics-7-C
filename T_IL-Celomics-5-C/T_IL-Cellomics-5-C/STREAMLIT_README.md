# T_IL-Cellomics-5-C Streamlit Application

A comprehensive GUI application for analyzing morphokinetic behavior of BRCA1-knockout breast cancer cells.

## 🚀 Quick Start

### Installation

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r streamlit_requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

## 📋 Features

### 1. 📊 Data Overview
- Load CSV files via upload or from directory
- View data statistics and summary metrics
- Identify missing values
- Explore column information

### 2. 🎯 Feature Selection
- PCA-based feature importance ranking
- Correlation analysis between features
- Interactive feature filtering
- Export selected features

### 3. 🎨 PCA & Clustering
- Principal Component Analysis (PCA)
- K-Means clustering with configurable K
- Silhouette score evaluation
- Interactive scatter plots by cluster and treatment

### 4. 📈 ANOVA Analysis
- One-Way ANOVA by cluster
- Two-Way ANOVA (Cluster × Treatment)
- P-value highlighting for significant features
- Downloadable results

### 5. 📋 Descriptive Statistics
- Mean, Standard Deviation, Standard Error by cluster
- 95% Confidence Intervals
- Visualization of statistics with error bars
- Export to CSV

### 6. 📊 Visualization Dashboard
- **Distribution plots** - Histograms with cluster coloring
- **Scatter plots** - Feature vs feature analysis
- **Box plots** - Distribution by groups
- **Time series** - Feature trajectories over time
- **Correlation heatmaps** - Feature relationships
- **Violin plots** - Distribution shape visualization

## 📁 Expected Data Format

The application expects CSV files with the following structure:

### Required Columns:
- `Experiment` - Experiment identifier
- `Parent` - Cell/track identifier
- `TimeIndex` - Time point index

### Optional Metadata:
- `Treatment` - Treatment group label
- `Cluster` - Pre-assigned cluster labels
- `dt` - Time interval

### Feature Columns:
All other numeric columns are treated as features for analysis.

### Example:
```csv
Experiment,Parent,TimeIndex,Treatment,Velocity_X,Velocity_Y,Area,...
EXP001,1,0,CON,0.5,0.3,125,...
EXP001,1,1,CON,0.6,0.4,128,...
```

## 🔧 Configuration

The app uses Streamlit's session state to maintain analysis results between pages. Key session variables:

- `pca_df` - PCA results with cluster assignments
- `pca_model` - Fitted PCA model
- `kmeans_model` - Fitted K-Means model
- `anova_results` - ANOVA analysis results
- `descriptive_stats` - Descriptive statistics by cluster
- `selected_features` - Features selected for analysis

## 📊 Workflow

1. **Load Data** → Upload or select CSV file from sidebar
2. **Explore** → Review data in Data Overview page
3. **Select Features** → Run feature selection to identify important variables
4. **Cluster** → Run PCA and K-Means clustering
5. **Analyze** → Perform ANOVA and calculate descriptive statistics
6. **Visualize** → Create custom visualizations
7. **Export** → Download results as CSV files

## 🛠️ Troubleshooting

### Common Issues:

1. **"No feature columns found"**
   - Ensure your CSV has numeric columns beyond the metadata columns

2. **"No cluster information available"**
   - Run the PCA & Clustering analysis first, or ensure your data has a `Cluster` column

3. **"Data length mismatch"**
   - This occurs when filtering removes rows. Try re-running clustering after loading fresh data

### Performance Tips:

- For large datasets (>100k rows), consider sampling before visualization
- Use the feature selection to reduce dimensionality before clustering
- Save intermediate results using the download buttons

## 📝 License

This project is part of the T_IL-Cellomics research initiative.


