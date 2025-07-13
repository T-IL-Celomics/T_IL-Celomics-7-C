# TASC: Time Series Analysis and Clustering GUI

A comprehensive, user-friendly Python GUI for advanced time series analysis, clustering, and visualization.  
Built with Streamlit, TASC integrates powerful statistical tools, PCA, autoencoders, Wasserstein distance analysis, k-means clustering, and more, all with publication-quality visualizations.

---

## Features

- **Visual GUI**: Intuitive Streamlit-based interface for running and analyzing time-series datasets.
- **Supports Large Datasets**: Designed to handle and process data from cellomics and similar experiments.
- **Statistical Analysis**: Includes ANOVA, hierarchical clustering, KDE, and more.
- **Clustering & Embeddings**: Supports PCA, k-means, Gaussian Mixture Models, and Autoencoder-based embeddings.
- **Distribution Comparison**: Wasserstein distance matrix/heatmap for comparing groups/experiments.
- **Export & Reporting**: Save plots, tables, and logs for publication or further processing.
- **Customizable**: Modular Python code for easy extension.

---

## Installation

1. **Clone or Download This Repository**

2. **Set up the Conda Environment**

   Create a conda environment called `tasc-env` and install requirements:
   ```bash
   conda create -n tasc-env python=3.8
   conda activate tasc-env
   pip install -r requirements.txt
   ```

3. **Run the GUI**

   Use the provided batch file (Windows) or run via Streamlit directly:
   - **Windows:**  
     Double-click the provided `TASC.bat` file (this will activate `tasc-env` and launch the GUI).
   - **Manual:**
     ```bash
     conda activate tasc-env
     streamlit run TASC_GUI.py
     ```

   The GUI will open in your web browser.

---

## Usage

1. **Prepare your data**  
   - Place your data files in the correct folder (see below).
   - Supported formats: Pickled pandas DataFrames (`.pkl`), Excel (`.xlsx`), etc.

2. **Start the GUI**  
   - Double-click `TASC.bat` *or* run `streamlit run TASC_GUI.py` after activating the environment.

3. **Load Data & Set Parameters**
   - Use the interface to upload or select data files.
   - Configure analysis parameters, select features, and set clustering options.

4. **Run Analysis**
   - Click “Run Analysis”.
   - Results, plots, and statistics will appear interactively.
   - All key results can be exported for further use.

---

## Exporting/Printing Results

- **To save your analysis session as an HTML file**, use the [SingleFile browser extension](https://github.com/gildas-lormeau/SingleFile) (available for Chrome/Firefox).
    - Open your completed analysis in the browser, click the SingleFile extension icon to save the whole page as HTML.
    - **Note:** Large tables are also saved separately to another HTML file, as the main Streamlit export can become unscrollable when too many tables are included.
    - After saving, you will have one HTML file for the Streamlit app and another HTML file for the tables, both in the chosen folder.

---

## File Structure

- `TASC_GUI.py` – Main GUI interface (Streamlit)
- `util.py` – Data processing, clustering, plotting, and statistical helper functions
- `WD_util.py` – Wasserstein distance functions
- `requirements.txt` – All dependencies for reproducibility
- `TASC.bat` – Windows batch file for easy launch
- `ANOVA.py` – Performs one-way ANOVA and Tukey HSD post-hoc tests on experimental summary tables, and outputs formatted results and tables.

---

## Requirements

- Python 3.8+  
- Conda environment `tasc-env` with dependencies in `requirements.txt`

---

## Example Output

An example HTML export of the TASC GUI results, saved using the SingleFile browser extension, is included in this repository.
Open the HTML file in your browser to preview what the analysis output and interface look like.

---

## Troubleshooting

- **Blank/empty plots or errors:** Ensure your data matches expected format (columns like `Experiment`, `PC1`, etc.).
- **Streamlit errors:** Try updating Streamlit: `pip install --upgrade streamlit`
- **GPU usage (for deep learning):** The code will automatically detect and use available GPUs if possible.

---


