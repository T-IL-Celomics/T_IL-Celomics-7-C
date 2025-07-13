# PyBatch - Batch Cell Motility Analysis GUI

PyBatch is an advanced Python GUI for batch analysis of cell tracking and motility experiments. Built with [Streamlit](https://streamlit.io/), it provides automated extraction, computation, visualization, and reporting of a wide range of motility and cell-level features from Imaris and Incucyte data.

## Features

- **Batch Processing:** Analyze multiple experiments, wells, or plates in a single workflow.
- **Data Integration:** Supports both Imaris `.xls` files and Incucyte outputs.
- **Comprehensive Feature Extraction:** Calculates speed, displacement, acceleration, MSD, shape, intensity, and many other metrics.
- **Interactive GUI:** Built with Streamlit for ease-of-use.
- **Automated Plotting:** Generates customizable graphs, heatmaps, and PDF reports for all analyzed parameters.
- **Export Options:** Save results as Excel, HTML, images, or PDF summaries.
- **Session Management:** Robust per-session state using custom session state modules.
- **Windows Batch File Support:** One-click launch via `.bat` file.

## File Structure

```
.
├── batch_calculations.py    # Core calculations, plotting, analysis classes
├── pybatch.py               # Main Streamlit GUI entry point
├── pybatch_objects.py       # Data structures and cell-level calculations
├── requirements.txt         # Python dependencies
├── pybatch.bat              # Windows batch file for quick launch
```

## Installation

1. **Clone or Download the Repository**

2. **Set Up a Python Environment**

   It's recommended to use [Anaconda](https://www.anaconda.com/) or `venv`:

   ```sh
   conda create -n pybatch-env python=3.8
   conda activate pybatch-env
   ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **(Windows only) Ensure Excel is Installed**

   Some Imaris parsing steps require access to a local Excel installation via `pywin32`.

## Usage

### 1. Launch the App

- **Double-click `pybatch.bat`**  
  or  
- **Run from command line:**
  ```sh
  streamlit run pybatch.py
  ```

### 2. Interact with the GUI

- Use the sidebar to select experiments, configure parameters, and launch analyses.
- Upload your Imaris or Incucyte files as prompted.
- Visualize results, generate reports, and export outputs as needed.

### 3. Outputs

- Plots and tables can be exported to PDF, Excel, and HTML.
- Session state is preserved per session.

### 4. Troubleshooting

- If you see errors about Excel, check that Microsoft Excel is installed.
- For issues with session state, try restarting the app.

## Notes

- The app is primarily tested on Windows.
- Data privacy: All processing occurs locally; no data is uploaded or shared.
- For best performance, use Chrome or Edge and update Streamlit to the latest version compatible with your system.

