# Foundation Model Forecasting for Cell Motility Time Series

This fork of the [Many-Model-Forecasting](https://github.com/databricks-industry-solutions/many-model-forecasting) project adapts the original structure to support **biological time series forecasting**, with a focus on analyzing morpho-kinetic features of individual cells over time (e.g., using columns like `Parent`, `TimeIndex`, and morpho-dynamic parameters).

> This version is designed for use outside Databricks, on **local machines** with **PySpark + CPU** execution.

---

## Key Adaptations

- Adapted for **biological cell movement time series**, identified by `Parent` (cell ID), `TimeIndex` (frame), and dynamic features.
- **No MLflow dependency** – model training and evaluation are handled locally.
- **Spark** is used for applying pandas UDFs across thousands of time series in parallel.
- Uses only **foundation models** (TimesFM, Chronos, Moirai) as defined in `my_models_conf.yaml`.
- Results saved locally in the `results/` folder as `.csv` files (per model).

---

## Key Modified Files

| File | Description |
|------|-------------|
| `LocalForecaster.py` | Orchestrates per-model evaluation and scoring using Spark |
| `TimesFMPipeline.py` | Forecasting logic for TimesFM models |
| `ChronosPipeline.py` | Forecasting logic for Chronos models |
| `MoiraiPipeline.py` | Forecasting logic for Moirai models |

Other core utilities (model registry, abstract classes) remain unchanged from the original MMF repo.

---

## How It Works

For each foundation model:
1. The pipeline loads a full time-series dataset with features like `Parent`,`Experiment`, `TimeIndex`, and target variable(s).
2. A per-cell time series is extracted and evaluated over its last 5 frames (backtest).
3. Forecasts and metrics (e.g., RMSE, MAE) are computed and saved to:
   - `results/{model_name}_metrics.csv` for evaluation
   - `results/{model_name}_forecast.csv` for scoring

---

## Input Format

The input dataset must include at least:
- `Parent` + `Experiment`: Cell or object ID
- `TimeIndex`: Frame or time step
- `...`: Dynamic features (e.g., velocity, displacement2)

A typical call to the pipeline is made via an external script (`TSA_analysis.py`, placed outside this folder) which loads the data, reads the config, and invokes the `LocalForecaster`.

---

## Notes

- `my_models_conf.yaml` (in the root directory) defines the active models and their parameters.
- To run forecasting, see the root-level `README.md` and `TSA_analysis.py`.

---

## Reference

Original MMF repository and documentation:  
https://github.com/databricks-industry-solutions/many-model-forecasting
