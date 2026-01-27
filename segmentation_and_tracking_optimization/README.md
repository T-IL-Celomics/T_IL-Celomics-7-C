# Segmentation & Tracking Optimization (Cell-ACDC ↔ Imaris)

## Goal
Reduce miss-tracked cells and improve mapping consistency between Cell-ACDC tracking and Imaris reference.

## Data
- Dataset: C-1_MI006-nir_C2_1 (describe briefly)
- Cell diameter: 15.67 µm
- Coordinate units: µm

## Pipeline
1. Segmentation (Cellpose / Cell-ACDC)
2. Tracking (TrackPy inside Cell-ACDC or exported detections)
3. Mapping & comparison vs Imaris (1-1 matching per time)
4. Metrics + visual QA (best/worst tracks)

## Key Metrics
- Track overlap length
- R² for X and Y per track
- RMSE / MAE
- # unmatched objects per time
- Timepoints where R² < 0.7

See: `04_results/summary/metrics_table.csv`

## Final Chosen Parameters
Tracking:
- search_range / max_jump: (fill)
- memory: (fill)
- min_track_len: (fill)

See: `03_tracking/settings/trackpy_params.yaml`

## Results (Summary)
- Best 10 tracks: `04_results/figures/best_tracks_overlay.png`
- Worst 10 tracks: `04_results/figures/worst_tracks_overlay.png`
- R² drops (timepoints): `04_results/figures/r2_drop_timepoints.png`

## Reproduce
Go to: `05_reproducibility/how_to_run.md`
