
```md
# Validation Against Imaris (Segmentation & Tracking)

This folder presents the **quantitative and qualitative validation** of the
Cell-ACDC segmentation + tracking pipeline against **Imaris** reference results.

Imaris is treated as the reference tracker for:
- object positions (X/Y)
- track continuity
- track identity over time

The goal is to evaluate:
- how close Cell-ACDC tracks are to Imaris tracks
- where and why mismatches occur
- whether optimization steps reduce mis-tracked cells

---

## 1) Validation setup

### Reference
- **Imaris** tracking export (positions + track IDs)

### Evaluated pipeline
- Segmentation: optimized Cellpose3
- Tracking: Cell-ACDC (TrackPy) with Imaris-derived parameters

Both outputs were converted to a **common Imaris-like structure**
to allow direct comparison per timepoint.

---

## 2) Track mapping strategy (ACDC ↔ Imaris)

Because Cell-ACDC and Imaris generate independent track IDs, a mapping step is required.

### Mapping rules
- Mapping is performed **per timepoint**
- 1-to-1 matching only:
  - each Imaris object can match at most one ACDC object
- Matching is based on spatial proximity (X/Y distance thresholds)
- Tracks must overlap for at least a minimum number of frames

Result:
- a mapping table: `ACDC Parent ID → Imaris Parent ID`

Details are documented in:
- `mapping_strategy.md`

---

## 3) Metrics used

We evaluate tracking agreement using:

### A) Track-level metrics
- Track length (number of frames)
- Overlap duration between ACDC and Imaris tracks
- Number of unmatched detections

### B) Position accuracy
For each matched track:
- Linear regression:  
  - Imaris X vs ACDC X  
  - Imaris Y vs ACDC Y
- Metrics:
  - R² (X), R² (Y)
  - RMSE (X), RMSE (Y)

### C) Temporal failure analysis
- Identify **timepoints where R² < 0.7**
- Report the corresponding Imaris and ACDC positions
- Helps locate moments of drift, jumps, or wrong associations

Metric definitions are detailed in:
- `metrics_definition.md`

---

## 4) Visual validation

To complement numeric metrics, we generate visual overlays:

### Best tracks
- Top 10 tracks with highest combined R²
- Overlaid X/Y trajectories (ACDC vs Imaris)

### Worst tracks
- Bottom 10 tracks with lowest combined R²
- Used to identify typical failure modes:
  - false linking
  - extra cells injected into tracks
  - track identity switches

Plots are saved in:
```

results/figures/

```

---

## 5) Results summary

Final validation outputs include:

- `track_metrics.csv`  
  Per-track statistics (length, R², RMSE, overlap)

- `r2_drop_timepoints.csv`  
  Timepoints where tracking agreement degrades

- Overlay plots for best and worst tracks

These results show:
- clear improvement after parameter optimization
- remaining errors mainly occur in dense regions or fast motion frames

---

## 6) Known limitations

- Imaris is not a perfect ground truth
- Mapping can fail in very dense scenes
- Segmentation errors propagate into tracking evaluation

Despite this, validation provides a strong **relative comparison**
between default and optimized Cell-ACDC configurations.

---

## 7) Reproducibility

Scripts used for validation are stored in:
```

scripts/

```

They include:
- track matching
- regression analysis
- visualization

To re-run validation:
1. Export Imaris tracking
2. Export Cell-ACDC tracking
3. Run mapping + comparison scripts
4. Generate summary tables and figures

---

## 8) Conclusion

Validation against Imaris confirms that:
- transferring Imaris-inspired parameters into Cell-ACDC
  significantly reduces mis-tracked cells
- optimized settings improve spatial consistency and track continuity
- remaining errors are biologically interpretable and systematic

This validation step justifies the chosen segmentation and tracking configuration.
```

---

