
```md
# Segmentation Optimization (Cellpose3 + External Tuning)

This folder documents how we optimized the **segmentation stage** for our dataset after testing multiple options inside **Cell-ACDC**.  
The key outcome is a reproducible external optimization pipeline (Python + Optuna) that searches for the best **Cellpose** parameters using ground-truth masks and objective metrics (Dice / IoU).

---

## 1) Background: What we did inside Cell-ACDC first

We started by running segmentation directly inside the **Cell-ACDC GUI**, testing multiple segmentation configurations/algorithms across several experiments.

- We recorded these experiments and will link them here:
  - Experiment links: *(to be added)*

After comparing results qualitatively and quantitatively, the best overall segmentation performance for our data was achieved with:

✅ **Cellpose3** (model type: `cyto`)

At that point, we noticed that:
- The GUI-based tuning is limited for systematic hyperparameter search.
- We needed a repeatable way to optimize parameters across many frames, using an objective metric.

So we moved to external optimization.

---

## 2) Why optimize outside Cell-ACDC?

External optimization enables:
- Automated hyperparameter search (instead of manual tweaking)
- Reproducible best parameters (saved and documented)
- Full control over evaluation metrics (Dice / IoU)
- Dataset-specific calibration (alignment between GT masks and predictions)

We used **Optuna** to search over Cellpose parameters and maximize mean Dice score.

---

## 3) Dataset structure expected by the script

The script expects a dataset folder like this:

```

dataset/
images/
frame000.tif
frame001.tif
...
masks/
frame000.tif
frame001.tif
...

````

Rules:
- Each image must have a matching mask with the **same base filename**.
- Images and masks are TIFF (`.tif` / `.tiff`).
- Masks can be binary or labeled; the script treats mask pixels `> 0` as foreground.

---

## 4) Optimization script overview (what it does)

The script performs 3 main stages:

### A) Load images + ground-truth masks
- Reads grayscale from TIFF
- Reads masks and binarizes them: `mask > 0`

### B) Calibration step (alignment + orientation)
In some datasets, exported GT masks may differ from predictions by:
- flips (up/down, left/right)
- rotations (90/180/270)
- global XY shift (pixel offset)

To avoid scoring errors caused by these mismatches, we do an automatic calibration:

1. Run a **baseline Cellpose** prediction on a small number of frames.
2. Try multiple GT transforms:
   - identity, flipud, fliplr, rot90/180/270, and combined transforms
3. For each transform:
   - estimate shift using `phase_cross_correlation`
   - clamp shift to a maximum (default: 40 px)
   - apply shift and compute Dice vs baseline
4. Choose the best transform by majority vote across frames, and use the median shift.

Output of this stage:
- chosen transform name (e.g. `rot90_fliplr`)
- global shift `(dy, dx)`
- ensures consistent GT alignment for all trials

### C) Optuna hyperparameter optimization (maximize mean Dice)
We optimize these Cellpose parameters:

- `diameter` (float, dataset-dependent range)
- `flow_threshold` (float)
- `cellprob_threshold` (float)
- `min_size` (int)

For each trial:
1. Run Cellpose on every frame with candidate params
2. Convert masks to binary: `pred = (masks > 0)`
3. Compute:
   - Dice score (objective)
   - IoU score (logged as extra info)
4. Return mean Dice across all frames

Optuna selects the parameters that maximize mean Dice.

---

## 5) Metrics

We use two standard segmentation overlap metrics:

### Dice
\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

### IoU
\[
IoU = \frac{|P \cap G|}{|P \cup G|}
\]

Where:
- `P` = predicted foreground pixels
- `G` = ground-truth foreground pixels

---

## 6) How to run

### Install dependencies
Recommended (conda):

```bash
conda create -n seg_opt python=3.10 -y
conda activate seg_opt

pip install numpy tifffile optuna scikit-image scipy
pip install cellpose
````

> If running on a GPU server, make sure your PyTorch/CUDA setup matches your machine.

### Run optimization

From the directory containing the script:

```bash
python optimize_cellpose_segmentation.py
```

Key configuration inside the script:

* `DATASET_DIR` (default: `dataset`)
* `MODEL_TYPE` (default: `cyto`)
* `USE_GPU` (True/False)
* `N_TRIALS` (default: 20)
* `CALIBRATION_FRAMES` (default: 8)

---

## 7) Outputs to record (what we save)

At the end, the script prints:

* **BEST mean Dice**
* **BEST params**
* **BEST mean IoU**
* alignment transform + shift used

We record these final chosen parameters here:

* `settings/best_params.json` *(recommended to add)*
* `notes.md` *(short explanation + final numbers)*

---

## 8) Notes / Next steps

* Once best segmentation params are selected, we re-run segmentation with the chosen parameters to generate final masks.
* These masks are then used as input to the **tracking optimization** stage (documented separately in the tracking folder).
* If the dataset contains drift over time, we may extend calibration to per-frame alignment (slower but more robust).

---

## References

* Cellpose (segmentation model)
* Optuna (hyperparameter optimization)
* phase_cross_correlation (registration for shift estimation)

```

---

## Next: I’ll make it even cleaner to your exact repo style
To finish this segmentation folder nicely, do **one** of these (no need for all):
1) Tell me the final “best params” you got from the Optuna output (copy the printed lines), **or**
2) Tell me the dataset name you want mentioned (e.g., `C-1_MI006-nir_C2_1`) and where the GT masks came from (Imaris export / manual / etc.)

Then I’ll also generate:
- `settings/best_params.json` template
- `run_commands.sh`
- `notes.md` with your exact story (Cell-ACDC experiments → why we moved out → final chosen params)
```
