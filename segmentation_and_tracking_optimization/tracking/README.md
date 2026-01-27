

```md
# Tracking Optimization (Cell-ACDC TrackPy) Using Imaris Parameters

This folder documents how we optimized **cell tracking** in **Cell-ACDC** by transferring tracking behavior from **Imaris** into the Cell-ACDC/TrackPy tracker settings.

Our main observation was that default tracking settings can introduce:
- false links (wrong cell assigned to a track)
- extra detections added into tracks (cells that are not truly part of the track)
- track fragmentation (unnecessary track breaks)

To reduce these issues, we used Imaris as a reference and tuned Cell-ACDC tracking parameters accordingly.

---

## 1) What we did (high level)

### Step A — Track inside Cell-ACDC
We first ran tracking in Cell-ACDC using its built-in pipeline:
- segmentation already completed (final masks)
- tracking based on TrackPy (linking detections over time)

### Step B — Extract parameters from Imaris
We exported/inspected Imaris tracking settings and derived the equivalent TrackPy-style parameters (or closest mapping).

### Step C — Apply Imaris-derived parameters in Cell-ACDC
We set Cell-ACDC tracking parameters to match the Imaris behavior as closely as possible.

### Step D — Validate against Imaris results
We compared Cell-ACDC tracks vs Imaris tracks using:
- track length overlap
- XY position agreement per timepoint
- R² per track (X and Y)
- identification of timepoints where performance drops (e.g., R² < 0.7)

---

## 2) Which parameters were transferred (Imaris → TrackPy/Cell-ACDC)

> Note: Imaris and TrackPy do not always use identical naming, so the mapping is “equivalent behavior” rather than identical UI fields.

Common mapping we used:

- **Max distance / Max gap distance (Imaris)**  
  ↔ `search_range` / `max_jump` (TrackPy/Cell-ACDC)  
  Meaning: maximum allowed displacement between consecutive frames to link detections.

- **Gap closing / Max gap size (Imaris)**  
  ↔ `memory` (TrackPy/Cell-ACDC)  
  Meaning: how many frames a cell can disappear and still be linked to the same track.

- **Minimum track duration / track length filter (Imaris)**  
  ↔ `min_track_len` (post-filtering or inside Cell-ACDC)  
  Meaning: remove short noisy tracks.

We store the final chosen parameters in:
- `settings/imaris_mapped_trackpy_params.yaml`

---

## 3) How we chose `max_jump` / `search_range`

We used a biology-informed rule-of-thumb as a starting point:

- `max_jump ≈ 0.5 × cell_diameter`

Then we adjusted based on real motion:
- if cells move faster (or time interval is larger), increase max_jump
- if false links happen frequently, decrease max_jump

In our dataset:
- estimated cell diameter: **15.67 µm**
- initial max_jump estimate: **~7–8 µm**
- final value chosen: *(fill after final selection)*

---

## 4) How to reproduce the tracking configuration

### Inside Cell-ACDC (GUI)
1. Open the dataset in Cell-ACDC
2. Select tracker = TrackPy (or Cell-ACDC tracking module)
3. Set parameters according to:
   - `settings/imaris_mapped_trackpy_params.yaml`
4. Run tracking
5. Export detections + tracks (CSV)

### External scripts (optional validation)
We used comparison scripts to evaluate tracking vs Imaris:
- best tracks overlay (ACDC vs Imaris)
- worst tracks overlay
- report of timepoints that reduce R² below threshold

Scripts are saved in:
- `scripts/` (to be added / finalized)

---

## 5) Results we report

We summarize:
- number of tracks
- median track length
- best 10 matched tracks (visual overlay)
- worst 10 matched tracks (visual overlay)
- timepoints where R² drops below 0.7 (table)

Final outputs are saved in:
- `../../04_results/summary/`
- `../../04_results/figures/`

---

## 6) Known failure modes (what we saw)

Even after tuning, mis-tracks can still occur due to:
- dense scenes (cells cross paths)
- segmentation mistakes (merged/split objects)
- large motion between frames (insufficient max_jump)
- drift / stage movement (needs registration correction)

Next improvements (if needed):
- add motion model / Kalman prediction
- add track validation by size/intensity consistency
- pre-register frames to reduce global drift

---

