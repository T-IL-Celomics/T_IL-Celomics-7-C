
---

```markdown
# ACDC Image Renaming and Structuring Tool

This Python script is designed to prepare microscopy time-lapse image data for use with **Cell-ACDC**. It automatically renames and reorganizes image files from the Tsarfaty Lab format into a clean structure that Cell-ACDC can process efficiently.

---

## Input Data Structure

Your input folder should look like this:

```

PhaseX\_DATA/
├── ORANGE/
│    └── B2/
│         └── B2\_1/
│              └── \[images with date/time in filenames]
├── GREEN/
│    └── ...
└── NIR/
└── ...

```

- `PhaseX_DATA`: Main folder where `X` is the phase number (e.g., `Phase1_DATA`, `Phase2_DATA`).  
- Inside: channel folders (ORANGE, GREEN, NIR — not all channels are required).  
- Inside channels: location folders (e.g. B2, B3, C2).  
- Inside locations: field folders (e.g. B2_1 = location B2, field 1; B2_2 = location B2, field 2).  
- Inside fields: `.tif` images with date and time info in filenames.

---

##  What the Script Does

The script:
- Sorts images in each field by the timestamp in their filename.
- Renames images by frame number (starting at 0).
- Adds field, location, and channel info to the filenames.
- Groups images by field into an output folder structure compatible with Cell-ACDC.

---

## 📂 Output Structure

After running the script:

```

ACDC\_IN\_PhaseX/
├── field1/
│    └── Images/
│         └── field1\_B2\_ORANGE\_0.tif
│         └── field1\_B2\_ORANGE\_1.tif
│         └── ...
└── field2/
└── Images/
└── field2\_B2\_GREEN\_0.tif
└── ...

````

---

## How to Use

1. Place `renameACDC.py` in the same directory as your `PhaseX_DATA` folder.  
2. Open terminal / Anaconda Prompt.  
3.  Navigate to the directory:
```bash
cd path/to/your/data
````

4. Run the script, specifying the phase:

```bash
python renameACDC.py Phase1
```

*(Replace `Phase1` with your actual phase name.)*

---

## ⚠ Requirements

* Python 3.x
* Standard libraries: `pathlib`, `re`, `shutil`, `sys`

---

## Notes

* The script assumes your folder is named exactly `PhaseX_DATA` (where X is the phase number you provide).
* All renamed images from a field will be grouped into a single folder for easy loading into Cell-ACDC.
* The output folder (`ACDC_IN_PhaseX`) will be created in the same directory as the input folder.

---

## Repository

https://github.com/T-IL-Celomics/T_IL-Celomics-7-C

---

## Contact

For questions or issues, please open an issue in this repository or contact the maintainer, at Wassemb@mail.tau.ac.il.

```


```
