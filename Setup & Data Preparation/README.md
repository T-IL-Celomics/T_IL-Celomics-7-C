✅ Absolutely — here’s a **clean, professional version** of your `README.md` without emojis, with clear structure and formatting appropriate for a formal GitHub repository:

---

```markdown
# ACDC Image Renaming and Structuring Tool

This Python script prepares microscopy time-lapse image data for use with **Cell-ACDC**. It renames and reorganizes image files from the Tsarfaty Lab format into a structure compatible with Cell-ACDC processing.

---

## Input Data Structure

Your input folder should be organized as follows:

```

PhaseX_DATA/
├── ORANGE/
│ └── B2/
│ └── B2_1/
│ └── [images with date/time in filenames]
├── GREEN/
│ └── ...
└── NIR/
└── ...

```

- `PhaseX_DATA`: Main folder where `X` is the phase number (e.g., `Phase1_DATA`, `Phase2_DATA`).
- Inside: channel folders (ORANGE, GREEN, NIR — not all channels are required).
- Inside channels: location folders (e.g., B2, B3, C2).
- Inside locations: field folders (e.g., B2_1 = location B2, field 1; B2_2 = location B2, field 2).
- Inside fields: `.tif` images containing date and time information in the filenames.

---

## What the Script Does

The script:
- Sorts images in each field by timestamp in the filename.
- Renames images by frame number, starting at 0.
- Adds field, location, and channel information to the filenames.
- Groups images by field into an output folder structure suitable for Cell-ACDC.

---

## Output Structure

After running the script, the output folder will look like this:

```

ACDC_IN_PhaseX/
├── field1/
│ └── Images/
│ ├── field1_B2_ORANGE_0.tif
│ ├── field1_B2_ORANGE_1.tif
│ └── ...
└── field2/
└── Images/
├── field2_B2_GREEN_0.tif
└── ...

````

---

## How to Use

1. Place `renameACDC.py` in the same directory as your `PhaseX_DATA` folder.
2. Open a terminal or Anaconda Prompt.
3. Navigate to the directory containing the script and input folder:
   ```bash
   cd path/to/your/data
````

4. Run the script, specifying the phase name:

   ```bash
   python renameACDC.py Phase1
   ```

   Replace `Phase1` with your actual phase name.

---

## Requirements

* Python 3.x
* Standard libraries: `pathlib`, `re`, `shutil`, `sys`

---

## Notes

* The input folder must be named exactly `PhaseX_DATA`, where `X` is the phase number you provide as an argument.
* All renamed images from a field will be grouped into a single folder for easy loading into Cell-ACDC.
* The output folder (`ACDC_IN_PhaseX`) will be created in the same directory as the input folder.

---

## Repository

[https://github.com/T-IL-Celomics/T\_IL-Celomics-7-C](https://github.com/T-IL-Celomics/T_IL-Celomics-7-C)

---

## Contact

For questions or issues, please open an issue in this repository or contact the maintainer at:
**[Wassemb@mail.tau.ac.il](mailto:Wassemb@mail.tau.ac.il)**

```

---

