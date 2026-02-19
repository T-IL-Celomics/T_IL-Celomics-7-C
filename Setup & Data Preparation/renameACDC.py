from pathlib import Path
import re
import shutil
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <PhaseName>")
    sys.exit(1)

phase_name = sys.argv[1]  # e.g. Phase1

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR / f"{phase_name}_DATA"
OUTPUT_DIR = SCRIPT_DIR / f"ACDC_IN_{phase_name}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regular expression to extract date/time info from filename
date_pattern = re.compile(r'(\d{4})y(\d{2})m(\d{2})d_(\d{2})h(\d{2})m')

def extract_datetime(file):
    match = date_pattern.search(file.name)
    if match:
        y, m, d, h, mi = map(int, match.groups())
        return (y, m, d, h, mi)
    return (0, 0, 0, 0, 0)

# -------------------------------------------------------
# First collect all fields across all channels
# -------------------------------------------------------

fields_dict = {}

for channel_dir in ROOT_DIR.iterdir():
    if not channel_dir.is_dir():
        continue

    channel_name = channel_dir.name

    for location_dir in channel_dir.iterdir():
        if not location_dir.is_dir():
            continue

        for field_dir in location_dir.iterdir():
            if not field_dir.is_dir():
                continue

            field_name = field_dir.name  # e.g. B2_1

            if field_name not in fields_dict:
                fields_dict[field_name] = []

            fields_dict[field_name].append((channel_name, field_dir))

# -------------------------------------------------------
# Process each field (B2_1, C3_2, etc.)
# -------------------------------------------------------

for field_name, channel_entries in fields_dict.items():

    # Decide if field1 or field2
    if field_name.endswith("_1"):
        parent_folder = "field1"
    elif field_name.endswith("_2"):
        parent_folder = "field2"
    else:
        print(f"Skipping unknown field format: {field_name}")
        continue

    # Create output folder: field1/B2_1/Images
    output_images_dir = OUTPUT_DIR / parent_folder / field_name / "Images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    for channel_name, field_dir in channel_entries:

        image_files = list(field_dir.glob("*.tif"))
        image_files.sort(key=extract_datetime)

        for i, img_file in enumerate(image_files):
            new_name = f"{field_name}_{channel_name}_{i}.tif"
            dest_path = output_images_dir / new_name

            print(f"Copying {img_file} -> {dest_path}")
            shutil.copy2(img_file, dest_path)

print("Done.")
