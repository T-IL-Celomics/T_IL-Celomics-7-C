from pathlib import Path
import re
import shutil
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <PhaseName>")
    sys.exit(1)

phase_name = sys.argv[1]  # e.g. Phase1

# Assume ROOT_DIR is a sibling of this script file
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR / f"{phase_name}_DATA"
OUTPUT_DIR = SCRIPT_DIR / f"ACDC_IN_{phase_name}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regular expression to extract date/time info from filename
date_pattern = re.compile(r'(\d{4})y(\d{2})m(\d{2})d_(\d{2})h(\d{2})m')

# Build a position map to assign Position_1, Position_2, etc.
location_list = sorted({loc.name for chan in ROOT_DIR.iterdir() if chan.is_dir() for loc in chan.iterdir() if loc.is_dir()})
location_map = {name: f"Position_{i+1}" for i, name in enumerate(location_list)}

# Loop through channels (e.g. NIR, GREEN, ORANGE)
for channel_dir in ROOT_DIR.iterdir():
    if channel_dir.is_dir():
        channel_name = channel_dir.name  # NIR, GREEN, etc.
        
        # Loop through locations (e.g. B2, C3)
        for location_dir in channel_dir.iterdir():
            if location_dir.is_dir():
                location_name = location_dir.name  # B2, etc.
                position_folder_name = location_map[location_name]
                
                # Loop through fields (e.g. B2_1, B2_2)
                for field_dir in location_dir.iterdir():
                    if field_dir.is_dir():
                        field_name = field_dir.name  # B2_1, B2_2
                        basename = 'field1' if '_1' in field_name else 'field2'
                        
                        # List and sort image files
                        image_files = list(field_dir.glob('*.tif'))
                        
                        def extract_datetime(file):
                            match = date_pattern.search(file.name)
                            if match:
                                y, m, d, h, mi = map(int, match.groups())
                                return (y, m, d, h, mi)
                            else:
                                return (0, 0, 0, 0, 0)
                        
                        image_files.sort(key=extract_datetime)
                        
                        # Create output folder structure: ACDC_IN_PhaseX/ONE_or_TWO/Position_N/Images
                        group_folder = OUTPUT_DIR / basename / "Images"
                        group_folder.mkdir(parents=True, exist_ok=True)
                        
                        # Copy and rename files
                        for i, img_file in enumerate(image_files):
                            new_name = f"{basename}_{location_name}_{channel_name}_{i}.tif"
                            dest_path = group_folder / new_name
                            print(f"Copying {img_file.name} -> {dest_path}")
                            shutil.copy2(img_file, dest_path)