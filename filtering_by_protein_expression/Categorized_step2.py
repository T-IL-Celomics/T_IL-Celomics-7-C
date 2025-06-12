import pandas as pd
from openpyxl import load_workbook
import os
import sys


# ===== Get paths =====
input_file_path = sys.argv[1]       # Main input file
category_file_path = sys.argv[2]    # Category file
output_dir = sys.argv[3]            # Output directory
suffix = sys.argv[4]                # Suffix (e.g., F2)
num_channels = int(sys.argv[5])     # Number of channels (2 or 3)


os.makedirs(output_dir, exist_ok=True)

# ============ Step 1: Split into batch files ============

# Dynamically load category files for all channels
channel_lookups = {}
for i in range(1, num_channels + 1):
    sheet = f'Intensity Mean Ch={i}'
    df = pd.read_excel(category_file_path, sheet_name=sheet)
    channel_lookups[f'Cha{i}_Norm'] = df.set_index(['Time', 'Parent', 'ID'])[f'Cha{i}_Norm'].to_dict()


# Load sheet names
xls = pd.ExcelFile(input_file_path)
sheet_names = xls.sheet_names

batch_size = 5
batch_index = 1
sheets_processed = []

MAX_SHEETNAME_LEN = 31  

channels_tag = f"{num_channels}channels"

all_sheets = []

for i in range(0, len(sheet_names), batch_size):
    batch = sheet_names[i:i+batch_size]
    for sheet_name in batch:
        print(f"Processing sheet: {sheet_name}")
        df_preview = pd.read_excel(input_file_path, sheet_name=sheet_name, nrows=0)
        if {'Time', 'Parent', 'ID'}.issubset(df_preview.columns):
            df_full = pd.read_excel(input_file_path, sheet_name=sheet_name)
            print(f"  Original rows: {len(df_full)}")
            index = df_full.set_index(['Time', 'Parent', 'ID']).index
            for ch in range(1, num_channels + 1):
                norm_col = f'Cha{ch}_Norm'
                print(f"    Mapping {norm_col} for {sheet_name}")
                df_full[norm_col] = index.map(channel_lookups[norm_col].get)
                print(f"    Non-NaN in {norm_col}: {df_full[norm_col].notna().sum()}")
                cat_col = f'Cha{ch}_Category'
                df_full[cat_col] = pd.cut(
                    df_full[norm_col],
                    bins=[-float('inf'), -0.524, 0.524, float('inf')],
                    labels=['Neg', 'Pos', 'High']
                )
            print(f"  Rows after mapping: {len(df_full)}")
            
            if not df_full.empty:
                safe_sheet_name = sheet_name[:MAX_SHEETNAME_LEN]
                all_sheets.append((safe_sheet_name, df_full))
            else:
                print(f"  Skipped empty sheet: {sheet_name}")
        else:
            print(f"  Writing original sheet (missing columns): {sheet_name}")
            df_orig = pd.read_excel(input_file_path, sheet_name=sheet_name)
            all_sheets.append((sheet_name, df_orig))
            
print("Step 1 complete: All sheets processed in memory.")

# ============ Step 2: Write all sheets to one workbook ============

output_file = os.path.join(output_dir, f"Gab_Normalized_Combined_{suffix}.xlsx")
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    sheet_names_used = set()
    for sheet_name, df in all_sheets:
        final_sheet_name = sheet_name[:MAX_SHEETNAME_LEN]
        suffix_count = 1
        while final_sheet_name in sheet_names_used:
            final_sheet_name = f"{sheet_name}_{suffix_count}"
            suffix_count += 1
        sheet_names_used.add(final_sheet_name)
        df.to_excel(writer, sheet_name=final_sheet_name, index=False)

print(f"Combined file created successfully: {output_file}")


