import pandas as pd
import os
import sys

# Get command-line arguments
output_dir = sys.argv[1]      # Output directory containing the combined Excel file
suffix = sys.argv[2]          # Suffix for file identification (e.g., F2)
filter_type = sys.argv[3]     # Filter type
num_channels = int(sys.argv[4])  # Number of channels (2 or 3)


# Construct input file path
file_path = os.path.join(output_dir, f"Gab_Normalized_Combined_{suffix}.xlsx")
output_path = ""
filter_func = None
# Set filtering logic and output file name based on filter_type and number of channels
if num_channels == 2:
    if filter_type == "PP":
        # Both channels must be 'Pos' or 'High'
        output_path = os.path.join(output_dir, f"Gab_Normalized_Combined_PP_{suffix}.xlsx")
        filter_func = lambda df: (
            df['Cha1_Category'].isin(['Pos', 'High']) &
            df['Cha2_Category'].isin(['Pos', 'High'])
        )
    elif filter_type == "PN":
        # Cha1 'Neg', Cha2 'Pos' or 'High'
        output_path = os.path.join(output_dir, f"Gab_Normalized_Combined_PN_{suffix}.xlsx")
        filter_func = lambda df: (
            (df['Cha1_Category'] == 'Neg') &
            df['Cha2_Category'].isin(['Pos', 'High'])
        )

elif num_channels == 3:
    if filter_type == "PPP":
        output_path = os.path.join(output_dir, f"Gab_Normalized_Combined_PPP_{suffix}.xlsx")
        filter_func = lambda df: (
            df['Cha1_Category'].isin(['Pos', 'High']) &
            df['Cha2_Category'].isin(['Pos', 'High']) &
            df['Cha3_Category'].isin(['Pos', 'High'])
        )
    elif filter_type == "PPN":
        output_path = os.path.join(output_dir, f"Gab_Normalized_Combined_PPN_{suffix}.xlsx")
        filter_func = lambda df: (
            (df['Cha1_Category'].isin(['Pos', 'High'])) &
            df['Cha2_Category'].isin(['Pos', 'High']) &
            (df['Cha3_Category'] == 'Neg')
        )
    elif filter_type == "PNP":
        output_path = os.path.join(output_dir, f"Gab_Normalized_Combined_PNP_{suffix}.xlsx")
        filter_func = lambda df: (
            df['Cha2_Category'].isin(['Pos', 'High']) &
            (df['Cha1_Category'] == 'Neg') &
            df['Cha3_Category'].isin(['Pos', 'High'])
        )
    

# Load the combined Excel file
xls = pd.ExcelFile(file_path)
MAX_SHEETNAME_LEN = 31  # Excel sheet name limit
required_cols = ['Cha1_Norm', 'Cha2_Norm'] if num_channels == 2 else ['Cha1_Norm', 'Cha2_Norm', 'Cha3_Norm']

# Process each sheet and apply the filter
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    for sheet_name in xls.sheet_names:
        print(f"Processing sheet: {sheet_name}")
        df = xls.parse(sheet_name)
        
        # Add category columns for each channel if not already present
        for ch in range(1, num_channels + 1):
            norm_col = f'Cha{ch}_Norm'
            cat_col = f'Cha{ch}_Category'
            if norm_col in df.columns and cat_col not in df.columns:
                df[cat_col] = pd.cut(
                df[norm_col],
                bins=[-float('inf'), -0.524, 0.524, float('inf')],
                labels=['Neg', 'Pos', 'High']
            )

        # Check that required columns exist
        if all(col in df.columns for col in required_cols):
            filtered_df = df[filter_func(df)]
            if not filtered_df.empty:
                safe_sheet_name = sheet_name[:MAX_SHEETNAME_LEN]
                filtered_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                print(f"Saved {len(filtered_df)} rows to sheet '{sheet_name}'.")
            else:
                print(f"No matching rows found in '{sheet_name}'.")
        else:
            print(f"Required columns missing in '{sheet_name}'. Writing original sheet.")
            safe_sheet_name = sheet_name[:MAX_SHEETNAME_LEN]
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            
print(f"\n✅ Filtering complete. Output saved to '{output_path}'")