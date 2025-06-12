import pandas as pd
import os
import sys
from scipy.stats import rankdata

"""
code_imaris_step1.py

Performs quantile normalization on intensity data from multiple channels in an Excel file,
assigns categories based on normalized values, and outputs:
    - Per-channel normalized and categorized data
    - A summary of quantile thresholds
    - A combined table with all normalized and category columns, as well as a combined weighted norm and combined category string

Usage:
    python code_imaris_step1.py <input_file_path> <output_dir> <suffix> <num_channels>

Arguments:
    input_file_path : str
        Path to the input Excel file (each channel in a separate sheet).
    output_dir : str
        Directory to save the output Excel file.
    suffix : str
        Suffix for output file naming (e.g., "F2").
    num_channels : int
        Number of channels (2 or 3).

Outputs:
    - Excel file named Segmented_Intensity_Cha1_Cha2_<suffix>.xlsx (or Cha1_Cha2_Cha3 if 3 channels)
      with the following sheets:
        * One sheet per channel: normalized and categorized data
        * Summary_Thresholds: table of 0.3 and 0.7 quantile values per channel
        * Combined_Norm: merged table with all normalized values, categories, combined weighted norm, and combined category string

Key Columns:
    - ChaX_Norm: Normalized value for channel X
    - ChaX_Category: Category for channel X ('Neg', 'Pos', 'High')
    - Combined_Weighted_Norm: Weighted average of all channel normalized values
    - Combined_Category: Concatenation of all channel categories (e.g., 'Neg/High/Pos')

Dependencies:
    - pandas
    - scipy
    - openpyxl or xlsxwriter (for Excel writing)

"""

# ===== Get paths from command line =====
input_file_path = sys.argv[1]  # Main Excel file
output_dir = sys.argv[2]       # Output directory
suffix = sys.argv[3]           # Suffix for identification (e.g., F2)
num_channels = int(sys.argv[4])  # number of channels (2 or 3)

os.makedirs(output_dir, exist_ok=True)

# Sheet names for Cha1 and Cha2 only
channels = {}
for i in range(1, num_channels + 1):
    channels[f'Cha{i}'] = f"Intensity Mean Ch={i}"

# Store processed data and summary
processed_data = {}
summary_rows = []

# Zscore normalization of protein intensity
def zscore_normalize(df, channel_label):
    mean = df['Value'].mean()
    std = df['Value'].std()
    zscores = (df['Value'] - mean) / std

    # Calculate quantiles for original values
    q03 = df['Value'].quantile(0.3)
    q07 = df['Value'].quantile(0.7)

    print(f"Z-score normalizing {channel_label} (mean={mean:.3f}, std={std:.3f})")
    summary_rows.append({
        'Channel': channel_label,
        'Q0.3_Value': q03,
        'Q0.7_Value': q07,
        'Mean': mean,
        'Std': std
    })
    df[f'{channel_label}_Norm'] = zscores

    # Add category column based on z-score
    cat_col = f'{channel_label}_Category'
    df[cat_col] = pd.cut(
        df[f'{channel_label}_Norm'],
        bins=[-float('inf'), -0.524, 0.524, float('inf')],  # ~0.3 and 0.7 quantiles for standard normal
        labels=['Neg', 'Pos', 'High']
    )

    return df

# Process each channel
for channel_label, sheet_name in channels.items():
    df = pd.read_excel(input_file_path, sheet_name=sheet_name)
    df = zscore_normalize(df, channel_label)
    processed_data[sheet_name] = df

# Create summary table
summary_df = pd.DataFrame(summary_rows)

# Merge normalized and category columns
channel_keys = list(channels.keys())
# Start with ID, Norm, and Category for the first channel
df_comb = processed_data[channels[channel_keys[0]]][['ID', f'{channel_keys[0]}_Norm', f'{channel_keys[0]}_Category']]
for ch in channel_keys[1:]:
    df_comb = df_comb.merge(
        processed_data[channels[ch]][['ID', f'{ch}_Norm', f'{ch}_Category']],
        on='ID'
    )

# Create combined category string (e.g., "Neg/High/Pos")
category_cols = [f"{ch}_Category" for ch in channel_keys]
df_comb['Combined_Category'] = df_comb[category_cols].astype(str).agg('/'.join, axis=1)
df_comb.reset_index(drop=True, inplace=True)

# Save results
channel_part = "_".join([f"Cha{i}" for i in range(1, num_channels + 1)])
output_file = os.path.join(
    output_dir,
    f"Segmented_Intensity_{channel_part}_{suffix}.xlsx"
)

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    for sheet, df in processed_data.items():
        df.to_excel(writer, sheet_name=sheet, index=False)
    summary_df.to_excel(writer, sheet_name="Summary_Thresholds", index=False)
    df_comb.to_excel(writer, sheet_name="Combined_Norm", index=False)

print(f"\n✅ Output file saved successfully: {output_file}")