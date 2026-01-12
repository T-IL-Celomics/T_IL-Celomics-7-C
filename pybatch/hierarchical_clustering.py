import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
from matplotlib import cm
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
import xlrd
import numpy as np
from fpdf import FPDF
import time
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import json
import cProfile
import io
import pstats
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

WORKING_WIDTH = 297
WORKING_HEIGHT = 180

CLUSTER_DROP_FIELDS = ["Parent", "Parent_OLD", "dt", "ID", "TimeIndex", "x_Pos", "y_Pos", "z_Pos", "Current_MSD_1",
                       "Final_MSD_1",
                       "Instantaneous_Speed_OLD", "Instantaneous_Angle_OLD", "Directional_Change_OLD", "Experiment",
                       "Acceleration_OLD"]

CLUSTER_ID_FIELDS = ["Area", "Displacement_From_Last_Id", "Instantaneous_Speed", "Velocity_X", "Velocity_Y",
                     "Velocity_Z", "Coll", "Coll_CUBE", "Acceleration", "Acceleration_X",
                     "Acceleration_Y", "Acceleration_Z", "Displacement2", "Directional_Change", "Directional_Change_X",
                     "Directional_Change_Y", "Directional_Change_Z", "Volume",
                     "Ellipticity_oblate", "Ellipticity_prolate", "Eccentricity", "Eccentricity_A", "Eccentricity_B",
                     "Eccentricity_C", "Sphericity", "EllipsoidAxisLengthB",
                     "EllipsoidAxisLengthC", "Ellip_Ax_B_X", "Ellip_Ax_B_Y", "Ellip_Ax_B_Z", "Ellip_Ax_C_X",
                     "Ellip_Ax_C_Y", "Ellip_Ax_C_Z", "Instantaneous_Angle",
                     "Instantaneous_Angle_X", "Instantaneous_Angle_Y", "Instantaneous_Angle_Z", "Min_Distance",
                     "IntensityCenterCh1", "IntensityCenterCh2", "IntensityCenterCh3", "IntensityMaxCh1",
                     "IntensityMaxCh2", "IntensityMaxCh3", "IntensityMeanCh1", "IntensityMeanCh2", "IntensityMeanCh3",
                     "IntensityMedianCh1", "IntensityMedianCh2", "IntensityMedianCh3", "IntensitySumCh1",
                     "IntensitySumCh2", "IntensitySumCh3"]
CLUSTER_CELL_FIELDS = ["Overall_Displacement", "Total_Track_Displacement", "Track_Displacement_X",
                       "Track_Displacement_Y", "Track_Displacement_Z",
                       "Linearity_of_Forward_Progression", "Mean_Curvilinear_Speed", "Mean_Straight_Line_Speed",
                       "Confinement_Ratio", "MSD_Directed_v2",
                       "MSD_Linearity_R2_Score", "MSD_Brownian_Motion_BIC_Score", "MSD_Brownian_D",
                       "MSD_Directed_Motion_BIC_Score", "MSD_Directed_D",
                       "Velocity_Time_of_Maximum_Height", "Velocity_Maximum_Height"]
CLUSTER_WAVE_FIELDS = ["Velocity_Full_Width_Half_Maximum", "Velocity_Ending_Value", "Velocity_Ending_Time",
                       "Velocity_Starting_Value",
                       "Velocity_Starting_Time"]


def list_names(dir_path: str, recursive: bool = False):
    p = Path(dir_path)
    it = p.rglob("*") if recursive else p.iterdir()
    return sorted([x.name for x in it if x.is_file()])


def hex_to_rgba(hex_color):
    """Convert hex color (with or without alpha) to RGBA tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        # RGB only
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)) + (1.0,)
    elif len(hex_color) == 8:
        # RGBA
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6))
    return (0, 0, 0, 1)


def abbreviate_combo(combo_str):
    """Convert combo string like 'Neg_Pos_High' to 'NPH'."""
    abbrev_map = {
        "Neg": "N",
        "Pos": "P",
        "High": "H",
        "NA": "A"
    }
    parts = combo_str.split("_")
    return "".join(abbrev_map.get(p, p[0].upper()) for p in parts)


class heirarchical_clustering:

    def __init__(self, exp_rename_dir, summary_table_folder):

        self.wells = []  # list of well names
        self.well_info = {}  # dict of well_name -> pd.DataFrame
        self.exp_rename_dir = exp_rename_dir
        self.summary_table_folder = summary_table_folder
        self.summary_table = None
        self.shortened_well_names = {}

    def _infer_dose_by_channel_from_df(self, well_df):
        """
        returns dict like {"ch1":"POS", "ch2":"NEG"} based on mode over time.
        if a channel column doesn't exist, it's ignored.
        """

        cols = [c for c in ["Cha1_Category", "Cha2_Category", "Cha3_Category"] if c in well_df.columns]

        if not cols or well_df.empty:
            return []

        # ensure no NaNs kill combos
        tmp = well_df[cols].fillna("NA")

        # build combos row-wise, but keep only the channels that exist in this df
        combos = tmp.apply(lambda r: "_".join([str(r[c]) for c in cols]), axis=1)

        # unique combos, drop the all-NA combo if you want
        combos = combos.unique().tolist() if hasattr(combos, 'unique') else list(set(combos))
        # optional: remove all-NA
        all_na = "_".join(["NA"] * len(cols))
        combos = [c for c in combos if c != all_na]

        return combos

    def load_wells_from_summary_table(self):
        """
        Load wells from individual summary_table_<experiment>_FULL.xlsx files in exp_rename_dir.
        """
        print("Loading wells from individual summary_table_*_FULL.xlsx files...")
        print(f"Looking in directory: {self.summary_table_folder}")

        # List all summary_table_*_FULL.xlsx files
        files = list_names(self.summary_table_folder)
        exp_full_files = [f for f in files if f.startswith("summary_table_") and f.endswith("_FULL.xlsx")]
        print(f"Found {len(exp_full_files)} summary_table_*_FULL.xlsx files")
        print(f"Sample files: {exp_full_files[:3]}")

        if len(exp_full_files) == 0:
            print("ERROR: No summary_table_*_FULL.xlsx files found!")
            self.wells = []
            self.parameters = []
            return

        # Extract well names from filenames (remove "summary_table_" and "_FULL.xlsx")
        self.wells = [f.replace("summary_table_", "").replace("_FULL.xlsx", "") for f in exp_full_files]
        print(f"Extracted {len(self.wells)} well names")
        print(f"Sample wells: {self.wells[:3]}")

        self.well_amount = len(self.wells)
        self.well_info = {}

        # Load data from each summary_table_*_FULL.xlsx file
        print("\nLoading individual well data files...")
        for i, well in enumerate(self.wells):
            file_path = os.path.join(self.summary_table_folder, f"summary_table_{well}_FULL.xlsx")
            print(f"  [{i + 1}/{len(self.wells)}] Loading {well}...")

            try:
                well_data = pd.read_excel(file_path)
                self.well_info[well] = well_data
                print(f"    Shape: {well_data.shape}, Columns: {len(well_data.columns)}")
            except Exception as e:
                print(f"    ERROR loading file: {e}")
                self.well_info[well] = pd.DataFrame()

        # Extract parameters from first well's columns
        if self.wells and not self.well_info[self.wells[0]].empty:
            self.parameters = list(self.well_info[self.wells[0]].columns)
            print(f"\nExtracted {len(self.parameters)} parameters from first well")
            print(f"Sample parameters: {self.parameters[:5]}")
        else:
            self.parameters = []
            print("WARNING: No valid data loaded!")

        # Check for z_Pos to determine dimensions
        if "z_Pos" in self.parameters:
            self.dimensions = 3
        else:
            self.dimensions = 2
        print(f"Data dimensions: {self.dimensions}D")

        # Create shortened well names for display
        if len(self.wells) > 0:
            try:
                self.shortened_well_names = {well: well[18:-4].replace("NNN0", "") if len(well) > 22 else well for well
                                             in self.wells}
                all_short_names = list(self.shortened_well_names.values())
                print(f"\nShortened names sample: {all_short_names[:3]}")

                # Check if all start with same pattern and shorten further if needed
                if all_short_names and len(all_short_names[0]) >= 4:
                    first_val = all_short_names[0][:4]
                    if all(len(name) >= 4 and name[:4] == first_val for name in all_short_names):
                        print("All wells have same prefix, using alternative slicing")
                        self.shortened_well_names = {well: well[22:-4].replace("NNN0", "") if len(well) > 26 else well
                                                     for well in self.wells}
                        all_short_names = list(self.shortened_well_names.values())

                # Check for duplicates
                if all_short_names:
                    duplicate_names = [name for name in all_short_names if all_short_names.count(name) > 1]
                    if duplicate_names:
                        print(f"Found duplicate shortened names: {set(duplicate_names)}, adding well position")
                        for well in self.wells:
                            self.shortened_well_names[well] = well[15:18] + self.shortened_well_names[well] if len(
                                well) > 18 else well

                print(f"Final shortened names: {list(self.shortened_well_names.values())[:5]}")
            except Exception as e:
                print(f"Error creating shortened names: {e}, using full well names instead")
                self.shortened_well_names = {well: well for well in self.wells}
        else:
            print("No wells loaded!")

        # Sort wells by shortened name
        self.wells.sort(key=lambda x: self.shortened_well_names.get(x, x))
        print(f"\nTotal wells loaded: {len(self.wells)}")

    def add_category_to_well_info(self):
        for well in self.wells:

            print(f"\n=== ADD CATEGORY: well={well} ===")

            well_df = self.well_info[well].copy()
            print("initial well_df shape:", well_df.shape)
            print("initial columns:", list(well_df.columns))

            # ensure category columns exist
            for i in (1, 2, 3):
                col = f"Cha{i}_Category"
                if col not in well_df.columns:
                    well_df[col] = np.nan

            fname = f"{well}.xlsx"
            files = list_names(self.exp_rename_dir)
            print("looking for file:", fname)
            print("found:", fname in files)

            if fname not in files:
                print("file not found → only NaNs added")
                self.well_info[well] = well_df
                continue

            path = os.path.join(self.exp_rename_dir, fname)
            acc_df = pd.read_excel(path, sheet_name="Acceleration")

            print("acc_df shape:", acc_df.shape)
            print("acc_df columns:", list(acc_df.columns))

            # normalize column names
            rename_map = {}
            if "Time" in acc_df.columns and "TimeIndex" not in acc_df.columns:
                rename_map["Time"] = "TimeIndex"
            if "timeindex" in acc_df.columns and "TimeIndex" not in acc_df.columns:
                rename_map["timeindex"] = "TimeIndex"
            if "parent" in acc_df.columns and "Parent" not in acc_df.columns:
                rename_map["parent"] = "Parent"

            if rename_map:
                print("renaming columns:", rename_map)
                acc_df = acc_df.rename(columns=rename_map)

            key = ["Parent", "TimeIndex"]
            missing = [c for c in key if c not in acc_df.columns or c not in well_df.columns]
            print("missing key columns:", missing)

            if missing:
                print("merge aborted due to missing keys")
                self.well_info[well] = well_df
                continue

            for k in key:
                well_df[k] = pd.to_numeric(well_df[k], errors="coerce").astype("Int64")
                acc_df[k] = pd.to_numeric(acc_df[k], errors="coerce").astype("Int64")

            cat_cols = [f"Cha{i}_Category" for i in (1, 2, 3) if f"Cha{i}_Category" in acc_df.columns]
            print("category columns found in acc_df:", cat_cols)

            if not cat_cols:
                print("no category columns in acc_df → nothing to merge")
                self.well_info[well] = well_df
                continue

            acc_small = acc_df[key + cat_cols].drop_duplicates(subset=key)
            print("acc_small shape:", acc_small.shape)

            merged = well_df.merge(acc_small, on=key, how="left", suffixes=("", "_raw"))
            print("merged shape:", merged.shape)

            for c in cat_cols:
                raw = f"{c}_raw"
                if raw in merged.columns:
                    n_filled = merged[raw].notna().sum()
                    print(f"merging {c}: filled {n_filled} rows")
                    merged[c] = merged[raw].combine_first(merged[c])
                    merged.drop(columns=[raw], inplace=True)

            print("final columns:", list(merged.columns))
            print("non-null counts:")
            print(merged[cat_cols].notna().sum())

            for i in (1, 2, 3):
                c = f"Cha{i}_Category"
                if c in merged.columns:
                    merged[c] = merged[c].fillna("NA")

            self.well_info[well] = merged

            print("=== DONE ===\n")

    def draw_cluster_analysis_by_treatment_dose(self, output_folder=None):
        indexes = []
        well_combos = {}
        for well in self.wells:
            well_combos[well] = self._infer_dose_by_channel_from_df(self.well_info[well])

        print("\n=== DEBUG combos ===")
        print("num wells:", len(self.wells))

        lens = {w: len(well_combos[w]) for w in self.wells}
        print("combos per well (min/mean/max):",
              min(lens.values()) if lens else None,
              (sum(lens.values()) / len(lens)) if lens else None,
              max(lens.values()) if lens else None)

        # print a few examples
        for w in list(self.wells):
            print("well:", w)
            print("  columns present:",
                  [c for c in ["Cha1_Category", "Cha2_Category", "Cha3_Category"] if c in self.well_info[w].columns])
            print(" combos:", well_combos[w][:10])

        for well in self.wells:
            for combo in well_combos[well]:
                indexes.append(f"{well}_{combo}")

        print("\n=== DEBUG indexes ===")
        print("num indexes:", len(indexes))
        print("first 10 indexes:", indexes[:10])
        print("last 10 indexes:", indexes[-10:])

        avg_df = pd.DataFrame(index=indexes, columns=self.parameters)
        avg_df.drop(columns=CLUSTER_DROP_FIELDS, inplace=True, errors="ignore")
        for well in self.wells:
            well_df = self.well_info[well]
            for combo in well_combos[well]:
                combo_list = combo.split("_")  # get only dose part
                filtered_df = None
                if len(combo_list) == 1:
                    filtered_df = well_df[(well_df['Cha1_Category'] == combo_list[0])]
                elif len(combo_list) == 2:
                    filtered_df = well_df[
                        (well_df['Cha1_Category'] == combo_list[0]) & (well_df['Cha2_Category'] == combo_list[1])]
                elif len(combo_list) == 3:
                    filtered_df = well_df[(well_df['Cha1_Category'] == combo_list[0]) &
                                          (well_df['Cha2_Category'] == combo_list[1]) &
                                          (well_df['Cha3_Category'] == combo_list[2])]
                else:
                    raise ValueError("wrong number of channels in combo for clustering.")
                if filtered_df.empty: continue
                for parameter in CLUSTER_ID_FIELDS:
                    if parameter in self.parameters:
                        parameter_array = filtered_df[parameter].dropna()
                        avg_df.loc[f"{well}_{combo}", parameter] = np.average(parameter_array)
                cells = filtered_df.Parent.unique()
                indexed_filtered = filtered_df.set_index("Parent")
                for parameter in CLUSTER_CELL_FIELDS + CLUSTER_WAVE_FIELDS:
                    if parameter in self.parameters:
                        cell_data = indexed_filtered[parameter].groupby(level=0).first()

                        if parameter in CLUSTER_WAVE_FIELDS:
                            cell_data = cell_data[cell_data != 0]

                        avg_df.loc[f"{well}_{combo}", parameter] = cell_data.mean()
        scaler = StandardScaler(with_std=True)

        new_index = []
        for old_idx in avg_df.index:
            well_part, combo_part = old_idx.split("_", 1)
            short_name = self.shortened_well_names.get(well_part, well_part)
            new_index.append(f"{short_name}_{combo_part}")

        if avg_df.shape[0] == 0:
            print("No (well, combo) indexes were generated -> skipping clustering page")
            return

        # Remove columns with all NaN values
        print(f"\nBefore cleaning: avg_df shape = {avg_df.shape}")
        print(f"NaN counts per column (top 10):")
        nan_counts = avg_df.isna().sum().sort_values(ascending=False)
        print(nan_counts.head(10))

        # Drop columns that are all NaN
        avg_df = avg_df.dropna(axis=1, how='all')
        print(f"After dropping all-NaN columns: avg_df shape = {avg_df.shape}")

        # Fill remaining NaN values with 0
        avg_df = avg_df.fillna(0)

        # Remove columns with zero variance (same value everywhere)
        print("\nChecking for zero-variance columns...")
        variances = avg_df.var(numeric_only=True)
        zero_var_cols = variances[variances == 0].index.tolist()
        if zero_var_cols:
            print(f"Found {len(zero_var_cols)} zero-variance columns, removing them")
            avg_df = avg_df.drop(columns=zero_var_cols)
        print(f"After removing zero-variance columns: avg_df shape = {avg_df.shape}")

        # Scale the data
        print(f"\nScaling data with {avg_df.shape[0]} rows and {avg_df.shape[1]} columns")
        scaled_data = scaler.fit_transform(avg_df)

        # Check for NaN or infinite values after scaling
        if np.isnan(scaled_data).any():
            print("WARNING: NaN values found after scaling!")
            nan_mask = np.isnan(scaled_data)
            print(f"NaN values per column: {nan_mask.sum(axis=0)}")
            # Replace NaN with 0
            scaled_data = np.nan_to_num(scaled_data, nan=0.0)

        if np.isinf(scaled_data).any():
            print("WARNING: Infinite values found after scaling!")
            # Replace infinite values with large but finite values
            scaled_data = np.where(np.isinf(scaled_data), np.sign(scaled_data) * 1e10, scaled_data)

        avg_df = pd.DataFrame(scaled_data, columns=avg_df.columns, index=new_index)
        print(f"Final avg_df shape before linkage: {avg_df.shape}")
        print(f"Data contains NaN: {avg_df.isna().any().any()}")
        print(f"Data contains inf: {np.isinf(avg_df.values).any()}")

        linkaged_pca = linkage(avg_df, "ward")

        # ---- category colors (fixed, always same) ----
        cat_color_hex = {
            "Neg": "#000000FF",      # Black
            "Pos": "#808080FF",      # Gray
            "High": "#FFFFFFFF",     # White
            "NA": "#fc0202c3"        # Red
        }
        # Convert hex colors to RGBA tuples for proper matplotlib rendering
        cat_color = {k: hex_to_rgba(v) for k, v in cat_color_hex.items()}

        row_names = list(avg_df.index)  # these are your plotted rows: "<exp>_<combo>"

        # experiment = text before first "_" in the row label
        exp_names = [rn.split("_", 1)[0] for rn in row_names]
        unique_exps = sorted(set(exp_names))

        print(f"\n=== EXPERIMENT GROUPING DEBUG ===")
        print(f"Total row names: {len(row_names)}")
        print(f"Unique experiments (before grouping): {len(unique_exps)}")
        print(f"Sample row names: {row_names[:5]}")
        print(f"Unique experiments: {unique_exps}")

        # Group experiments by well letter + experiment name (ignoring numbers in between)
        # e.g., c03METRNNIRNOCO and c02METRNNIRNOCO group as "cMETRNNIRNOCO"
        import re
        def get_exp_base(exp_name):
            """Extract well letter + experiment name, removing numbers in between."""
            # Match: letter(s) followed by any numbers, then the rest
            match = re.match(r'^([a-z]+)\d+(.*)$', exp_name, re.IGNORECASE)
            if match:
                return match.group(1) + match.group(2)  # well letter + experiment name
            return exp_name  # fallback if pattern doesn't match

        exp_bases = {e: get_exp_base(e) for e in unique_exps}
        unique_bases = sorted(set(exp_bases.values()))

        print(f"Experiment bases (after grouping): {len(unique_bases)}")
        print(f"Experiment base mapping:")
        for exp, base in sorted(exp_bases.items()):
            print(f"  {exp} -> {base}")

        # give each unique experiment base a unique color
        exp_palette = sns.color_palette("tab20", n_colors=max(3, len(unique_bases)))
        base_color = {b: exp_palette[i] for i, b in enumerate(unique_bases)}

        print(f"Base color assignment:")
        for base, color in sorted(base_color.items()):
            print(f"  {base} -> {color}")

        # Each experiment gets the color of its base
        exp_color = {e: base_color[exp_bases[e]] for e in unique_exps}

        exp_colors = pd.Series([exp_color[e] for e in exp_names], index=row_names, name="Experiment")
        print(f"=== END EXPERIMENT GROUPING ===\n")

        def parse_combo(rn: str):
            _, combo = rn.split("_", 1)
            parts = combo.split("_")
            parts = (parts + ["NA", "NA", "NA"])[:3]
            return parts[0], parts[1], parts[2]

        cha1_colors = pd.Series([cat_color.get(parse_combo(rn)[0], cat_color["NA"]) for rn in row_names],
                                index=row_names, name="Cha1")
        cha2_colors = pd.Series([cat_color.get(parse_combo(rn)[1], cat_color["NA"]) for rn in row_names],
                                index=row_names, name="Cha2")
        cha3_colors = pd.Series([cat_color.get(parse_combo(rn)[2], cat_color["NA"]) for rn in row_names],
                                index=row_names, name="Cha3")

        row_colors = pd.concat([exp_colors, cha1_colors, cha2_colors, cha3_colors], axis=1)

        s = sns.clustermap(data=avg_df, row_linkage=linkaged_pca, cmap=sns.color_palette("coolwarm", n_colors=256),
                           vmin=-2, vmax=2, figsize=(30, 15),
                           cbar_kws=dict(use_gridspec=False),
                           row_colors=row_colors)
        # s.ax_heatmap.set_xlabel("Parameters", fontsize=25, fontweight='bold')
        # s.ax_heatmap.set_ylabel("Well", fontsize=25, fontweight='bold')
        # s.cax.set_yticklabels(s.cax.get_yticklabels());
        # pos = s.ax_heatmap.get_position();
        # cbar = s.cax
        # cbar.set_position([0.02, pos.bounds[1], 0.02, pos.bounds[3]]);

        # move row_colors labels to the top
        ax_rc = s.ax_row_colors

        ax_rc.xaxis.set_ticks_position("top")
        ax_rc.xaxis.set_label_position("top")

        ax_rc.set_xticklabels(
            ax_rc.get_xticklabels(),
            rotation=0,
            ha="center",
            fontsize=8
        )

        handles = []

        print(f"\n=== LEGEND CREATION DEBUG ===")
        print(f"Creating legend entries for {len(unique_bases)} experiment bases:")

        # experiment legend grouped by base experiment (well letter + name, ignoring numbers)
        for base in unique_bases:
            exps_in_base = sorted([e for e in unique_exps if exp_bases[e] == base])
            print(f"  Base: {base} -> variants: {exps_in_base}")
            if len(exps_in_base) == 1:
                label = f"{exps_in_base[0]}"
                handles.append(patches.Patch(facecolor=base_color[base], label=label))
                print(f"    Label: {label}")
            else:
                label = f"{base} ({', '.join(exps_in_base)})"
                handles.append(patches.Patch(facecolor=base_color[base], label=label))
                print(f"    Label: {label}")

        # category legend (fixed mapping)
        for k in ["Neg", "Pos", "High", "NA"]:
            handles.append(patches.Patch(facecolor=cat_color[k], label=k))

        print(f"Total legend handles: {len(handles)}")
        print(f"=== END LEGEND CREATION ===\n")

        # Raise the legend higher (move up)
        s.fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 1.15),  # Raised legend higher
            fontsize=8
        )

        # Color the column names (parameters) based on parameter type
        col_names = list(avg_df.columns)

        # Define morphological and movement parameters
        morphological_params = [
            "Area", "Volume",
            "Ellipticity_oblate", "Ellipticity_prolate", "Eccentricity", "Eccentricity_A", "Eccentricity_B",
            "Eccentricity_C",
            "Sphericity", "EllipsoidAxisLengthB", "EllipsoidAxisLengthC",
            "Ellip_Ax_B_X", "Ellip_Ax_B_Y", "Ellip_Ax_B_Z", "Ellip_Ax_C_X", "Ellip_Ax_C_Y", "Ellip_Ax_C_Z"
        ]

        movement_params = [
            "Displacement_From_Last_Id", "Instantaneous_Speed", "Velocity_X", "Velocity_Y", "Velocity_Z",
            "Coll", "Coll_CUBE", "Acceleration", "Acceleration_X", "Acceleration_Y", "Acceleration_Z",
            "Displacement2", "Directional_Change", "Directional_Change_X", "Directional_Change_Y",
            "Directional_Change_Z",
            "Instantaneous_Angle", "Instantaneous_Angle_X", "Instantaneous_Angle_Y", "Instantaneous_Angle_Z",
            "Min_Distance",
            "Overall_Displacement", "Total_Track_Displacement", "Track_Displacement_X", "Track_Displacement_Y",
            "Track_Displacement_Z",
            "Linearity_of_Forward_Progression", "Mean_Curvilinear_Speed", "Mean_Straight_Line_Speed",
            "Confinement_Ratio", "MSD_Directed_v2", "MSD_Linearity_R2_Score", "MSD_Brownian_Motion_BIC_Score",
            "MSD_Brownian_D", "MSD_Directed_Motion_BIC_Score", "MSD_Directed_D",
            "Velocity_Time_of_Maximum_Height", "Velocity_Maximum_Height",
            "Velocity_Full_Width_Half_Maximum", "Velocity_Ending_Value", "Velocity_Ending_Time",
            "Velocity_Starting_Value", "Velocity_Starting_Time"
        ]

        # Map colors based on parameter type
        param_colors = {}
        for col in col_names:
            if col in morphological_params:
                param_colors[col] = "#000000"  # Black for morphological
            elif col in movement_params:
                param_colors[col] = "#1f77b4"  # Blue for movement
            else:
                param_colors[col] = "#7f7f7f"  # Gray for others (intensity, etc.)

        # Apply colors to column labels
        for i, label in enumerate(s.ax_heatmap.get_xticklabels()):
            col_name = col_names[i] if i < len(col_names) else str(label)
            color = param_colors.get(col_name, "#000000")
            label.set_color(color)

        s.ax_heatmap.set_xticklabels(s.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')

        ax = s.ax_heatmap
        ax.set_yticks(np.arange(len(avg_df.index)) + 0.5)  # +0.5 is important for clustermap
        
        # Abbreviate row names for display
        abbreviated_labels = []
        for idx in avg_df.index:
            well_part, combo_part = idx.split("_", 1)
            short_name = self.shortened_well_names.get(well_part, well_part)
            short_combo = abbreviate_combo(combo_part)
            abbreviated_labels.append(f"{short_name}_{short_combo}")
        
        ax.set_yticklabels(abbreviated_labels, fontsize=6)

        if output_folder:
            plt.savefig(os.path.join(output_folder, "clustermap_treatment_dose.jpg"), bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.suptitle(" By Dose Cluster Analysis" , fontweight='bold', fontsize=30)
            plt.show()


if __name__ == "__main__":
    # Update these paths with your actual data locations
    exp_rename_dir = r"D:\\jeries\\renaming"  # Path to folder with renamed experiment files
    summary_table_folder = r"D:\\jeries\\output_test_2"  # Path to summary table
    output_folder = r"D:\\jeries\\output_test_2"  # Where to save clustering results

    clustering = heirarchical_clustering(exp_rename_dir=exp_rename_dir, summary_table_folder=summary_table_folder)
    clustering.load_wells_from_summary_table()
    clustering.add_category_to_well_info()
    clustering.draw_cluster_analysis_by_treatment_dose(output_folder=output_folder)

