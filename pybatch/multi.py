import pandas as pd
import os

import streamlit as st
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
from sklearn import decomposition
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import json
import cProfile
import io
import pstats
import warnings

UNIT_DICT = {'Instantaneous_Speed': ' [$\\mu$m/h]',
             'Velocity_X': ' [$\\mu$m/h]',
             'Velocity_Y': ' [$\\mu$m/h]',
             'Velocity_Z': ' [$\\mu$m/h]',
             'Acceleration': ' [$\\mu$m/$h^2$]',
             'Acceleration_X': ' [$\\mu$m/$h^2$]',
             'Acceleration_Y': ' [$\\mu$m/$h^2$]',
             'Acceleration_Z': ' [$\\mu$m/$h^2$]',
             'Coll': '',
             'Coll_CUBE': '',
             'Displacement2': '[$\\mu$$m^2$]',
             'Directional_Change': ' [radians]',
             'Volume': '[$\\mum^3$]',
             'Ellipticity_oblate': '',
             'Ellipticity_prolate': '',
             'Eccentricity': '',
             'Sphericity': '',
             'EllipsoidAxisLengthB': ' [$\\mu$m]',
             'EllipsoidAxisLengthC': ' [$\\mu$m]',
             'Ellip_Ax_B_X': '',
             'Ellip_Ax_B_Y': '',
             'Ellip_Ax_C_X': '',
             'Ellip_Ax_C_Y': '',
             'Instantaneous_Angle': ' [radians]',
             'Velocity_Full_Width_Half_Maximum': ' [min]',
             'Velocity_Time_of_Maximum_Height': ' [min]',
             'Velocity_Maximum_Height': ' [$\\mu$m/h]',
             'Velocity_Ending_Value': ' [$\\mu$m/h]',
             'Velocity_Ending_Time': ' [min]',
             'Velocity_Starting_Value': ' [$\\mu$m/h]',
             'Velocity_Starting_Time': ' [min]',
             'Area': ' [$\\mu$$m^2$]',
             'Overall_Displacement': ' [$\\mu$m]',
             'Total_Track_Displacement': ' [$\\mu$m]',
             'Track_Displacement_X': ' [$\\mu$m]',
             'Track_Displacement_Y': ' [$\\mu$m]',
             'Track_Displacement_Z': ' [$\\mu$m]',
             'Linearity_of_Forward_Progression': '',
             'Confinement_Ratio': '',
             'Mean_Curvilinear_Speed': ' [$\\mu$m/h]',
             'Mean_Straight_Line_Speed': ' [$\\mu$m/h]',
             'Current_MSD_1': ' [$\\mu$m]',
             'Final_MSD_1': ' [$\\mu$m]',
             'MSD_Linearity_R2_Score': '',
             'MSD_Brownian_Motion_BIC_Score': '',
             'MSD_Directed_Motion_BIC_Score': '',
             'MSD_Brownian_D': '',
             'MSD_Directed_D': '',
             'MSD_Directed_v2': '',
             'Min_Distance': ' [$\\mu$m]',
             'Displacement_From_Last_Id': ' [$\\mu$m]',
             'IntensityCenterCh1': '',
             'IntensityCenterCh2': '',
             'IntensityMaxCh1': '',
             'IntensityMaxCh2': '',
             'IntensityMeanCh1': '',
             'IntensityMeanCh2': '',
             'IntensityMedianCh1': '',
             'IntensityMedianCh2': '',
             'IntensitySumCh1': '',
             'IntensitySumCh2': '',
             'Vx / Vy': '',
             'Eccentricity_A': '',
             'Eccentricity_B': '',
             'Eccentricity_C': '',
             'Acceleration_OLD': ' [$\\mu$m/$h^2$]',
             'Directional_Change_OLD': ' [radians]',
             'Directional_Change_X': ' [radians]',
             'Directional_Change_Y': ' [radians]',
             'Directional_Change_Z': ' [radians]',
             'Instantaneous_Angle_OLD': ' [radians]',
             'Instantaneous_Angle_X': ' [radians]',
             'Instantaneous_Angle_Y': ' [radians]',
             'Instantaneous_Angle_Z': ' [radians]',
             'Instantaneous_Speed_OLD': ' [$\\mu$m/h]',
             'Ellip_Ax_B_Z': '',
             'Ellip_Ax_C_Z': '',
             'Mean_Speed': ' [$\\mu$m/h]',
             'Track_Displacement_Length': ' [$\\mu$m]',
             'Velocity': ' [$\\mu$m/h]'}

TIME_FIELDS = ["Area",
               "Acceleration",
               "Acceleration_X",
               "Acceleration_Y",
               "Acceleration_Z",
               "Coll",
               "Coll_CUBE",
               "Directional_Change",
               "Directional_Change_X",
               "Directional_Change_Y",
               "Directional_Change_Z",
               "Displacement_From_Last_Id",
               "Displacement2",
               "Ellip_Ax_B_X",
               "Ellip_Ax_B_Y",
               "Ellip_Ax_B_Z",
               "Ellip_Ax_C_X",
               "Ellip_Ax_C_Y",
               "Ellip_Ax_C_Z",
               "EllipsoidAxisLengthB",
               "EllipsoidAxisLengthC",
               "Ellipticity_oblate",
               "Ellipticity_prolate",
               "IntensityCenterCh1",
               "IntensityCenterCh2",
               "IntensityMaxCh1",
               "IntensityMaxCh2",
               "IntensityMeanCh1",
               "IntensityMeanCh2",
               "IntensityMedianCh1",
               "IntensityMedianCh2",
               "IntensitySumCh1",
               "IntensitySumCh2",
               "Instantaneous_Angle",
               "Instantaneous_Angle_X",
               "Instantaneous_Angle_Y",
               "Instantaneous_Angle_Z",
               "Instantaneous_Speed",
               "Sphericity",
               "Velocity",
               "Velocity_X",
               "Velocity_Y",
               "Velocity_Z",
               "Eccentricity",
               "Eccentricity_A",
               "Eccentricity_B",
               "Eccentricity_C",
               "Min_Distance"]

IRRELEVANT_FIELDS = ["Parent", "ID", "Parent_OLD", "Experiment", "TimeIndex", "dt", "Unnamed: 0", "x_Pos", "y_Pos",
                     "z_Pos"]

CLUSTER_WAVE_FIELDS = ["Velocity_Full_Width_Half_Maximum", "Velocity_Ending_Value", "Velocity_Ending_Time",
                       "Velocity_Starting_Value", "Velocity_Starting_Time"]

INFO_LOCATIONS = [0, 2, 5, 11, 14, 15, 18, 22, 26, 30, 34, 38, 42, 46]

WORKING_WIDTH = 297
WORKING_HEIGHT = 190

TWO_GRAPH_X = [0, 148.5]
TWO_GRAPH_Y = [20, 20]
TWO_GRAPH_WIDTH = 148.5
TWO_GRAPH_HEIGHT = 190
FOUR_GRAPH_X = [0, 148.5, 0, 148.5]
FOUR_GRAPH_Y = [20, 20, 115, 115]
FOUR_GRAPH_WIDTH = 148.5
FOUR_GRAPH_HEIGHT = 95

PARAM_GRAPHS = ["log_means.jpg", "zscore.jpg"]
PCA_GRAPHS = ["variance_explained.jpg", "pca_biplot.jpg"]
CLUSTER_GRAPHS = ["default_silhouette.jpg", "kmeans_silhouette.jpg",
                  "pca_scatter_default.jpg", "pca_scatter_kmeans.jpg"]

LINE_COLORS = list(colors.TABLEAU_COLORS.keys()) + list(colors.XKCD_COLORS.keys())[:30]

################## PDF related functions ###################

class PDF(FPDF):

    def set_automatic_coords(self, amount):
        for row_num in range(1, 5):
            current_row_width = WORKING_WIDTH / int(np.ceil(amount / row_num))
            another_row_width = WORKING_HEIGHT / (row_num + 1)
            if another_row_width > current_row_width:
                self.row_num = row_num + 1
            else:
                self.row_num = row_num
                break
        self.graph_width = min(WORKING_WIDTH / int(np.ceil(amount / self.row_num)),
                               WORKING_HEIGHT / self.row_num)
        self.graph_x_coords = []
        self.graph_y_coords = []
        last_row_graphs = amount
        height_break = (WORKING_HEIGHT - self.graph_width * self.row_num) / self.row_num
        # find coords for all rows except last:
        if self.row_num > 1:
            full_row_graphs = int(np.ceil(amount / self.row_num))
            last_row_graphs -= full_row_graphs * (self.row_num - 1)
            full_row_break = (WORKING_WIDTH - (self.graph_width * full_row_graphs)) / (full_row_graphs - 1)
            for i in range(self.row_num - 1):
                self.graph_x_coords += [(full_row_break + self.graph_width) * j for j in range(full_row_graphs)]
                self.graph_y_coords += [30 + (height_break / 2) + (
                        (self.graph_width + height_break) * i)] * full_row_graphs
        # find coords for last row:
        if last_row_graphs < amount / self.row_num:
            last_row_break = (WORKING_WIDTH - (self.graph_width * last_row_graphs)) / last_row_graphs
            self.graph_x_coords += [(last_row_break / 2) + (self.graph_width + last_row_break) * i for i in
                                    range(last_row_graphs)]
        else:
            last_row_break = (WORKING_WIDTH - (self.graph_width * last_row_graphs)) / (last_row_graphs - 1)
            self.graph_x_coords += [(self.graph_width + last_row_break) * i for i in range(last_row_graphs)]
        self.graph_y_coords += [30 + (height_break / 2) + (
                (self.graph_width + height_break) * (self.row_num - 1))] * last_row_graphs

    def set_params(self, exp_name, well_amount):
        self.exp_name = exp_name
        for size in range(20, 5, -1):
            self.set_font("arial", "b", size)
            if self.get_string_width(self.exp_name) <= WORKING_WIDTH:
                self.header_size = size
                break
        self.well_amount = well_amount
        self.set_margins(0, 0, 0)

    def new_page(self, header):
        self.add_page()
        self.set_font("arial", "b", self.header_size)
        self.cell(WORKING_WIDTH, h=10, txt=self.exp_name, ln=1, align="C")
        for size in range(self.header_size - 2, 1, -1):
            self.set_font("arial", "", size)
            if self.get_string_width(header) <= WORKING_WIDTH:
                self.cell(WORKING_WIDTH, h=10, txt=header, ln=1, align="C")
                break

    def write_table_to_pdf(self, columns):
        y_skip = (WORKING_HEIGHT - (10 * len(columns))) / 2
        self.ln(h=y_skip)
        longest_row_list = [""] * len(columns)
        for i in range(len(columns[0])):
            for j in range(len(columns)):
                if columns[j][i] > longest_row_list[j]:
                    longest_row_list[j] = columns[j][i]
        longest_row = "".join(longest_row_list)
        ratios = [len(longest_row_list[j]) / len(longest_row) for j in range(len(longest_row_list))]
        for size in range(self.header_size - 4, 1, -1):
            self.set_font("arial", "", size)
            if self.get_string_width(
                    longest_row) < WORKING_WIDTH - 20:  # -20 is just to be sure it doesn't go out of the lines
                break
        for i in range(len(columns[0])):
            for j in range(len(columns)):
                self.cell(WORKING_WIDTH * ratios[j], txt=columns[j][i], h=10, border=1, align="C")
            self.ln(h=10)

    def make_run_info_page(self):
        self.new_page("Run information")
        columns = [["Date of run", "Time of run", "Computer name", "PyMulti version"],
                   [time.strftime("%Y-%m-%d"), time.strftime("%H:%M"), os.getenv("COMPUTERNAME"), "1.1"]]
        self.write_table_to_pdf(columns)

    def make_path_pages(self, path_list):
        self.new_page("File path list")
        longest_path = max(path_list, key=len)
        for size in range(self.header_size - 4, 1, -1):
            self.set_font("arial", "", size)
            if self.get_string_width(longest_path) < WORKING_WIDTH - 20:  # -20 is just to be sure it doesn't go out of the lines
                break
        for p in path_list:
            self.cell(WORKING_WIDTH, txt=p, h=10, align="C", ln=1)

    def make_shortened_name_page(self, well_names, shortened_well_names):
        self.new_page("Well information")
        initials = ["Initials"] + [well[:2] for well in well_names]
        exp_num = ["Exp num"] + [well[2:5] for well in well_names]
        date = ["Date"] + [well[5:11] for well in well_names]
        channel = ["Channel"] + [well[11:14] for well in well_names]
        plate_num = ["Plate num"] + [well[14] for well in well_names]
        locations = ["Location"] + [well[15:18] for well in well_names]
        cell_line = ["Cell line"] + [well[18:22] for well in well_names]
        treatments = ["Treatments"] + [well[22:-4].replace("NNN0", "") for well in well_names]
        exp_type = ["Exp type"] + [well[-4:] for well in well_names]
        columns = [initials, exp_num, date, channel, plate_num, locations, cell_line, treatments, exp_type,
                   ["Shortened Name"] + [shortened_well_names[well] for well in well_names]]
        self.write_table_to_pdf(columns)

    def make_parameter_page(self, graph_dir, parameters):
        self.new_page("Parameter averages over time")
        self.set_automatic_coords(len(parameters))
        for i in range(len(parameters)):
            parameter = parameters[i]
            graph_path = os.path.join(graph_dir, parameter.replace("/", "div") + ".jpg")
            self.image(graph_path, x=self.graph_x_coords[i], y=self.graph_y_coords[i], w=self.graph_width)

    def make_clustergram_page(self, graph_path):
        self.new_page("Cluster Analysis")
        self.image(graph_path, x=0, y=20, w=WORKING_WIDTH, h=WORKING_HEIGHT)

    def make_pca_pages(self, graph_dir):
        self.new_page("Parameter Boxplots")
        for i in range(2):
            self.image(os.path.join(graph_dir, PARAM_GRAPHS[i]), x=TWO_GRAPH_X[i], y=TWO_GRAPH_Y[i],
                       w=TWO_GRAPH_WIDTH, h=TWO_GRAPH_HEIGHT)
        self.new_page("Classical PCA")
        for i in range(2):
            self.image(os.path.join(graph_dir, PCA_GRAPHS[i]), x=TWO_GRAPH_X[i], y=TWO_GRAPH_Y[i],
                       w=TWO_GRAPH_WIDTH, h=TWO_GRAPH_HEIGHT)

    def make_clusters_pages(self, graph_dir, num_clusters):
        self.new_page("%d Default Clusters - Heatmap" % num_clusters)
        self.image(os.path.join(graph_dir, "default_heatmap.jpg"), x=0, y=20, w=WORKING_WIDTH, h=WORKING_HEIGHT)
        self.new_page("%d KMeans Clusters - Heatmap" % num_clusters)
        self.image(os.path.join(graph_dir, "kmeans_heatmap.jpg"), x=0, y=20, w=WORKING_WIDTH, h=WORKING_HEIGHT)
        self.new_page("Clustering info - %d clusters" % num_clusters)
        for i in range(4):
            self.image(os.path.join(graph_dir, CLUSTER_GRAPHS[i]), x=FOUR_GRAPH_X[i], y=FOUR_GRAPH_Y[i],
                       w=FOUR_GRAPH_WIDTH, h=FOUR_GRAPH_HEIGHT)

    def make_correlation_pages(self, graph_dir):
        for correlation_method in ["pearson", "spearman"]:
            for axis in ["treatments", "parameters"]:
                self.new_page("%s - %s correlation" % (axis, correlation_method))
                self.image(os.path.join(graph_dir, correlation_method + "_" + axis + ".jpg"),
                           x=TWO_GRAPH_X[0], y=TWO_GRAPH_Y[0], w=TWO_GRAPH_WIDTH, h=TWO_GRAPH_HEIGHT)
                self.image(os.path.join(graph_dir, correlation_method + "_" + axis + "_p.jpg"),
                           x=TWO_GRAPH_X[1], y=TWO_GRAPH_Y[1], w=TWO_GRAPH_WIDTH, h=TWO_GRAPH_HEIGHT)
                self.new_page("%s - %s correlation" % (axis, correlation_method))
                self.image(os.path.join(graph_dir, correlation_method + "_" + axis + "_abs" + ".jpg"),
                           x=TWO_GRAPH_X[0], y=TWO_GRAPH_Y[0], w=TWO_GRAPH_WIDTH, h=TWO_GRAPH_HEIGHT)
                self.image(os.path.join(graph_dir, correlation_method + "_" + axis + "_p.jpg"),
                           x=TWO_GRAPH_X[1], y=TWO_GRAPH_Y[1], w=TWO_GRAPH_WIDTH, h=TWO_GRAPH_HEIGHT)


################## Calculation related functions ###################


def unify_dataframe(path_list, full=True):
    final_path_list = []
    for path in path_list:
        if os.path.isfile(path):
            if (full and "FULL" in path) or (not full and "FULL" not in path):
                final_path_list.append(path)
        else:
            dir_paths = os.listdir(path)
            final_path_list += [os.path.join(path, f) for f in dir_paths if "summary_table_" in f and
                                (full and "FULL" in f) or (not full and "FULL" not in f)]
    unified_dataframe = pd.concat([pd.read_excel(f) for f in final_path_list], ignore_index=True)
    unified_dataframe.drop(columns=["Unnamed: 0", "Parent_OLD"], inplace=True, errors="ignore")
    return final_path_list, unified_dataframe


def shorten_names(unified_df):
    well_names = list(unified_df.Experiment.unique())
    shortened_well_names = {well: well[18:-4].replace("NNN0", "") for well in well_names}
    all_short_names = list(shortened_well_names.values())
    # Delete cell line if all are the same
    first_val = all_short_names[0][:4]
    if False not in [name[:4] == first_val for name in list(shortened_well_names.values())]:
        shortened_well_names = {well: well[22:-4].replace("NNN0", "") for well in well_names}
    # Include cell location if several have the same
    if True in [all_short_names.count(name) > 1 for name in all_short_names]:
        for well in well_names:
            shortened_well_names[well] = well[15:18] + shortened_well_names[well]
    well_names.sort(key=lambda x: shortened_well_names[x])
    return well_names, shortened_well_names


def create_avg_dfs(unified_df, shortened_well_names, include_columns):
    exp_list = unified_df.Experiment.unique()
    df = unified_df.set_index("Experiment")
    if include_columns:
        parameters = [col for col in df.columns if col in include_columns]
    else:
        parameters = [col for col in df.columns if col not in IRRELEVANT_FIELDS]
    avges_per_exp = {}
    total_avg_df = pd.DataFrame(np.nan, dtype=float, index=exp_list, columns=parameters)
    for exp in exp_list:
        exp_df = df.loc[exp].set_index("TimeIndex")
        exp_df["Vx / Vy"] = exp_df.Velocity_X / exp_df.Velocity_Y
        time_indexes = exp_df.index.unique()
        avg_per_time_df = pd.DataFrame(np.nan, dtype=float, index=time_indexes, columns=parameters + ["Vx / Vy"])
        for col in parameters + ["Vx / Vy"]:
            # Make total average
            vals = exp_df[col]
            vals.dropna(inplace=True)
            vals = np.array(vals, dtype=float)
            if col in CLUSTER_WAVE_FIELDS:
                vals = vals[vals.nonzero()]
            total_avg_df.loc[exp][col] = np.average(vals)
            # Make per time point average
            if col in TIME_FIELDS:
                for time_index in time_indexes:
                    try:
                        exp_time_df = exp_df.loc[time_index]
                        vals = exp_time_df[col].copy()
                        vals.dropna(inplace=True)
                        vals = np.array(vals, dtype=float)
                        if col in CLUSTER_WAVE_FIELDS:
                            vals = vals[vals.nonzero()]
                        avg_per_time_df.loc[time_index][col] = np.average(vals)
                    except KeyError:
                        pass
        avges_per_exp[exp] = avg_per_time_df
    scaler = StandardScaler(with_std=True)
    scaled_avg_df = pd.DataFrame(scaler.fit_transform(total_avg_df), columns=total_avg_df.columns,
                                 index=[shortened_well_names[w] for w in total_avg_df.index])
    return avges_per_exp, total_avg_df, scaled_avg_df, parameters


def get_pca_info(scaled_avg_df):
    parameters = scaled_avg_df.columns
    pca = decomposition.PCA(random_state=42)
    pca.fit(scaled_avg_df)
    pca_transformed = pca.transform(scaled_avg_df)
    variance_explained = pca.explained_variance_ratio_
    component_num = len(variance_explained)
    variance_info = pd.DataFrame(
        [[variance_explained[i], sum(variance_explained[:i + 1])] for i in range(component_num)],
        index=range(1, component_num + 1), columns=["Variance", "Cumulative_Variance"])
    pc_labels = ["PC%d - %d percent variance explained" % (i, variance_info["Variance"][i] * 100) for i in [1, 2]]
    #pca_info = pd.DataFrame(pca.components_, index=range(1, component_num + 1), columns=parameters)
    return variance_info, pca, pca_transformed, pc_labels


def get_cluster_info(pca_transformed, nums_of_clusters):
    default_clusters_dict = kmeans_clusters_dict = {}
    for num in nums_of_clusters:
        kmeans_clusters_dict[num] = KMeans(n_clusters=num, random_state=0).fit(pca_transformed)
        default_clusters_dict[num] = AgglomerativeClustering(n_clusters=num).fit(pca_transformed)
    return default_clusters_dict, kmeans_clusters_dict


def get_correlation_info(scaled_avg_df):
    pearson_treatment_df = pd.DataFrame(tuple, columns=scaled_avg_df.index, index=scaled_avg_df.index)
    spearman_treatment_df = pearson_treatment_df.copy()
    for i in range(len(scaled_avg_df.index)):
        for j in range(i, len(scaled_avg_df.index)):
            treat1 = scaled_avg_df.index[i]
            treat2 = scaled_avg_df.index[j]
            pearson_val = pearsonr(scaled_avg_df.loc[treat1], scaled_avg_df.loc[treat2])
            spearman_val = spearmanr(scaled_avg_df.loc[treat1], scaled_avg_df.loc[treat2])
            pearson_treatment_df[treat1][treat2] = pearson_val
            pearson_treatment_df[treat2][treat1] = pearson_val
            spearman_treatment_df[treat1][treat2] = spearman_val
            spearman_treatment_df[treat2][treat1] = spearman_val
    pearson_parameter_df = pd.DataFrame(tuple, columns=scaled_avg_df.columns, index=scaled_avg_df.columns)
    spearman_parameter_df = pearson_parameter_df.copy()
    for i in range(len(scaled_avg_df.columns)):
        for j in range(i, len(scaled_avg_df.columns)):
            param1 = scaled_avg_df.columns[i]
            param2 = scaled_avg_df.columns[j]
            pearson_val = pearsonr(scaled_avg_df[param1], scaled_avg_df[param2])
            spearman_val = spearmanr(scaled_avg_df[param1], scaled_avg_df[param2])
            pearson_parameter_df[param1][param2] = pearson_val
            pearson_parameter_df[param2][param1] = pearson_val
            spearman_parameter_df[param1][param2] = spearman_val
            spearman_parameter_df[param2][param1] = spearman_val
    return [pearson_treatment_df, pearson_parameter_df, spearman_treatment_df, spearman_parameter_df]


################## Graph related functions ###################


def draw_parameter_graph(parameter, exp_list, avges_per_exp, time_indexes, dt, alt_names=None, graph_num="",
                         output_path=None):
    # Graph setup
    fig, graph_ax = plt.subplots()
    graph_ax.set_title(graph_num + parameter)
    graph_ax.set_xlabel(r"Time [min]")
    graph_ax.set_xbound(lower=min(time_indexes) * dt, upper=max(time_indexes) * dt)
    graph_ax.set_ylabel("Experiment")
    graph_ax.set_yticks([0.5 + i for i in range(len(exp_list))])
    graph_ax.tick_params(axis='both', which='major', labelsize=8)
    graph_ax.tick_params(axis='both', which='minor', labelsize=6)
    if alt_names:
        graph_ax.set_yticklabels([alt_names[exp] for exp in exp_list])
    else:
        graph_ax.set_yticklabels(exp_list)
    # Get min max and color setup
    min_val = np.nanmin([np.nanmin(df[parameter]) for df in avges_per_exp.values()])
    max_val = np.nanmax([np.nanmax(df[parameter]) for df in avges_per_exp.values()])
    normalized = colors.Normalize(vmin=min_val, vmax=max_val)
    # Set rectangles
    for i in range(len(exp_list)):
        exp = exp_list[i]
        for time_index in time_indexes:
            try:
                avg_value = avges_per_exp[exp].loc[time_index][parameter]
                normalized_value = normalized(avg_value)
                rect = patches.Rectangle((time_index * dt, i), width=dt, height=1, color=cm.jet(normalized_value))
                graph_ax.add_patch(rect)
            except KeyError:
                pass
    # Colorbar
    colorbar = fig.colorbar(cm.ScalarMappable(norm=normalized, cmap="jet"), ax=graph_ax)
    colorbar.set_label(parameter + UNIT_DICT[parameter])
    # Save or show plot
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


def create_parameter_page(pdf_report, avges_per_exp, dt, time_indexes, well_names, shortened_well_names, parameters,
                          output_path):
    page = "1"
    graph_dir = os.path.join(output_path, "parameter_avges")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    parameters = [param for param in parameters if param in TIME_FIELDS]
    parameters.append("Vx / Vy")
    for i in range(len(parameters)):
        parameter = parameters[i]
        if i <= 25:
            graph_num = page + chr(97 + i) + " "
        else:
            graph_num = page + chr(97 + (i // 26)) + chr(97 + (i % 26)) + " "
        graph_path = os.path.join(graph_dir, parameter.replace("/", "div") + ".jpg")
        if not os.path.exists(graph_path):
            draw_parameter_graph(parameter, well_names, avges_per_exp, time_indexes, dt, alt_names=shortened_well_names,
                                 graph_num=graph_num, output_path=graph_path)
    pdf_report.make_parameter_page(graph_dir, parameters)


def draw_cluster_analysis(scaled_avg_df, output_path=None):
    linkaged_pca = linkage(scaled_avg_df, "ward")
    s = sns.clustermap(data=scaled_avg_df, row_linkage=linkaged_pca, cmap=sns.color_palette("coolwarm", n_colors=256),
                       vmin=-2, vmax=2, figsize=(30, 15),
                       cbar_kws=dict(use_gridspec=False))
    # s.ax_heatmap.set_xlabel("Parameters", fontsize=25, fontweight='bold')
    # s.ax_heatmap.set_ylabel("Well", fontsize=25, fontweight='bold')
    # s.cax.set_yticklabels(s.cax.get_yticklabels());
    # pos = s.ax_heatmap.get_position();
    # cbar = s.cax
    # cbar.set_position([0.02, pos.bounds[1], 0.02, pos.bounds[3]]);
    s.ax_heatmap.set_xticklabels(s.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.suptitle("2a - Cluster analysis", fontweight='bold', fontsize=12)
    # Save or show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def create_clustermap_page(pdf_report, scaled_avg_df, output_path):
    graph_dir = os.path.join(output_path, "cluster_analysis")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    graph_path = os.path.join(graph_dir, "clustermap.jpg")
    if not os.path.exists(graph_path):
        draw_cluster_analysis(scaled_avg_df, output_path=graph_path)
    pdf_report.make_clustergram_page(graph_path)


def draw_means_graph(graph_df, data_type, output_path=None):
    # Box plots
    relevant_columns = [col for col in graph_df.columns if col not in IRRELEVANT_FIELDS]
    graph_df = graph_df.drop(columns=IRRELEVANT_FIELDS, errors="ignore")
    if data_type == "mean":
        graph_df = abs(graph_df)
    graph_ax = graph_df.plot(kind="box", showmeans=True, meanline=True)
    # Graph setup
    if output_path:
        title = " for each parameter"
        if data_type == "mean":
            title = "3a - Mean Absolute Value" + title
        elif data_type == "zscore":
            title = "3b - Zscore" + title
        graph_ax.set_title(title)
    graph_ax.set_xticklabels(relevant_columns, rotation=90)
    if data_type == "mean":
        graph_ax.set_ylabel("Value - log scale")
        graph_ax.set_yscale("log")
    elif data_type == "zscore":
        graph_ax.set_ylabel("Value")
    # Save or show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def draw_variance_graph(variance_info, output_path=None):
    # Graph setup
    fig, graph_ax = plt.subplots()
    title = "Variance explained by PCA components"
    if output_path:
        title = "4a - " + title
    graph_ax.set_title(title)
    graph_ax.set_xlabel("Principal Component")
    graph_ax.set_ylabel(r"Variance explained")
    graph_ax.set_ybound(lower=0, upper=100)
    graph_ax.set_yticks([0] + [i / 10 for i in range(1, 11)])
    graph_ax.set_yticklabels([str(i) + r"%" for i in range(0, 110, 10)])
    graph_ax.set_xticks(variance_info.index)
    graph_ax.set_xticklabels(variance_info.index, rotation=90)
    # Plotting
    graph_ax.bar(variance_info.index, variance_info.Variance)
    graph_ax.plot(variance_info.Cumulative_Variance)
    # Save or show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def draw_biplot(pca, scaled_avg_df, output_path=None):
    # Graph setup
    fig, graph_ax = plt.subplots()
    title = "PCA Biplot"
    if output_path:
        title = "4b - " + title
    graph_ax.set_title(title)
    graph_ax.set_xlabel("Component 1")
    graph_ax.set_ylabel("Component 2")
    # Plotting
    xvector = pca.components_[0]
    yvector = pca.components_[1]
    xs = pca.transform(scaled_avg_df)[:, 0]
    ys = pca.transform(scaled_avg_df)[:, 1]
    for i in range(len(xvector)):
        # arrows project features (ie columns from df) as vectors onto PC axes
        graph_ax.arrow(0, 0, xvector[i] * max(xs), yvector[i] * max(ys), color='r', width=0.0005, head_width=0.0025)
        graph_ax.text(xvector[i] * max(xs) * 1.2, yvector[i] * max(ys) * 1.2, list(scaled_avg_df.columns.values)[i],
                      color='r')
    #for i in range(len(xs)):
    #    # circles project documents (ie rows from df) as points onto PC axes
    #    graph_ax.plot(xs[i], ys[i], 'bo')
    #    graph_ax.text(xs[i] * 1.2, ys[i] * 1.2, list(scaled_avg_df.index)[i], color='b')
    # Save or show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def create_classical_pca_page(pdf_report, unified_df, variance_info, pca, scaled_avg_df, output_path):
    graph_dir = os.path.join(output_path, "classical PCA")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    graph_path = os.path.join(graph_dir, "log_means.jpg")
    if not os.path.exists(graph_path):
        draw_means_graph(unified_df, "mean", output_path=graph_path)
    graph_path = os.path.join(graph_dir, "zscore.jpg")
    if not os.path.exists(graph_path):
        draw_means_graph(scaled_avg_df, "zscore", output_path=graph_path)
    graph_path = os.path.join(graph_dir, "variance_explained.jpg")
    if not os.path.exists(graph_path):
        draw_variance_graph(variance_info, output_path=graph_path)
    graph_path = os.path.join(graph_dir, "pca_biplot.jpg")
    if not os.path.exists(graph_path):
        draw_biplot(pca, scaled_avg_df, output_path=graph_path)
    pdf_report.make_pca_pages(graph_dir)


def draw_cluster_heatmap(graph_df, num, cluster_mode, page_add=0, output_path=None):
    # Graph setup
    fig, graph_ax = plt.subplots()
    title = "Treatment - Parameter MEANS"
    if output_path:
        page_num = 5 + (3 * page_add)
        if cluster_mode == "kmeans":
            page_num += 1
        title = str(page_num) + " - " + title
    graph_ax.set_title(title)
    groups = [sum(graph_df.Clusters <= i) for i in range(num)]
    graph_df = graph_df.sort_values("Clusters")
    graph_df.drop(columns="Clusters", inplace=True)
    graph_df = graph_df.transpose()
    # Plotting
    graph_ax = sns.heatmap(graph_df, cmap=sns.color_palette("coolwarm", n_colors=256), vmin=-2, vmax=2, cbar=False,
                           ax=graph_ax)
    graph_ax.set_xlabel("Treatments")
    graph_ax.set_ylabel("Parameters")
    graph_ax.set_yticks([0.5 + i for i in range(len(graph_df.index))])
    graph_ax.set_yticklabels(graph_df.index, fontdict={'fontsize': 8})
    graph_ax.set_xticks([0.5 + i for i in range(len(graph_df.columns))])
    graph_ax.set_xticklabels(graph_df.columns, rotation=90, horizontalalignment='right', fontdict={'fontsize': 8})
    graph_ax.vlines(groups[:-1], *graph_ax.get_ylim())
    groups = [0] + groups
    graph_ax2 = graph_ax.twiny()
    graph_ax2.set_xbound(*graph_ax.get_xbound())
    graph_ax2.set_xlabel("Groups")
    graph_ax2.set_xticks([sum(groups[i:i+2]) / 2 for i in range(num)])
    graph_ax2.set_xticklabels([str(i) for i in range(1, num+1)])
    # Save or show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def draw_silhouette_graph(graph_df, num, cluster_mode, page_add=0, output_path=None):
    # Graph setup
    groups = [0] + [sum(graph_df.Clusters <= i) for i in range(num)]
    graph_df = graph_df.sort_values("Clusters")
    clusters = graph_df.Clusters
    graph_df.drop(columns="Clusters", inplace=True)
    fig, graph_ax = plt.subplots()
    title = "Silhouette - " + cluster_mode
    if output_path:
        if cluster_mode == "default":
            title = "a - " + title
        elif cluster_mode == "kmeans":
            title = "b - " + title
        title = str(7 + (3 * page_add)) + title
    graph_ax.set_title(title)
    graph_ax.set_xlabel("Silhouette Value")
    graph_ax.set_xlim(-1, 1)
    graph_ax.set_ylabel("Cluster")
    graph_ax.set_ylim(-0.5, groups[-1] + 1.5)
    graph_ax.set_yticks([sum(groups[i:i+2]) / 2 for i in range(num)])
    graph_ax.set_yticklabels([str(i) for i in range(1, num+1)])
    # Plotting
    silhouette_avg = silhouette_score(graph_df, clusters)
    sample_silhouette_values = silhouette_samples(graph_df, clusters)
    for i in range(num):
        # Aggregate the silhouette scores for samples belonging to
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        graph_ax.barh(range(groups[i], groups[i+1]), ith_cluster_silhouette_values, color=LINE_COLORS[i])
    graph_ax.vlines(silhouette_avg, *graph_ax.get_ybound(), linestyles="dashed")
    # Save or show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def draw_pca_scatter(pca_df, pc_labels, cluster_mode="", page_add=0, output_path=None):
    # Graph setup
    fig, graph_ax = plt.subplots()
    title = "PCA Scatter Plot - " + cluster_mode
    if output_path:
        if cluster_mode == "default":
            title = "c - " + title
        elif cluster_mode == "kmeans":
            title = "d - " + title
        title = str(7 + (3 * page_add)) + title
    graph_ax.set_title(title)
    graph_ax.set_xlabel(pc_labels[0])
    graph_ax.set_ylabel(pc_labels[1])
    # Plotting
    graph_ax.scatter(pca_df[1], pca_df[2], c=[LINE_COLORS[i] for i in pca_df.Clusters])
    # Save or show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def create_cluster_pages(pdf_report, i, num, kmeans_clusters, default_clusters, scaled_avg_df, pca_df, pc_labels,
                         output_path):
    graph_dir = os.path.join(output_path, "%d_clusters" % num)
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    scaled_avg_df = scaled_avg_df.copy()
    pca_df = pca_df.copy()
    scaled_avg_df["Clusters"] = kmeans_clusters.labels_
    pca_df["Clusters"] = kmeans_clusters.labels_
    graph_path = os.path.join(graph_dir, "kmeans_heatmap.jpg")
    if not os.path.exists(graph_path):
        draw_cluster_heatmap(scaled_avg_df, num, "kmeans", page_add=i, output_path=graph_path)
    graph_path = os.path.join(graph_dir, "kmeans_silhouette.jpg")
    if not os.path.exists(graph_path):
        draw_silhouette_graph(pca_df, num, "kmeans", page_add=i, output_path=graph_path)
    graph_path = os.path.join(graph_dir, "pca_scatter_kmeans.jpg")
    if not os.path.exists(graph_path):
        draw_pca_scatter(pca_df, pc_labels, cluster_mode="kmeans", page_add=i, output_path=graph_path)
    scaled_avg_df["Clusters"] = default_clusters.labels_
    pca_df["Clusters"] = default_clusters.labels_
    graph_path = os.path.join(graph_dir, "default_heatmap.jpg")
    if not os.path.exists(graph_path):
        draw_cluster_heatmap(scaled_avg_df, num, "default", page_add=i, output_path=graph_path)
    graph_path = os.path.join(graph_dir, "default_silhouette.jpg")
    if not os.path.exists(graph_path):
        draw_silhouette_graph(scaled_avg_df, num, "default", page_add=i, output_path=graph_path)
    graph_path = os.path.join(graph_dir, "pca_scatter_default.jpg")
    if not os.path.exists(graph_path):
        draw_pca_scatter(pca_df, pc_labels, cluster_mode="default", page_add=i, output_path=graph_path)
    pdf_report.make_clusters_pages(graph_dir, num)


def draw_correlation_graph(correlation_df, axis, correlation_method, value, page_num=0, absolute=False,
                           output_path=None):
    # Graph setup
    title = "%s - %s correlation - %s value" % (axis, correlation_method, value)
    if absolute:
        title += " absolute"
    if output_path:
        if value == "r":
            if absolute:
                title = str(page_num) + "b - " + title
            else:
                title = str(page_num) + "a - " + title
        elif value == "p":
            title = str(page_num) + "c - " + title
    # Plotting
    if value == "r":
        graph_df = correlation_df.applymap(lambda x: x[0])
        if absolute:
            graph_df = graph_df.abs()
            vmin = 0
            cmap = sns.color_palette("Blues", n_colors=256)
            cmap.reverse()
        else:
            vmin = -1
            cmap = "BrBG"
    elif value == "p":
        graph_df = correlation_df.applymap(lambda x: x[1])
        vmin = 0
        cmap="Blues"
    heatmap = sns.heatmap(graph_df, vmin=vmin, vmax=1, cmap=cmap)
    heatmap.set_title(title)
    # Save or show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def create_correlation_pages(pdf_report, correlation_info, page_num, output_path):
    pearson_treatment_df, pearson_parameter_df, spearman_treatment_df, spearman_parameter_df = correlation_info
    graph_dir = os.path.join(output_path, "correlation_graphs")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    # Pearson correlations
    graph_path = os.path.join(graph_dir, "pearson_treatments.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(pearson_treatment_df, "treatments", "pearson", "r", page_num=page_num,
                               output_path=graph_path)
    graph_path = os.path.join(graph_dir, "pearson_treatments_abs.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(pearson_treatment_df, "treatments", "pearson", "r", page_num=page_num, absolute=True,
                               output_path=graph_path)
    graph_path = os.path.join(graph_dir, "pearson_treatments_p.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(pearson_treatment_df, "treatments", "pearson", "p", page_num=page_num,
                               output_path=graph_path)
    page_num += 1
    graph_path = os.path.join(graph_dir, "pearson_parameters.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(pearson_parameter_df, "parameters", "pearson", "r", page_num=page_num,
                               output_path=graph_path)
    graph_path = os.path.join(graph_dir, "pearson_parameters_abs.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(pearson_parameter_df, "parameters", "pearson", "r", page_num=page_num, absolute=True,
                               output_path=graph_path)
    graph_path = os.path.join(graph_dir, "pearson_parameters_p.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(pearson_parameter_df, "parameters", "pearson", "p", page_num=page_num,
                               output_path=graph_path)
    page_num += 1
    # Spearman correlation
    graph_path = os.path.join(graph_dir, "spearman_treatments.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(spearman_treatment_df, "treatments", "spearman", "r", page_num=page_num,
                               output_path=graph_path)
    graph_path = os.path.join(graph_dir, "spearman_treatments_abs.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(spearman_treatment_df, "treatments", "spearman", "r", page_num=page_num, absolute=True,
                               output_path=graph_path)
    graph_path = os.path.join(graph_dir, "spearman_treatments_p.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(spearman_treatment_df, "treatments", "spearman", "p", page_num=page_num,
                               output_path=graph_path)
    page_num += 1
    graph_path = os.path.join(graph_dir, "spearman_parameters.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(spearman_parameter_df, "parameters", "spearman", "r", page_num=page_num,
                               output_path=graph_path)
    graph_path = os.path.join(graph_dir, "spearman_parameters_abs.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(spearman_parameter_df, "parameters", "spearman", "r", page_num=page_num, absolute=True,
                               output_path=graph_path)
    graph_path = os.path.join(graph_dir, "spearman_parameters_p.jpg")
    if not os.path.exists(graph_path):
        draw_correlation_graph(spearman_parameter_df, "parameters", "spearman", "p", page_num=page_num,
                               output_path=graph_path)
    pdf_report.make_correlation_pages(graph_dir)


################## Main function ###################


def multi_run(data_paths, output_path, exp_name, nums_of_clusters, include_columns, full=True):
    # Preliminary data organization - should take some time to process
    print("Loading data...")
    if type(data_paths) == list:
        path_list, unified_df = unify_dataframe(data_paths, full=full)
    elif type(data_paths) == tuple:  # Weird fix just to make it easier for me to debug - no need to reload each time
        path_list, unified_df = data_paths
    dt = unified_df.dt.unique()
    if len(dt) > 1:
        print("Bad dts:", dt)
        raise ValueError("Your experiments had more than one dt, can't compare")
    else:
        dt = int(dt)
    time_indexes = unified_df.TimeIndex.unique()
    # Initial calculations
    print("Running initial calculations - normalization, clustering etc...")
    well_names, shortened_well_names = shorten_names(unified_df)
    well_amount = len(well_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avges_per_exp, total_avg_df, scaled_avg_df, parameters = create_avg_dfs(unified_df, shortened_well_names,
                                                                                include_columns)
    variance_info, pca, pca_transformed, pc_labels = get_pca_info(scaled_avg_df)
    pca_df = pd.DataFrame(pca_transformed, index=scaled_avg_df.index, columns=range(1, pca.n_components_ + 1))
    default_clusters_dict, kmeans_clusters_dict = get_cluster_info(pca_transformed, nums_of_clusters)
    correlation_info = get_correlation_info(scaled_avg_df)
    print("Done with initial stages, creating PDF report")
    # Create PDF and make first pages
    pdf_report = PDF(orientation="L")
    pdf_report.set_params(exp_name, well_amount)
    pdf_report.make_run_info_page()
    pdf_report.make_path_pages(path_list)
    pdf_report.make_shortened_name_page(well_names, shortened_well_names)
    print("Done with first pages, creating graphs for the report")
    # Data pages
    create_parameter_page(pdf_report, avges_per_exp, dt, time_indexes, well_names, shortened_well_names, parameters,
                          output_path)
    create_clustermap_page(pdf_report, scaled_avg_df, output_path)
    create_classical_pca_page(pdf_report, unified_df, variance_info, pca, scaled_avg_df, output_path)
    for i in range(len(nums_of_clusters)):
        num = nums_of_clusters[i]
        create_cluster_pages(pdf_report, i, num, kmeans_clusters_dict[num], default_clusters_dict[num], scaled_avg_df,
                             pca_df, pc_labels, output_path)
    page_num = 5 + (3 * len(nums_of_clusters))
    create_correlation_pages(pdf_report, correlation_info, page_num, output_path)
    # Save PDF
    pdf_report.output(os.path.join(output_path, exp_name + "_multi_report.pdf"))
    return "Done"


##################### GUI #########################


def get_input_paths(input_paths, i):
    input_valid = True
    input_paths[i] = st.text_input("Input path #%d:" % (i + 1), key="input%d" % i,
                                   value=input_paths[i]).replace("\"", "")
    if input_paths[i] != "":
        if not os.path.exists(input_paths[i]):
            st.error("Path doesn't exist!")
            input_valid = False
        elif os.path.splitext(input_paths[i])[-1] not in ["", ".xls", ".xlsx", ".xlsm"]:
            st.error("Path must be a folder or xls file (summary table).")
            input_valid = False
    if input_paths[-1] != "" and input_valid:
        input_paths.append("")
        input_paths, new_valid = get_input_paths(input_paths, i + 1)
        input_valid = new_valid
    return input_paths, input_valid


def get_cluster_numbers(nums_of_clusters, i):
    nums_of_clusters[i] = st.number_input("Enter a number of clusters:", value=nums_of_clusters[i],
                                          key="nums%d" % i)
    if nums_of_clusters[-1] != 0:
        nums_of_clusters.append(0)
        nums_of_clusters = get_cluster_numbers(nums_of_clusters, i + 1)
    return nums_of_clusters

@st.cache
def get_column_options(path_list, full):
    sample_path_list = []
    for path in path_list[:-1]:
        if os.path.isfile(path):
            if (full and "FULL" in path) or (not full and "FULL" not in path):
                sample_path_list.append(path)
        else:
            dir_paths = os.listdir(path)
            sample_path_list.append([os.path.join(path, f) for f in dir_paths if "summary_table_" in f and
                                     (full and "FULL" in f) or (not full and "FULL" not in f)][0])
    column_options = pd.concat([pd.read_excel(path, nrows=0) for path in sample_path_list])
    column_options.drop(columns=IRRELEVANT_FIELDS, inplace=True, errors="ignore")
    return column_options


def multi_gui():
    st.header("Multi Batch Analysis")
    st.subheader("Input paths")
    st.write("Enter paths to be analyzed - summary tables (.xlsx) or folders.")
    input_paths = [""]
    input_paths, input_valid = get_input_paths(input_paths, 0)
    full = st.checkbox("Run on full versions of summary tables?", value=True)
    st.subheader("Output path")
    st.write("Enter output path (create the folder and make sure it's empty.)")
    output_valid = True
    output_path = st.text_input("Output path:", value="").replace("\"", "")
    if output_path == "":
        output_valid = False
    elif not os.path.exists(output_path):
        st.error("Path doesn't exist!")
        output_valid = False
    elif os.listdir(output_path):
        st.error("Folder is not empty!")
        output_valid = False
    st.subheader("Cluster options")
    st.write("Select the numbers of clusters you want to use for cluster analysis")
    clusters_valid = False
    nums_of_clusters = [0]
    nums_of_clusters = get_cluster_numbers(nums_of_clusters, 0)
    nums_of_clusters = list(set(nums_of_clusters))
    nums_of_clusters.sort(reverse=True)
    if len(nums_of_clusters) > 1:
        clusters_valid = True
    st.subheader("Columns to include")
    st.write("This part may take a minute or two to load.")
    if st.checkbox("Click here to select which parameters to include", value=False):
        column_options = get_column_options(input_paths, full)
        with st.form(key="parameter selection"):
            include_columns = [col for col in column_options if st.checkbox("Include %s" % col, value=True)]
            st.form_submit_button("Done selecting parameters")
    else:
        include_columns = None
    st.subheader("Experiment Name")
    exp_name = st.text_input("Enter your experiment name:")
    if input_valid and clusters_valid and output_valid and exp_name:
        if st.button("Run now!"):
            with st.spinner("Running multi analysis (check cmd for progress information)"):
                done = multi_run(input_paths[:-1], output_path, exp_name, nums_of_clusters[:-1],
                                 include_columns=include_columns, full=full)
            if done == "Done":
                st.balloons()
                st.success("Done running multi analysis!")


multi_gui()

