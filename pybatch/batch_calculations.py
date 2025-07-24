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

WORKING_WIDTH = 297
WORKING_HEIGHT = 180
SINGLE_GRAPH_X = 58.5
SINGLE_GRAPH_Y = 30
SINGLE_GRAPH_WIDTH = 180
TWO_GRAPH_X = [0.0, 148.5]
TWO_GRAPH_Y = [31.5, 31.5]
TWO_GRAPH_WIDTH = 148.5
EIGHT_GRAPH_X = [0.0, 74.25, 148.5, 222.75, 0.0, 74.25, 148.5, 222.75]
EIGHT_GRAPH_Y = [30.0, 30.0, 30.0, 30.0, 120.0, 120.0, 120.0, 120.0]
EIGHT_GRAPH_WIDTH = 74.25
FIVE_GRAPH_WIDTH = 90
FIVE_GRAPH_X = [0.0, 103.5, 207.0, 29.25, 177.75]
FIVE_GRAPH_Y = [30.0, 30.0, 30.0, 120.0, 120.0]
FOUR_GRAPH_WIDTH = 90
FOUR_GRAPH_X = [0.0, 207.0, 0.0, 207.0]
FOUR_GRAPH_Y = [30.0, 30.0, 120.0, 120.0]
SIX_GRAPH_X = [10.0, 113.5, 217.0, 10.0, 113.5, 217.0]
SIX_GRAPH_Y = [50, 50, 50, 125, 125, 125]
TEN_GRAPH_WIDTH = 60
TEN_GRAPH_X = [0.0, 79.0, 158.0, 237.0, 0.0, 79.0, 158.0, 237.0, 44.25, 192.75]
TEN_GRAPH_Y = [30.0, 30.0, 30.0, 30.0, 90.0, 90.0, 90.0, 90.0, 150.0, 150.0]
EIGHTEEN_GRAPH_WIDTH = 49.5
EIGHTEEN_GRAPH_X = [0.0, 49.5, 99.0, 148.5, 198.0, 247.5, 0.0, 49.5, 99.0, 148.5, 198.0, 247.5, 0.0, 49.5, 99.0, 148.5,
                    198.0, 247.5]
EIGHTEEN_GRAPH_Y = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 150.0, 150.0, 150.0, 150.0,
                    150.0, 150.0]
THIRTYTWO_GRAPH_WIDTH = 37.125
THIRTYTWO_GRAPH_X = [0.0, 37.125, 74.25, 111.375, 148.5, 185.625, 222.75, 259.875, 0.0, 37.125, 74.25, 111.375, 148.5,
                     185.625, 222.75, 259.875, 0.0, 37.125, 74.25, 111.375, 148.5, 185.625, 222.75, 259.875, 0.0,
                     37.125, 74.25, 111.375, 148.5, 185.625, 222.75, 259.875]
THIRTYTWO_GRAPH_Y = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0,
                     120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 165.0, 165.0, 165.0, 165.0, 165.0, 165.0,
                     165.0, 165.0]
FOURTY_GRAPH_WIDTH = 36.0
FOURTY_GRAPH_X = [0.0, 37.285714285714285, 74.57142857142857, 111.85714285714286, 149.14285714285714,
                  186.42857142857142, 223.71428571428572, 261.0, 0.0, 37.285714285714285, 74.57142857142857,
                  111.85714285714286, 149.14285714285714, 186.42857142857142, 223.71428571428572, 261.0, 0.0,
                  37.285714285714285, 74.57142857142857, 111.85714285714286, 149.14285714285714, 186.42857142857142,
                  223.71428571428572, 261.0, 0.0, 37.285714285714285, 74.57142857142857, 111.85714285714286,
                  149.14285714285714, 186.42857142857142, 223.71428571428572, 261.0, 0.0, 37.285714285714285,
                  74.57142857142857, 111.85714285714286, 149.14285714285714, 186.42857142857142, 223.71428571428572,
                  261.0]
FOURTY_GRAPH_Y = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 102.0,
                  102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 138.0, 138.0, 138.0, 138.0, 138.0, 138.0, 138.0,
                  138.0, 174.0, 174.0, 174.0, 174.0, 174.0, 174.0, 174.0, 174.0]

MIN_SCRATCH_DIFF = 0.05

PARAM_GRAPHS= {"Instantaneous_Speed": ["y_pos_time", "average", "layers", "layers_scaled"],
               "Velocity_X": ["y_pos_time", "absolute_y_pos_time", "average", "absolute_average", "absolute_layers", "absolute_layers_scaled"],
               "Velocity_Y": ["y_pos_time", "absolute_y_pos_time", "average", "absolute_average", "absolute_layers", "absolute_layers_scaled"],
               "Velocity_Z": ["y_pos_time", "absolute_y_pos_time", "average", "absolute_average", "absolute_layers", "absolute_layers_scaled"],
               "Displacement_From_Last_Id": ["y_pos_time", "average", "layers", "layers_scaled"],
               "Coll": ["y_pos_time", "average", "layers", "layers_scaled"],
               "Coll_CUBE": ["y_pos_time", "average", "layers", "layers_scaled"],
               "Acceleration": ["y_pos_time", "average", "layers", "layers_scaled"],
               "Acceleration_X": ["y_pos_time", "absolute_y_pos_time", "average", "absolute_average", "absolute_layers", "absolute_layers_scaled"],
               "Acceleration_Y": ["y_pos_time", "absolute_y_pos_time", "average", "absolute_average", "absolute_layers", "absolute_layers_scaled"],
               "Acceleration_Z": ["y_pos_time", "absolute_y_pos_time", "average", "absolute_average", "absolute_layers", "absolute_layers_scaled"],
               "Displacement2": ["average"],
               "Directional_Change": ["average"],
               "Volume": ["average"],
               "Ellipticity_oblate": ["average"],
               "Ellipticity_prolate": ["average"],
               "Eccentricity": ["average"],
               "Sphericity": ["average"],
               "EllipsoidAxisLengthB": ["average"],
               "EllipsoidAxisLengthC": ["average"],
               "Ellip_Ax_B_X": ["average"],
               "Ellip_Ax_B_Y": ["average"],
               "Ellip_Ax_C_X": ["average"],
               "Ellip_Ax_C_Y": ["average"],
               "Instantaneous_Angle": ["average"],
               "IntensityCenterCh1": ["y_pos_time", "average"],
               "IntensityCenterCh2": ["y_pos_time", "average"],
               "IntensityCenterCh3": ["y_pos_time", "average"],
               "IntensityMaxCh1": ["y_pos_time", "average"],
               "IntensityMaxCh2": ["y_pos_time", "average"],
               "IntensityMaxCh3": ["y_pos_time", "average"],
               "IntensityMeanCh1": ["y_pos_time", "average"],
               "IntensityMeanCh2": ["y_pos_time", "average"],
               "IntensityMeanCh3": ["y_pos_time", "average"],
               "IntensityMedianCh1": ["y_pos_time", "average"],
               "IntensityMedianCh2": ["y_pos_time", "average"],
               "IntensityMedianCh3": ["y_pos_time", "average"],
               "IntensitySumCh1": ["y_pos_time", "average"],
               "IntensitySumCh2": ["y_pos_time", "average"],
               "IntensitySumCh3": ["y_pos_time", "average"],
               "Min_Distance": ["y_pos_time", "layers", "layers_scaled"]}

PARAM_PAIR_GRAPHS = {("Velocity_X", "Velocity_Y"): ["average", "absolute"],
                     ("Velocity_X", "Velocity_Z"): ["average", "absolute"],
                     ("Velocity_Z", "Velocity_Y"): ["average", "absolute"],
                     ("Acceleration_X", "Acceleration_Y"): ["average", "absolute"],
                     ("Acceleration_X", "Acceleration_Z"): ["average", "absolute"],
                     ("Acceleration_Z", "Acceleration_Y"): ["average", "absolute"],
                     ("Area", "Coll"): ["average"],
                     ("Area", "Sphericity"): ["average"],
                     ("Area", "Instantaneous_Speed"): ["average"],
                     ("Instantaneous_Speed", "Coll"): ["average"],
                     ("Displacement_From_Last_Id", "Coll"): ["average"],
                     ("Instantaneous_Speed", "Sphericity"): ["average"],
                     ("Instantaneous_Speed", "Min_Distance"): ["average"],
                     ("Acceleration", "Min_Distance"): ["average"],
                     ("IntensityCenterCh1", "IntensityCenterCh2"): ["average"],
                     ("IntensityCenterCh1", "IntensityCenterCh3"): ["average"],
                     ("IntensityCenterCh2", "IntensityCenterCh3"): ["average"],
                     ("IntensityMaxCh1", "IntensityMaxCh2"): ["average"],
                     ("IntensityMaxCh1", "IntensityMaxCh3"): ["average"],
                     ("IntensityMaxCh2", "IntensityMaxCh3"): ["average"],
                     ("IntensityMeanCh1", "IntensityMeanCh2"): ["average"],
                     ("IntensityMeanCh1", "IntensityMeanCh3"): ["average"],
                     ("IntensityMeanCh2", "IntensityMeanCh3"): ["average"],
                     ("IntensityMedianCh1", "IntensityMedianCh2"): ["average"],
                     ("IntensityMedianCh1", "IntensityMedianCh3"): ["average"],
                     ("IntensityMedianCh2", "IntensityMedianCh3"): ["average"],
                     ("IntensitySumCh1", "IntensitySumCh2"): ["average"],
                     ("IntensitySumCh1", "IntensitySumCh3"): ["average"],
                     ("IntensitySumCh2", "IntensitySumCh3"): ["average"]}

WAVE_PARAMETERS = ["Velocity_Full_Width_Half_Maximum", "Velocity_Time_of_Maximum_Height", "Velocity_Maximum_Height", "Velocity_Ending_Value",
                   "Velocity_Ending_Time", "Velocity_Starting_Value", "Velocity_Starting_Time"]
LEFT_WAVE_PARAMETERS = ["Velocity_Full_Width_Half_Maximum", "Velocity_Time_of_Maximum_Height", "Velocity_Maximum_Height", "Velocity_Starting_Value", "Velocity_Starting_Time"]
RIGHT_WAVE_PARAMETERS = ["Velocity_Full_Width_Half_Maximum", "Velocity_Time_of_Maximum_Height", "Velocity_Maximum_Height", "Velocity_Ending_Value", "Velocity_Ending_Time"]

DISPLACEMENT_PARAMS = ["Overall_Displacement", "Total_Track_Displacement", "Track_Displacement_X", "Track_Displacement_Y", "Track_Displacement_Z"]

MOTILITY_PARAMS = ["Linearity_of_Forward_Progression", "Mean_Curvilinear_Speed", "Mean_Straight_Line_Speed", "Confinement_Ratio"]

MSD_PARAMS = ["MSD_Linearity_R2_Score", "MSD_Brownian_Motion_BIC_Score", "MSD_Brownian_D", "MSD_Directed_Motion_BIC_Score", "MSD_Directed_D", "MSD_Directed_v2"]

UNIT_DICT = {"Instantaneous_Speed": r" [$\mu$m/h]", "Velocity_X": r" [$\mu$m/h]", "Velocity_Y": r" [$\mu$m/h]", "Velocity_Z": r" [$\mu$m/h]",
             "Acceleration": r" [$\mu$m/$h^2$]", "Acceleration_X": r" [$\mu$m/$h^2$]", "Acceleration_Y": r" [$\mu$m/$h^2$]", "Acceleration_Z": r" [$\mu$m/$h^2$]",
             "Coll": "", "Coll_CUBE": "", "Displacement2": r"[$\mu$$m^2$]", "Directional_Change": " [radians]", "Volume": r"[$\mum^3$]", "Ellipticity_oblate": "",
             "Ellipticity_prolate": "", "Eccentricity": "", "Sphericity": "", "EllipsoidAxisLengthB": r" [$\mu$m]", "EllipsoidAxisLengthC": r" [$\mu$m]",
             "Ellip_Ax_B_X": "", "Ellip_Ax_B_Y": "", "Ellip_Ax_C_X": "", "Ellip_Ax_C_Y": "", "Instantaneous_Angle": " [radians]",
             "Velocity_Full_Width_Half_Maximum": " [min]", "Velocity_Time_of_Maximum_Height": " [min]", "Velocity_Maximum_Height": r" [$\mu$m/h]",
             "Velocity_Ending_Value": r" [$\mu$m/h]", "Velocity_Ending_Time": " [min]", "Velocity_Starting_Value": r" [$\mu$m/h]", "Velocity_Starting_Time": " [min]",
             "Area": r" [$\mu$$m^2$]", "Overall_Displacement": r" [$\mu$m]", "Total_Track_Displacement": r" [$\mu$m]", "Track_Displacement_X": r" [$\mu$m]",
             "Track_Displacement_Y": r" [$\mu$m]", "Track_Displacement_Z": r" [$\mu$m]", "Linearity_of_Forward_Progression": "", "Confinement_Ratio": "",
             "Mean_Curvilinear_Speed": r" [$\mu$m/h]", "Mean_Straight_Line_Speed": r" [$\mu$m/h]", "MSD_Linearity_R2_Score": "", "MSD_Brownian_Motion_BIC_Score": "",
             "MSD_Brownian_D": "", "MSD_Directed_Motion_BIC_Score": "", "MSD_Directed_D": "", "MSD_Directed_v2": "", "Min_Distance": r" [$\mu$m]",
             "Displacement_From_Last_Id": r" [$\mu$m]", "IntensityCenterCh1": "", "IntensityCenterCh2": "","IntensityCenterCh3": "", "IntensityMaxCh1": "", "IntensityMaxCh2": "","IntensityMaxCh3": "", "IntensityMeanCh1": "",
             "IntensityMeanCh2": "","IntensityMeanCh3": "", "IntensityMedianCh1": "", "IntensityMedianCh2": "","IntensityMedianCh3": "", "IntensitySumCh1": "", "IntensitySumCh2": "","IntensitySumCh3": ""}

MARKERS = ["o", "^", "s", "P", "D", "X", "v", "<", ">", "p", ".", "1", "3", "4"] + ["$%s$" % chr(97+i) for i in range(26)]
LINE_COLORS = list(colors.TABLEAU_COLORS.keys()) + list(colors.XKCD_COLORS.keys())[:30]

CLUSTER_DROP_FIELDS = ["Parent", "Parent_OLD", "dt", "ID", "TimeIndex", "x_Pos", "y_Pos", "z_Pos", "Current_MSD_1", "Final_MSD_1",
                       "Instantaneous_Speed_OLD", "Instantaneous_Angle_OLD", "Directional_Change_OLD", "Experiment", "Acceleration_OLD"]
CLUSTER_ID_FIELDS = ["Area", "Displacement_From_Last_Id", "Instantaneous_Speed", "Velocity_X", "Velocity_Y", "Velocity_Z", "Coll", "Coll_CUBE", "Acceleration", "Acceleration_X",
                     "Acceleration_Y", "Acceleration_Z", "Displacement2", "Directional_Change", "Directional_Change_X", "Directional_Change_Y", "Directional_Change_Z", "Volume",
                     "Ellipticity_oblate", "Ellipticity_prolate", "Eccentricity", "Eccentricity_A", "Eccentricity_B", "Eccentricity_C", "Sphericity", "EllipsoidAxisLengthB",
                     "EllipsoidAxisLengthC", "Ellip_Ax_B_X", "Ellip_Ax_B_Y", "Ellip_Ax_B_Z", "Ellip_Ax_C_X", "Ellip_Ax_C_Y", "Ellip_Ax_C_Z", "Instantaneous_Angle",
                     "Instantaneous_Angle_X", "Instantaneous_Angle_Y", "Instantaneous_Angle_Z", "Min_Distance", "IntensityCenterCh1", "IntensityCenterCh2","IntensityCenterCh3", "IntensityMaxCh1",
                     "IntensityMaxCh2","IntensityMaxCh3", "IntensityMeanCh1", "IntensityMeanCh2","IntensityMeanCh3", "IntensityMedianCh1", "IntensityMedianCh2","IntensityMedianCh3", "IntensitySumCh1", "IntensitySumCh2","IntensitySumCh3"]
CLUSTER_CELL_FIELDS = ["Overall_Displacement", "Total_Track_Displacement", "Track_Displacement_X", "Track_Displacement_Y", "Track_Displacement_Z",
                       "Linearity_of_Forward_Progression", "Mean_Curvilinear_Speed", "Mean_Straight_Line_Speed", "Confinement_Ratio", "MSD_Directed_v2",
                       "MSD_Linearity_R2_Score", "MSD_Brownian_Motion_BIC_Score", "MSD_Brownian_D", "MSD_Directed_Motion_BIC_Score", "MSD_Directed_D",
                       "Velocity_Time_of_Maximum_Height", "Velocity_Maximum_Height"]
CLUSTER_WAVE_FIELDS = ["Velocity_Full_Width_Half_Maximum", "Velocity_Ending_Value", "Velocity_Ending_Time", "Velocity_Starting_Value",
                       "Velocity_Starting_Time"]

DICTS = ["min_max_dict", "middles_dict", "layers_min_max_dict", "scratch_dict", "scratch_min_max_dict", "msd_info"]

POSITIONS = ["x_Pos", "y_Pos", "z_Pos"]

average_df = None

class PDF(FPDF):
    pass
""" # return this if you want page numbers
    def footer(self):
        # Go to 1 cm from bottom
        self.set_y(-10)
        # Select Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Print centered page number
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')
"""

class Batch_Experiment(object):

    def __init__(self, exp_name, exp_sub_name, wells, scratch, protocol_file="", incucyte_files="", imaris_xls_files="",
                 dt=45, design="auto", table_info=None):
        self.exp_name = exp_name
        self.exp_sub_name = exp_sub_name
        self.wells = wells
        self.scratch = scratch
        self.well_amount = len(wells)
        self.well_info = {}
        self.protocol_file =protocol_file
        self.incucyte_files = incucyte_files
        self.imaris_xls_files = imaris_xls_files
        self.dt = dt
        self.min_max_dict = {}
        self.middles_dict = {}
        self.layers_min_max_dict = {}
        self.scratch_dict = {}
        self.scratch_min_max_dict = {}
        self.msd_info = {}
        self.graph_counter = 97
        self.design = design
        self.table_info = table_info

    def __has_attribute__(self, attribute):
        if attribute in self.__dict__.keys():
            return True
        else:
            return False

    def _min_max(self, attribute, min_val=None, max_val=None, multiplier=1, log=False, absolute=False, average=False, scaled=True):
        parameter = attribute
        if log: # no longer in use
            attribute += "_log"
        if absolute:
            attribute += "_abs"
        if average:
            attribute += "_avg"
        if scaled:
            attribute += "_scaled"
        if attribute in self.min_max_dict.keys():
            return self.min_max_dict[attribute]
        all_values = []
        for well_df in self.well_info.values():
            # --- Fix: skip wells missing the parameter ---
            if parameter not in well_df.columns:
                continue
            well_df = well_df.dropna(subset=[parameter])
            if average:
                time_indexes = np.array(well_df.TimeIndex.unique())
                time_indexes.sort()
                time_values = np.array([well_df[well_df.TimeIndex == time_index][parameter] for time_index in time_indexes])
                if log:
                    time_values = np.log(time_values)
                if absolute:
                    time_values = np.absolute(time_values)
                values = np.array([np.average(time_value) for time_value in time_values])
            else:
                values = np.array(well_df[parameter])
                if log:
                    values = np.log(values)
                if absolute:
                    values = np.absolute(values)
            all_values += list(values)
            if min_val:
                all_values.append(min_val)
            if max_val:
                all_values.append(max_val)
        all_values = np.array(all_values)
        min_val = min(all_values)
        max_val = max(all_values)
        if scaled:
            mean = all_values.mean()
            std = all_values.std()
            temp_min = mean - (2.5 * std)
            temp_max = mean + (2.5 * std)
            if temp_min > min_val:
                min_val = temp_min
            if temp_max < max_val:
                max_val = temp_max
        min_val *= multiplier
        max_val *= multiplier
        self.min_max_dict[attribute] = (min_val, max_val)
        return min_val, max_val

    def _layer_min_max(self, attribute, min_val=0, max_val=1, axis="y_Pos", log=False, absolute=False):
        parameter = attribute
        if log:
            attribute += "_log"
        if absolute:
            attribute += "_absolute"
        if attribute in self.layers_min_max_dict.keys():
            return self.layers_min_max_dict[attribute]
        for well in self.wells:
            well_df = self.well_info[well]
            # --- Fix: skip wells missing the parameter ---
            if parameter not in well_df.columns:
                continue
            well_df = well_df.dropna(subset=[parameter])
            middle, jump, distances = self._middle_info(well, axis)
            for time_index in well_df.TimeIndex.unique():
                time_df = well_df[well_df.TimeIndex == time_index]
                layer_dfs = [time_df[abs(time_df[axis] - middle) >= jump * i] for i in range(len(distances))]
                layer_dfs = [layer_dfs[i][abs(layer_dfs[i][axis] - middle) < jump * (i+1)] for i in range(len(distances))]
                values = np.array([layer_df[parameter] for layer_df in layer_dfs])
                if log:
                    values = np.log(values)
                if absolute:
                    values = np.absolute(values)
                avges = [np.average(value_series) for value_series in values]
                temp_min, temp_max = min(avges), max(avges)
                if temp_min < min_val:
                    min_val = temp_min
                if temp_max > max_val:
                    max_val = temp_max
        self.layers_min_max_dict[attribute] = (min_val, max_val)
        return min_val, max_val

    def _scratch_min_max(self, attribute, min_val=0, max_val=1, absolute=False, intervals=15, groups=False, fixed_distance=30, scaled=False):
        parameter = attribute
        if absolute:
            attribute += "_absolute"
        if scaled:
            attribute += "_scaled"
        if attribute in self.scratch_min_max_dict.keys():
            return self.scratch_min_max_dict[attribute]
        all_values = []
        for well in self.wells:
            well_df = self.well_info[well]
            # --- Fix: skip wells missing the parameter ---
            if parameter not in well_df.columns:
                continue
            well_df = well_df.dropna(subset=[parameter])
            scratch_info_per_time = self._scratch_info(well)
            time_indexes = well_df.TimeIndex.unique()
            for time_index in time_indexes:
                time_df = well_df[well_df.TimeIndex == time_index]
                scratch_top, scratch_bottom, top_interval, bottom_interval = scratch_info_per_time[time_index]
                if not groups:
                    top_interval = fixed_distance
                    bottom_interval = fixed_distance
                for i in range(intervals):
                    top_group = time_df[time_df.y_Pos >= scratch_top + (i * top_interval)]
                    if i != intervals - 1:
                        top_group = top_group[top_group.y_Pos < scratch_top + ((i+1) * top_interval)]
                    bottom_group = time_df[time_df.y_Pos <= scratch_bottom - (i * bottom_interval)]
                    if i != intervals - 1:
                        bottom_group = bottom_group[bottom_group.y_Pos > scratch_bottom - ((i+1) * bottom_interval)]
                    top_group = top_group[parameter]
                    bottom_group = bottom_group[parameter]
                    if absolute:
                        top_group = np.absolute(top_group)
                        bottom_group = np.absolute(bottom_group)
                    try:
                        temp_avg = (sum(top_group) + sum(bottom_group)) / (len(top_group) + len(bottom_group))
                    except ZeroDivisionError:
                        temp_avg = 0
                    all_values.append(temp_avg)
        min_val = min(all_values)
        max_val = max(all_values)
        if scaled:
            all_values = np.array(all_values)
            mean = all_values.mean()
            std = all_values.std()
            temp_min = mean - (2.5 * std)
            temp_max = mean + (2.5 * std)
            if temp_min > min_val:
                min_val = temp_min
            if temp_max < max_val:
                max_val = temp_max
        self.scratch_min_max_dict[attribute] = (min_val, max_val)
        return min_val, max_val

    def _sunplot_max(self):
        try:
            return self.sunplot_ax_size
        except:
            pass
        max_distance = 0
        x_max_distances = []
        y_max_distances = []
        for well in self.wells:
            well_df = self.well_info[well]
            cells = well_df.Parent.unique()
            for cell in cells:
                x_positions = np.array(well_df.x_Pos[well_df.Parent == cell])
                y_positions = np.array(well_df.y_Pos[well_df.Parent == cell])
                x_distances = x_positions - x_positions[0]
                y_distances = y_positions - y_positions[0]
                x_max_distances.append(max(abs(x_distances)))
                y_max_distances.append(max(abs(y_distances)))
        self.sunplot_ax_size = max([max(x_max_distances), max(y_max_distances)])
        return self.sunplot_ax_size

    def _middle_info(self, well, axis, intervals=15, log=False):
        dict_key = well + "_" + axis
        if log:
            dict_key += "_log"
        if dict_key in self.middles_dict.keys():
            return self.middles_dict[dict_key]
        axis_column = self.well_info[well][axis]
        min_val, max_val = min(axis_column), max(axis_column)
        middle = round((max_val - min_val) / 2)
        jump = round((max_val - min_val) / (2 * intervals))
        distances = ["%d - %d" % (jump * i, jump * (i+1)) for i in range(intervals)]
        distances[-1] = distances[-1].split("-")[0] + "+"
        self.middles_dict[dict_key] = [middle, jump, distances]
        return middle, jump, distances

    def _scratch_info(self, well, intervals=15):
        """
        Finds the top and bottom of the scratch for each time points, and divides the remaining area into a number (10 by default) of equally sized intervals.
        A scratch is defined as a distance between two cell centers of mass on the Y axis that is more than 2 standard deviations above the mean Y distance.
        """
        output_folder = os.path.join(self.output_path, "scratches")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if well in self.scratch_dict.keys():
            return self.scratch_dict[well]
        well_df = self.well_info[well]
        time_indexes = well_df.TimeIndex.unique()
        time_indexes.sort()
        time_indexes = list(time_indexes)
        scratch_per_time = {}
        max_y_span = max(well_df.y_Pos) - min(well_df.y_Pos)
        scratch_closed = False
        previous_time = 0
        middle = (max(well_df.y_Pos) + min(well_df.y_Pos)) / 2
        intervals_from_middle = max_y_span / (2 * intervals)
        for time_index in time_indexes:
            if not scratch_closed:
                df = well_df[well_df.TimeIndex == time_index]
                y_positions = np.array(df.y_Pos)
                y_positions.sort()
                y_positions = y_positions[::-1]
                diffs = (y_positions[:-1] - y_positions[1:]) / max_y_span
                mean_diff = np.mean(diffs)
                diff_std = np.std(diffs)
                outlier_diff_size = mean_diff + 2 * diff_std
                noise_cells_allowed = len(df) / 100
                for i in range(1, 10):
                    scratch_indices = np.where(diffs >= i * outlier_diff_size)[0] # dynamic
                    if scratch_indices.size == 0:
                        scratch_closed = True
                    elif scratch_indices[-1] - scratch_indices[0] > noise_cells_allowed:
                        if i == 9:
                            print("============================================ERROR MESSAGE============================================")
                            for i in range(len(diffs)):
                                if diffs[i] > i * outlier_diff_size:
                                    print("index:", i, "y positions:",y_positions[i], y_positions[i+1], "diff:", diffs[i])
                            raise ValueError("Found more than one scratch. Try adjusting the min scratch diff, or call Ori")
                    else:
                        scratch_top = y_positions[scratch_indices[0]]
                        scratch_bottom = y_positions[scratch_indices[-1] + 1]
                        if previous_time != 0:
                            previous_top, previous_bottom, _, _ = scratch_per_time[previous_time]
                            if scratch_top < previous_bottom or scratch_bottom > previous_top:
                                scratch_closed = True
                        if scratch_top - scratch_bottom < max_y_span / 1000:
                            scratch_closed = True
                        else:
                            top_interval = (max(y_positions) - scratch_top) / intervals
                            bottom_interval = (scratch_bottom - min(y_positions)) / intervals
                            scratch_per_time[time_index] = (scratch_top, scratch_bottom, top_interval, bottom_interval)
                        # Creating timepoint scratch graphs
                        min_x = min(df.x_Pos)
                        x_size = max(df.x_Pos) - min_x
                        fig, ax = plt.subplots()
                        if not scratch_closed:
                            rect = patches.Rectangle((min_x, scratch_bottom), width=x_size, height=scratch_top - scratch_bottom, color="red")
                            ax.add_patch(rect)
                        ax.scatter(df.x_Pos, df.y_Pos, zorder=50)
                        ax.set_title(self.shortened_well_names[well] + " TimeIndex %d" % time_index)
                        ax.set_xlabel("x_Pos")
                        ax.set_ylabel("y_Pos")
                        fig.savefig(os.path.join(output_folder, self.shortened_well_names[well] + "_%d" % time_index + ".jpg"))
                        plt.close(fig)
                        break # break here because we found a good scratch - no need to continue making a more generous minimum distance
            if scratch_closed:
                scratch_per_time[time_index] = (middle, middle, intervals_from_middle, intervals_from_middle)
            previous_time = time_index
        self.scratch_dict[well] = scratch_per_time
        return scratch_per_time

    def _get_msd_info(self):
        self.max_msd = 0
        self.max_tau = 1
        for well in self.wells:
            if well not in self.msd_info.keys():
                self.msd_info[well] = {"average": {}, "msd_dicts": [], "max_msd": 0, "max_tau": 1}
                well_df = self.well_info[well].copy()
                cells = well_df.Parent.unique()
                well_df.set_index("Parent", inplace=True)
                for cell in cells:
                    cell_df = well_df.loc[[cell]] if isinstance(well_df.loc[cell], pd.Series) else well_df.loc[cell]
                    if isinstance(cell_df, pd.Series):
                        cell_df = cell_df.to_frame().T
                    cell_df.set_index("TimeIndex", inplace=True)
                    if len(cell_df) > 6:
                        temp_dict = {}
                        cell_df = cell_df.reindex(range(cell_df.index[0], cell_df.index[-1] + 1)) # fills with NaN
                        max_tau = int(np.ceil((cell_df.index[-1] - cell_df.index[0]) / 2))
                        self.msd_info[well]["max_tau"] = max([self.msd_info[well]["max_tau"], max_tau])
                        for tau in range(1, max_tau + 1):
                            deltas_squared = np.array([0] * (len(cell_df) - tau))
                            for pos in POSITIONS[:self.dimensions]:
                                deltas_squared = deltas_squared + (np.array(cell_df[pos][tau:]) - np.array(cell_df[pos][:-tau])) ** 2
                            deltas_squared = deltas_squared[~np.isnan(deltas_squared)] # throws out missing values
                            if len(deltas_squared) != 0:
                                msd = sum(deltas_squared) / len(deltas_squared)
                                temp_dict[tau] = msd
                                self.msd_info[well]["max_msd"] = max([msd, self.msd_info[well]["max_msd"]])
                        if len(temp_dict) >= 3 and 1 in temp_dict.keys():
                            self.msd_info[well]["msd_dicts"].append(temp_dict)
                for tau in range(1, self.msd_info[well]["max_tau"]):
                    tau_msds = []
                    for msd_dict in self.msd_info[well]["msd_dicts"]:
                        try:
                            tau_msds.append(msd_dict[tau])
                        except KeyError:
                            pass
                    if tau_msds != []:
                        self.msd_info[well]["average"][tau] = sum(tau_msds) / len(tau_msds)
            self.max_msd = max(self.max_msd, self.msd_info[well]["max_msd"])
            self.max_tau = max(self.max_tau, self.msd_info[well]["max_tau"])
        self.msd_info = self.msd_info

    def load_wells_from_summary_table(self, summary_table):
        """
        This function takes in account that the wells were provided in an xls2 format - so it changes self.wells to the
        full names as they appear in the summary table before loading the well info.
        """
        full_well_names = list(set(summary_table["Experiment"]))
        if len(self.wells[0]) == 3:
            locations = self.wells
        elif len(self.wells[0]) == 8:
            locations = self.wells[-3:]
        else:
            locations = [well[9:12] for well in self.wells]
        self.wells = [well for well in full_well_names if well[15:18] in locations]

        self.well_amount = len(self.wells)
        # fixed self.wells to be in the summary_table format
        self.parameters = list(summary_table.keys())
        if "z_Pos" in self.parameters:
            self.dimensions = 3
        else:
            self.dimensions = 2
        for well in self.wells:
            self.well_info[well] = summary_table[summary_table["Experiment"] == well]

        self.shortened_well_names = {well: well[18:-4].replace("NNN0", "") for well in self.wells}
        all_short_names = list(self.shortened_well_names.values())
        first_val = all_short_names[0][:4]
        if False not in [name[:4] == first_val for name in list(self.shortened_well_names.values())]:
            self.shortened_well_names = {well: well[22:-4].replace("NNN0", "") for well in self.wells}
        if True in [all_short_names.count(name) > 1 for name in all_short_names]:
            for well in self.wells:
                self.shortened_well_names[well] = well[15:18] + self.shortened_well_names[well]
        self.wells.sort(key=lambda x: self.shortened_well_names[x])

    def load_wells_from_summary_folder(self, summary_folder):
        all_files = os.listdir(summary_folder)
        full_well_files = [f for f in all_files if "_full" in f.lower() and "summary_table_" in f.lower() and f[0] != "~"]
        full_well_names = [f.split("_")[-2] for f in full_well_files]

        if len(self.wells[0]) == 3:
            locations = self.wells
        elif len(self.wells[0]) == 8:
            locations = [well[-3:] for well in self.wells]
        else:
            locations = [well[9:12] for well in self.wells]
        self.wells = [well for well in full_well_names if well[15:18] in locations]

        self.well_amount = len(self.wells)
        # fixed self.wells to be in the summary_table format
        self.parameters = []
        for well in self.wells:
            well_files = [f for f in full_well_files if well in f]
            if len(well_files) != 1:
                raise ValueError("Couldn't find the right file for %s or found too many, check summary folder" % well)
            well_file = well_files[0]
            well_df = pd.read_excel(os.path.join(summary_folder, well_file))
            well_df.drop(columns=["Unnamed: 0", "Parent_OLD"], inplace=True, errors="ignore")
            well_df.dropna(axis="columns", how="all", inplace=True)
            parameters = list(well_df.keys())
            if self.parameters == []:
                self.parameters = parameters
                if "z_Pos" in self.parameters:
                    self.dimensions = 3
                else:
                    self.dimensions = 2
            elif parameters != self.parameters:
                print(f"Warning: might have Different parameters for well {well}. Using intersection of columns. can be due to different number of channels between two wells")
                cols = list(set(parameters) & set(self.parameters))
                if(len(self.parameters) < len(parameters)):
                    self.parameters = parameters
                    cols = list(set(self.parameters))
                well_df = well_df[cols]
            self.well_info[well] = well_df


        self.shortened_well_names = {well: well[18:-4].replace("NNN0", "") for well in self.wells}
        all_short_names = list(self.shortened_well_names.values())
        first_val = all_short_names[0][:4]
        if False not in [name[:4] == first_val for name in list(self.shortened_well_names.values())]:
            self.shortened_well_names = {well: well[22:-4].replace("NNN0", "") for well in self.wells}
        if True in [all_short_names.count(name) > 1 for name in all_short_names]:
            for well in self.wells:
                self.shortened_well_names[well] = well[15:18] + self.shortened_well_names[well]
        self.wells.sort(key=lambda x: self.shortened_well_names[x])

    def make_output_folders(self, original_output_path):
        #Creates the output path and making sure it's empty
        if not os.path.exists(original_output_path):
            os.mkdir(original_output_path)
        output_path = os.path.join(original_output_path, self.exp_name.replace(" ", "_"))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path = os.path.join(output_path, self.exp_sub_name.replace(" ", "_"))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        already_run = os.listdir(output_path)
        if already_run != []:
            if os.path.join(original_output_path, self.exp_name + "_report.pdf") in os.listdir(original_output_path):
                already_run = "ALL"
            #raise ValueError("Folder %s has stuff in it! Can't overwrite, clear it out or change the experiment name or sub name." % output_path)
        self.output_path = output_path
        return already_run

    def set_automatic_coords(self):
        for row_num in range(1, 5):
            current_row_width = WORKING_WIDTH / int(np.ceil(self.well_amount / row_num))
            another_row_width = WORKING_HEIGHT / (row_num + 1)
            if another_row_width > current_row_width:
                self.row_num = row_num + 1
            else:
                self.row_num = row_num
                break
        self.graph_width = min(WORKING_WIDTH / int(np.ceil(self.well_amount / self.row_num)),
                               WORKING_HEIGHT / self.row_num)
        self.graph_x_coords = []
        self.graph_y_coords = []
        last_row_graphs = self.well_amount
        height_break = (WORKING_HEIGHT - self.graph_width * self.row_num) / self.row_num
        # find coords for all rows except last:
        if self.row_num > 1:
            full_row_graphs = int(np.ceil(self.well_amount / self.row_num))
            last_row_graphs -= full_row_graphs * (self.row_num - 1)
            full_row_break = (WORKING_WIDTH - (self.graph_width * full_row_graphs)) / (full_row_graphs - 1)
            for i in range(self.row_num - 1):
                self.graph_x_coords += [(full_row_break + self.graph_width) * j for j in range(full_row_graphs)]
                self.graph_y_coords += [30 + (height_break / 2) + ((self.graph_width + height_break) * i)] * full_row_graphs
        # find coords for last row:
        if last_row_graphs < self.well_amount / self.row_num:
            last_row_break = (WORKING_WIDTH - (self.graph_width * last_row_graphs)) / last_row_graphs
            self.graph_x_coords += [(last_row_break / 2) + (self.graph_width + last_row_break) * i for i in
                                    range(last_row_graphs)]
        else:
            last_row_break = (WORKING_WIDTH - (self.graph_width * last_row_graphs)) / (last_row_graphs - 1)
            self.graph_x_coords += [(self.graph_width + last_row_break) * i for i in range(last_row_graphs)]
        self.graph_y_coords += [30 + (height_break / 2) + ((self.graph_width + height_break) * (self.row_num - 1))] * last_row_graphs

    def _get_auto_table_coords(self):
        well_treatments = []
        for well in self.wells:
            short_name = self.shortened_well_names[well]
            if len(short_name) > 8:
                raise ValueError("A well (%s) has more than 2 treatments - we can't create a table." % short_name)
            well_treatments += [short_name[i:i+4] for i in range(len(short_name) // 4)]
        well_treatments = list(set(well_treatments))
        treatment_names = list(set([treat[:3] for treat in well_treatments if treat[:3] not in ["CON", "CTR"]]))
        if len(treatment_names) > 2:
            raise ValueError("Found more than 2 treatments - %s. Couldn't create a table" % ", ".join(treatment_names))
        treatment_options = {treat: [wt for wt in well_treatments if wt[:3] == treat] for treat in treatment_names}
        treatment_names.sort(lambda x: len(treatment_options[x]), reverse=True)
        columns = list(treatment_options[treatment_names[0]])
        columns.sort(key=lambda x: int(x[-1]))
        self.column_treatments = columns
        self.column_names = ["Control"] + columns
        rows = list(treatment_options[treatment_names[-1]])
        rows.sort(key=lambda x: int(x[-1]))
        self.row_treatments = rows
        self.row_names = ["Control"] + rows

    def set_table_coords(self):
        if not self.table_info: # not sure this is useful but I already wrote it
            self._get_auto_table_coords()
        else:
            self.column_treatments = self.table_info["column_treatments"]
            self.row_treatments = self.table_info["row_treatments"]
            self.column_names = self.table_info["column_names"]
            self.row_names = self.table_info["row_names"]
            if self.design == "auto_table":
                self.column_names = ["Control"] + self.column_names
                self.row_names = ["Control"] + self.row_names
        self.wells.sort(key=lambda x: self.table_info["cell_order"].index(x[15:18]))
        self.row_len = len(self.column_names)
        self.col_len = len(self.row_names)
        width = WORKING_WIDTH - 20
        height = WORKING_HEIGHT - 10
        self.graph_width = min([width / self.row_len, height / self.col_len])
        row_break = (width - (self.graph_width * self.row_len)) / (self.row_len - 1)
        self.graph_x_coords = [20 + (self.graph_width + row_break) * i for i in range(self.row_len)] * self.col_len
        height_break = (height - (self.graph_width * self.col_len)) / self.col_len
        self.graph_y_coords = []
        for i in range(self.col_len):
            self.graph_y_coords += [40 + (height_break / 2) + (self.graph_width + height_break) * i] * self.row_len

    def create_pdf_file(self):
        self.pdf = PDF(orientation="L")
        self.pdf.set_margins(0, 0, 0)
        for size in range(20, 5, -1):
            self.pdf.set_font("arial", "b", size)
            if self.pdf.get_string_width(self.exp_name) <= WORKING_WIDTH:
                self.header_size = size
                break
        # Following part calculates how many rows and cols we need for graph pages. Might need some work in future
        if self.well_amount <= 1:
            raise ValueError("Must compare at least two wells.")
        elif self.well_amount > 40:
            raise ValueError("Script can't handle more than 40 graphs on one page.")
        if self.design == "auto":
            self.set_automatic_coords()
        elif "table" in self.design:
            self.set_table_coords()

    def new_page(self, header, table=False):
        self.pdf.add_page()
        self.pdf.set_font("arial", "b", self.header_size)
        self.pdf.cell(WORKING_WIDTH, h=10, txt=self.exp_name, ln=1, align="C")
        self.pdf.cell(WORKING_WIDTH, h=10, txt=self.exp_sub_name, ln=1, align="C")
        for size in range(self.header_size - 2, 1, -1):
            self.pdf.set_font("arial", "", size)
            if self.pdf.get_string_width(header) <= WORKING_WIDTH:
                self.pdf.cell(WORKING_WIDTH, h=10, txt=header, ln=1, align="C")
                break
        self.graph_counter = 97
        if table and "table" in self.design:
            self.pdf.set_font("arial", "", 8)
            self.pdf.cell(20, h=10, txt="", ln=0)
            for i in range(self.row_len):
                self.pdf.cell((WORKING_WIDTH - 20) / self.row_len, h=10, txt=self.column_names[i], ln=0, align="C")
                #self.pdf.line(self.graph_x_coords[i], 30, self.graph_x_coords[i], 210)
            self.pdf.ln()
            y_positions = list(set(self.graph_y_coords))
            y_positions.sort()
            col_size = min(20, self.graph_width)
            for i in range(self.col_len):
                self.pdf.set_xy(0, y_positions[i])
                self.pdf.cell(col_size, h=10, txt=self.row_names[i], ln=1, align="C")
                #self.pdf.line(0, self.graph_y_coords[i], 297, self.graph_y_coords[i])

    def write_table_to_pdf(self, columns):
        y_skip = ((WORKING_HEIGHT - (10 * len(columns))) / 2) - 10
        self.pdf.ln(h=y_skip)
        longest_row_list = [""] * len(columns)
        for i in range(len(columns[0])):
            for j in range(len(columns)):
                if columns[j][i] > longest_row_list[j]:
                    longest_row_list[j] = columns[j][i]
        longest_row = "".join(longest_row_list)
        ratios = [len(longest_row_list[j]) / len(longest_row) for j in range(len(longest_row_list))]
        for size in range(self.header_size - 4, 1, -1):
            self.pdf.set_font("arial", "", size)
            if self.pdf.get_string_width(longest_row) < WORKING_WIDTH - 20: # -20 is just to be sure it doesn't go out of the lines
                break
        for i in range(len(columns[0])):
            for j in range(len(columns)):
                self.pdf.cell(WORKING_WIDTH * ratios[j], txt=columns[j][i], h=10, border=1, align="C")
            self.pdf.ln(h=10)

    def make_first_page(self):
        self.new_page("Run information")
        columns = [["Date of run", "Time of run", "Computer name", "Pybatch version"],
                   [time.strftime("%Y-%m-%d"), time.strftime("%H:%M"), os.getenv("COMPUTERNAME"), "1.1"]]
        self.write_table_to_pdf(columns)

    def make_second_page(self):
        self.new_page("File information")
        columns = [["Protocol_file", "Incucyte files", "Imaris xls files"], [self.protocol_file, self.incucyte_files, self.imaris_xls_files]]
        if self.__has_attribute__("xls1_path"):
            columns[0].append("Excel Renaming (1)")
            columns[1].append(self.xls1_path)
        if self.__has_attribute__("xls2_path"):
            columns[0].append("Excel Layout (2)")
            columns[1].append(self.xls2_path)
        self.write_table_to_pdf(columns)

    def make_third_page(self):
        self.new_page("Well information")
        initials = ["Initials"] + [well[:2] for well in self.wells]
        exp_num = ["Exp num"] + [well[2:5] for well in self.wells]
        date = ["Date"] + [well[5:11] for well in self.wells]
        channel = ["Channel"] + [well[11:14] for well in self.wells]
        plate_num = ["Plate num"] + [well[14] for well in self.wells]
        locations = ["Location"] + [well[15:18] for well in self.wells]
        cell_line = ["Cell line"] + [well[18:22] for well in self.wells]
        treatments = ["Treatments"] + [well[22:-4].replace("NNN0", "") for well in self.wells]
        exp_type = ["Exp type"] + [well[-4:] for well in self.wells]
        columns = [initials, exp_num, date, channel, plate_num, locations, cell_line, treatments, exp_type,
                   ["Shortened Name"] + [self.shortened_well_names[well] for well in self.wells]]
        self.write_table_to_pdf(columns)

    def draw_sunplot(self, well, output_folder=None, axis1="X", axis2="Y", cell_amount_limit=None):
        fig, graph_ax = plt.subplots()
        well_df = self.well_info[well]
        well_name = self.shortened_well_names[well]
        ax_size = self._sunplot_max()
        cells = list(well_df.Parent.unique())
        cells.sort(key=lambda x: list(well_df.Overall_Displacement[well_df.Parent == x])[0], reverse=True)
        if cell_amount_limit:
            cells = np.random.choice(cells, size=cell_amount_limit, replace=False)
            #quarter_amount = int(cell_amount_limit / 4)
            #cells = cells[:quarter_amount] + \
            #        list(np.random.choice(cells, size=cell_amount_limit - (2 * quarter_amount), replace=False)) + \
            #        cells[-quarter_amount:]
        for cell in cells:
            x_positions = np.array(well_df[axis1.lower()+"_Pos"][well_df.Parent == cell])
            y_positions = np.array(well_df[axis2.lower()+"_Pos"][well_df.Parent == cell])
            x_distances = x_positions - x_positions[0]
            y_distances = y_positions - y_positions[0]
            graph_ax.plot(x_distances, y_distances)
        graph_ax.set_title("%d%s - " % (self.pdf.page_no(), chr(self.graph_counter)) + well_name)
        self.graph_counter += 1
        graph_ax.set_xlabel(axis1 + r" position [$\mu$m]")
        graph_ax.set_xbound(lower=-ax_size, upper=ax_size)
        graph_ax.set_ylabel(axis2 + r" position [$\mu$m]")
        graph_ax.set_ybound(lower=-ax_size, upper=ax_size)
        graph_ax.set_box_aspect(1)
        if output_folder:
            fig.savefig(os.path.join(output_folder, well + "_" + axis1 + axis2 + "_sunplot.jpg"))
            plt.close(fig)
        else:
            plt.show()

    def make_sunplot_page(self):
        output_folder = os.path.join(self.output_path, "sunplots")
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        if self.dimensions == 2:
            axis_pairs = ["XY"]
        elif self.dimensions == 3:
            axis_pairs = ["XY", "XZ", "YZ"]
        cell_amount_limit = min([len(well_df.Parent.unique()) for well_df in self.well_info.values()])
        for axis_pair in axis_pairs:
            self.new_page("Sunplots - trajectories of %d randomly selected cells from initial %s,%s position" %
                          (cell_amount_limit, axis_pair[0], axis_pair[1]), table=True)
            for i in range(self.well_amount):
                well = self.wells[i]
                if not os.path.exists(os.path.join(output_folder, well + "_" + axis_pair + "_sunplot.jpg")):
                    self.draw_sunplot(well, output_folder=output_folder, axis1=axis_pair[0], axis2=axis_pair[1],
                                      cell_amount_limit=cell_amount_limit)
                self.pdf.image(os.path.join(output_folder, well + "_" + axis_pair + "_sunplot.jpg"),
                               x=self.graph_x_coords[i], y=self.graph_y_coords[i], w=self.graph_width)

    def draw_scratch(self, well, output_folder=None):
        fig, graph_ax = plt.subplots()
        well_df = self.well_info[well]
        well_name = self.shortened_well_names[well]
        graph_ax.set_title("%d%s - " % (self.pdf.page_no(), chr(self.graph_counter)) + well_name)
        self.graph_counter += 1
        if self.graph_counter == 123:
            self.graph_counter = 65
        graph_ax.set_xlabel(r"Time [min]")
        graph_ax.set_xbound(lower=-1000, upper=1000)
        graph_ax.set_ylabel(r"Y position [$\mu$m]")
        graph_ax.set_ybound(lower=-1000, upper=1000)
        scratch_info_per_time = self._scratch_info(well)
        time_indexes = well_df.TimeIndex.unique()
        time_indexes.sort()
        for time_index in time_indexes:
            scratch_top, scratch_bottom, _, _ = scratch_info_per_time[time_index]
            graph_ax.vlines(time_index * self.dt, ymin=scratch_bottom, ymax=scratch_top, colors="red")
        graph_ax.scatter(np.multiply(well_df.TimeIndex, self.dt), well_df.y_Pos, zorder=0)
        graph_ax.set_box_aspect(1)
        if output_folder:
            fig.savefig(os.path.join(output_folder, well + ".jpg"))
            plt.close(fig)
        else:
            plt.show()

    def make_scratch_pages(self):
        print("making scratch pages")
        output_folder = os.path.join(self.output_path, "scratches")
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page("Scratches", table=True)

        for i in range(self.well_amount):
            well = self.wells[i]
            if not os.path.exists(os.path.join(output_folder, well + ".jpg")):
                self.draw_scratch(well, output_folder=output_folder)
            self.pdf.image(os.path.join(output_folder, well + ".jpg"), x=self.graph_x_coords[i], y=self.graph_y_coords[i], w=self.graph_width)
        all_images = os.listdir(output_folder)
        all_images = [image for image in all_images if "_" in image]
        for well in self.wells:
            well_name = self.shortened_well_names[well]
            well_images = [image for image in all_images if well_name == image.split("_")[0]]
            well_images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if len(well_images) <= 8:
                scratch_graph_x = EIGHT_GRAPH_X
                scratch_graph_y = EIGHT_GRAPH_Y
                scratch_graph_width = EIGHT_GRAPH_WIDTH
            elif len(well_images) <= 18:
                scratch_graph_x = EIGHTEEN_GRAPH_X
                scratch_graph_y = EIGHTEEN_GRAPH_Y
                scratch_graph_width = EIGHTEEN_GRAPH_WIDTH
            elif len(well_images) <= 32:
                scratch_graph_x = THIRTYTWO_GRAPH_X
                scratch_graph_y = THIRTYTWO_GRAPH_Y
                scratch_graph_width = THIRTYTWO_GRAPH_WIDTH
            else:
                scratch_graph_x = FOURTY_GRAPH_X
                scratch_graph_y = FOURTY_GRAPH_Y
                scratch_graph_width = FOURTY_GRAPH_WIDTH
            for part in range(int((len(well_images) - 1) / 40) + 1):
                title = well_name + " scratch over time"
                if part > 0:
                    title += " (continued)"
                self.new_page(title)
                part_len = len(well_images) - 40 * part
                if part_len > 40:
                    part_len = 40
                for i in range(part_len):
                    self.pdf.image(os.path.join(output_folder, well_images[i + (part_len * part)]),
                                   x=scratch_graph_x[i], y=scratch_graph_y[i], w=scratch_graph_width)

    def draw_y_pos_time_graph(self, well, parameter, log=False, output_folder=None, absolute=False):
        fig, graph_ax = plt.subplots()
        well_df = self.well_info[well]
        if parameter not in well_df.columns:
            print(f"draw_y_pos_time_graph : Skipping {parameter} for well {well}: column missing.")
            return  # Skip this plot
        well_df = well_df.dropna(subset=[parameter])
        well_name = self.shortened_well_names[well]
        min_time, max_time = self._min_max("TimeIndex", max_val=self.dt, multiplier=self.dt, scaled=False)
        min_pos, max_pos = self._min_max("y_Pos", max_val=2000, scaled=False)
        min_val, max_val = self._min_max(parameter, log=log, absolute=absolute)
        x = well_df["TimeIndex"] * self.dt
        y = well_df["y_Pos"]
        c = well_df[parameter]
        cb_label = parameter + UNIT_DICT[parameter]
        graph_ax.set_title("%d%s - " % (self.pdf.page_no(), chr(self.graph_counter)) + well_name)
        self.graph_counter += 1
        if self.graph_counter == 123:
            self.graph_counter = 65
        graph_ax.set_xlabel("Time [min]")
        graph_ax.set_xbound(lower=min_time, upper=max_time)
        graph_ax.set_ylabel(r"Y position [$\mu$m]")
        graph_ax.set_ybound(lower=min_pos, upper=max_pos)
        if log:
            c = np.log(c)
            cb_label = "log of " + cb_label
        if absolute:
            c = np.absolute(c)
            cb_label += " absolute"
        sc = graph_ax.scatter(x, y, c=c, s=10, vmin=min_val, vmax=max_val, cmap="jet")
        colorbar = fig.colorbar(sc, ax=graph_ax)
        colorbar.set_label(cb_label)
        ticks = [min_val + ((max_val - min_val) * i / 6) for i in range(7)]
        colorbar.set_ticks(ticks)
        ticks = [str(round(tick, 2)) for tick in ticks]
        ticks[0] = "<" + ticks[0]
        ticks[-1] = ">" + ticks[-1]
        colorbar.set_ticklabels(ticks)
        graph_ax.set_box_aspect(1)

        fig.tight_layout()
        if output_folder:
            fig.savefig(os.path.join(output_folder, well + ".jpg"))
            plt.close(fig)
        else:
            plt.show()

    def make_y_pos_time_page(self, parameter, log=False, absolute=False):
        """
        This function should be generalized to receive the graph function as well as the specific parameter.
        """
        title = parameter + " per y position over time"
        if log:
            title = "log of " + title
        if absolute:
            title += " absolute"
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title, table=True)

        for i in range(self.well_amount):
            well = self.wells[i]
            image_path = os.path.join(output_folder, well + ".jpg")
            if not os.path.exists(image_path):
                self.draw_y_pos_time_graph(well, parameter, log=log, output_folder=output_folder, absolute=absolute)
            if os.path.exists(image_path):
                self.pdf.image(os.path.join(output_folder, well + ".jpg"),
                               x=self.graph_x_coords[i], y=self.graph_y_coords[i], w=self.graph_width)
            else:
                print(f"make_y_pos_time_page : Skipping PDF image for {well} and {parameter}: image not created.")

    def draw_average_over_time(self, parameter, output_folder=None, absolute=False):
        fig, graph_ax = plt.subplots()
        title = "Average " + parameter + " over time"
        if absolute:
            title = "Absolute a" + title[1:]
        graph_ax.set_title("%d%s - " % (self.pdf.page_no(), chr(self.graph_counter)) + title)
        self.graph_counter += 1
        if self.graph_counter == 123:
            self.graph_counter = 65
        graph_ax.set_xlabel("Time [min]")
        graph_ax.set_ylabel("Average " + parameter + UNIT_DICT[parameter] + r" (+-$\frac{\sigma}{\sqrt{n}}$)")
        lines = []
        well_names = []
        for i in range(self.well_amount):
            color = LINE_COLORS[i]
            well = self.wells[i]
            well_df = self.well_info[well]
            # --- Fix: skip wells missing the parameter ---
            if parameter not in well_df.columns:
                continue
            well_df = well_df.dropna(subset=[parameter])
            well_name = self.shortened_well_names[well]
            x = well_df.TimeIndex.unique()
            x.sort()
            vals = [well_df[well_df.TimeIndex == time_index][parameter] for time_index in x]
            x = x * self.dt
            if absolute:
                vals = [np.absolute(val) for val in vals]
            y = [np.average(val) for val in vals]
            yerr = [np.std(val) / (len(val) ** 0.5) for val in vals]
            line = graph_ax.plot(x, y, color=color)[0]
            color = line.get_color()
            for j in range(len(yerr)):
                graph_ax.errorbar(x[j], y[j], yerr=yerr[j], ecolor=color)
            lines.append(line)
            well_names.append(well_name)
        if self.well_amount <= 5:
            graph_ax.legend(lines, well_names)
        graph_ax.set_box_aspect(1)
        fig.tight_layout()
        if output_folder:
            fig.savefig(os.path.join(output_folder, "over_time.jpg"))
            plt.close(fig)
        else:
            plt.show()

    def draw_average_barplot(self, parameter, output_folder=None, absolute=False, wave=False, alt_name=None, per_cell=False):
        global average_df
        fig, graph_ax = plt.subplots()
        graph_ax.set_title("%d%s - Average " % (self.pdf.page_no(), chr(self.graph_counter)) + parameter)
        self.graph_counter += 1
        if self.graph_counter == 123:
            self.graph_counter = 65
        graph_ax.set_xlabel("Treatment")
        graph_ax.set_ylabel("Average " + parameter + UNIT_DICT[parameter])
        x = []  # range(self.well_amount)
        height = []
        yerrors = []
        x_tick_labels = []
        exp_col = []
        values_col = []
        for idx, well in enumerate(self.wells):
            well_df = self.well_info[well]
            if parameter not in well_df.columns:
                continue  # skip wells missing the parameter
            well_df = well_df.dropna(subset=[parameter])
            if per_cell == False:
                values = np.array(well_df[parameter])
            else:
                cells = well_df.Parent.unique()
                if wave == "all":
                    cells = [cell for cell in cells if
                             well_df[well_df.Parent == cell].iloc[0].Velocity_Full_Width_Half_Maximum != 0]
                elif wave == "double":
                    cells = [cell for cell in cells if
                             well_df[well_df.Parent == cell].iloc[0].Velocity_Starting_Value != 0 and \
                             well_df[well_df.Parent == cell].iloc[0].Velocity_Ending_Value != 0]
                elif wave == "left":
                    cells = [cell for cell in cells if
                             well_df[well_df.Parent == cell].iloc[0].Velocity_Starting_Value != 0 and \
                             well_df[well_df.Parent == cell].iloc[0].Velocity_Ending_Value == 0]
                elif wave == "right":
                    cells = [cell for cell in cells if
                             well_df[well_df.Parent == cell].iloc[0].Velocity_Starting_Value == 0 and \
                             well_df[well_df.Parent == cell].iloc[0].Velocity_Ending_Value != 0]
                values = [well_df[well_df.Parent == cell].iloc[0][parameter] for cell in cells]
            if len(values) == 0:
                continue  # skip if no values
            if absolute:
                values = np.absolute(values)

        
            exp_col.append(well)
            values_col.append(sum(values)/len(values))

            x.append(len(x))  # index for bar position
            height.append(np.average(values))
            yerrors.append(np.std(values) / (len(values) ** 0.5))
            x_tick_labels.append(self.shortened_well_names[well])

        df = pd.DataFrame({"Experiment":exp_col , "Parameter":[parameter]*len(exp_col) , "Average_Value":values_col})
        if average_df is None:
            average_df = df
        else:
            average_df = pd.concat([average_df,df], ignore_index=True)

        graph_ax.bar(x, height, color=LINE_COLORS[:len(x)], yerr=yerrors)
        graph_ax.set_xticks(x)
        graph_ax.set_xticklabels(x_tick_labels, rotation=45 if self.well_amount < 15 else 90, ha="right")
        graph_ax.set_box_aspect(1)
        fig.tight_layout()
        if output_folder:
            if alt_name:
                fig.savefig(os.path.join(output_folder, alt_name + ".jpg"))
            else:
                fig.savefig(os.path.join(output_folder, "barplot.jpg"))
            plt.close(fig)
        else:
            plt.show()

    def make_average_page(self, parameter, absolute=False):

        title = parameter + " average values"
        if absolute:
            title = "absolute " + title
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title)
        if not os.path.exists(os.path.join(output_folder, "over_time.jpg")):
            self.draw_average_over_time(parameter, output_folder=output_folder, absolute=absolute)
        self.pdf.image(os.path.join(output_folder, "over_time.jpg"), x=TWO_GRAPH_X[0], y=TWO_GRAPH_Y[0],
                       w=TWO_GRAPH_WIDTH)
        if not os.path.exists(os.path.join(output_folder, "barplot.jpg")):
            self.draw_average_barplot(parameter, output_folder=output_folder, absolute=absolute)
        self.pdf.image(os.path.join(output_folder, "barplot.jpg"), x=TWO_GRAPH_X[1], y=TWO_GRAPH_Y[1],
                       w=TWO_GRAPH_WIDTH)

    def draw_layer_graph(self, well, parameter, output_folder=None, intervals=15, absolute=False, groups=False,
                         fixed_distance=30, scaled=False):
        fig, graph_ax = plt.subplots()
        well_df = self.well_info[well]
        # --- Fix: skip wells missing the parameter ---
        if parameter not in well_df.columns:
            print(f"draw layer graph : Skipping {parameter} for well {well}: column missing.")
            return
        well_df = well_df.dropna(subset=[parameter])
        well_name = self.shortened_well_names[well]
        min_time, max_time = self._min_max("TimeIndex", max_val=self.dt, multiplier=self.dt, scaled=False)
        min_val, max_val = self._scratch_min_max(parameter, absolute=absolute, groups=groups,
                                                 fixed_distance=fixed_distance, scaled=scaled)
        scratch_info_per_time = self._scratch_info(well, intervals=intervals)
        x = well_df.TimeIndex.unique() * self.dt
        x.sort()
        y = np.array(range(intervals))
        normalized = colors.Normalize(vmin=min_val, vmax=max_val)
        graph_ax.set_title("%d%s - " % (self.pdf.page_no(), chr(self.graph_counter)) + well_name)
        self.graph_counter += 1
        if self.graph_counter == 123:
            self.graph_counter = 65
        graph_ax.set_xlabel("Time [min]")
        graph_ax.set_xbound(lower=0, upper=max(x))
        graph_ax.set_ybound(lower=0, upper=intervals)
        graph_ax.set_box_aspect(1)
        if groups:
            graph_ax.set_ylabel(r"Cells grouped by distance from scratch")
            graph_ax.set_yticks([0] + list(y+0.5) + [intervals])
            graph_ax.set_yticklabels(["Closest"] + ["Group %d" % (i+1) for i in y] + ["Farthest"])
        else:
            graph_ax.set_ylabel(r"Distance from scratch [$\mu$m]")
            graph_ax.set_yticks(list(y+0.5))
            graph_ax.set_yticklabels(["%d-%d" % (fixed_distance*i, fixed_distance*(i+1)) for i in range(intervals-1)] + [">%d" % (fixed_distance*(intervals-1))])
        total = 0
        for minute in x:
            scratch_top, scratch_bottom, top_interval, bottom_interval = scratch_info_per_time[int(minute / self.dt)]
            if not groups:
                top_interval = fixed_distance
                bottom_interval = fixed_distance
            time_df = well_df[well_df.TimeIndex == minute / self.dt]
            skipped_df = time_df[time_df.y_Pos < scratch_top]
            skipped_df = skipped_df[skipped_df.y_Pos > scratch_bottom]
            time_total = len(skipped_df)
            for multiplier in y:
                top_group = time_df[time_df.y_Pos >= scratch_top + (multiplier * top_interval)]
                if multiplier != y[-1]:
                    top_group = top_group[top_group.y_Pos < scratch_top + ((multiplier+1) * top_interval)]
                bottom_group = time_df[time_df.y_Pos <= scratch_bottom - (multiplier * bottom_interval)]
                if multiplier != y[-1]:
                    bottom_group = bottom_group[bottom_group.y_Pos > scratch_bottom - ((multiplier+1) * bottom_interval)]
                time_total += len(top_group) + len(bottom_group)
                top_group = top_group[parameter]
                bottom_group = bottom_group[parameter]
                if absolute:
                    top_group = np.absolute(top_group)
                    bottom_group = np.absolute(bottom_group)
                try:
                    avg_val = (sum(top_group) + sum(bottom_group)) / (len(top_group) + len(bottom_group))
                except ZeroDivisionError:
                    avg_val = 0
                rect = patches.Rectangle((minute, multiplier), width=self.dt, height=1, color=cm.jet(normalized(avg_val)))
                graph_ax.add_patch(rect)
            if time_total == len(time_df):
                total += time_total
            else:
                raise Exception("TIME TOTAL DIDNT ADD UP %d %d" % (time_total, len(time_df)))
        if total != len(well_df):
            raise Exception("TOTAL DIDNT ADD UP %d %d" % (total, len(well_df)))
        cb_label = parameter + UNIT_DICT[parameter]
        colorbar = fig.colorbar(cm.ScalarMappable(norm=normalized, cmap="jet"), ax=graph_ax)
        colorbar.set_label(cb_label)
        if scaled:
            ticks = [min_val + ((max_val - min_val) * i / 6) for i in range(7)]
            colorbar.set_ticks(ticks)
            ticks = [str(round(tick, 2)) for tick in ticks]
            ticks[0] = "<" + ticks[0]
            ticks[-1] = ">" + ticks[-1]
            colorbar.set_ticklabels(ticks)
        fig.tight_layout()
        if output_folder:
            if groups:
                fig.savefig(os.path.join(output_folder, well + "_groups.jpg"))
            else:
                fig.savefig(os.path.join(output_folder, well + ".jpg"))
            plt.close(fig)
        else:
            plt.show()

    def make_layers_page(self, parameter, absolute=False, scaled=False):
        title = parameter + " of layers"
        if absolute:
            title += " absolute"
        if scaled:
            title += " scaled"
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title + " by distance", table=True)
        for i in range(self.well_amount):
            well = self.wells[i]
            if not os.path.exists(os.path.join(output_folder, well + ".jpg")):
                self.draw_layer_graph(well, parameter, output_folder=output_folder, absolute=absolute, scaled=scaled)
            self.pdf.image(os.path.join(output_folder, well + ".jpg"),
                           x=self.graph_x_coords[i], y=self.graph_y_coords[i], w=self.graph_width)
        #self.new_page(title + " by groups")
        #for i in range(self.well_amount):
        #    well = self.wells[i]
        #    self.draw_layer_graph(well, parameter, output_folder=output_folder, absolute=absolute, groups=True, scaled=scaled)
        #    self.pdf.image(os.path.join(output_folder, well + "_groups.jpg"), x=self.graph_x_coords[i], y=self.graph_y_coords[i], w=self.graph_width)

    def draw_param_vs_param_graph(self, well, parameter1, parameter2, output_folder=None, absolute=False, average=False):
        fig, graph_ax = plt.subplots()
        if well == "overlayed":
            wells = self.wells
            graph_ax.set_title("%d%s - All wells overlayed" % (self.pdf.page_no(), chr(self.graph_counter)))
            self.graph_counter += 1
            if self.graph_counter == 123:
                self.graph_counter = 65
        else:
            wells = [well]
            well_name = self.shortened_well_names[well]
            graph_ax.set_title("%d%s - " % (self.pdf.page_no(), chr(self.graph_counter)) + well_name)
            self.graph_counter += 1
            if self.graph_counter == 123:
                self.graph_counter = 65
        min_time, max_time = self._min_max("TimeIndex", max_val=self.dt, multiplier=self.dt, scaled=False)
        min1, max1 = self._min_max(parameter1, absolute=absolute, average=average, scaled=False)
        min2, max2 = self._min_max(parameter2, absolute=absolute, average=average, scaled=False)
        xlabel = parameter1 + UNIT_DICT[parameter1]
        ylabel = parameter2 + UNIT_DICT[parameter2]
        cb_label = "Time [min]"
        if average:
            xlabel = "average " + xlabel
            ylabel = "average " + ylabel
        if absolute:
            xlabel = "absolute " + xlabel
            ylabel = "absolute " + ylabel
        graph_ax.set_xlabel(xlabel)
        graph_ax.set_ylabel(ylabel)
        if UNIT_DICT[parameter1] == UNIT_DICT[parameter2]:
            graph_ax.set_xbound(lower=min([min1, min2]), upper=max([max1, max2]))
            graph_ax.set_ybound(lower=min([min1, min2]), upper=max([max1, max2]))
            graph_ax.set_aspect(1)
            graph_ax.set_autoscale_on(False)
        else:
            graph_ax.set_xbound(lower=min1, upper=max1)
            graph_ax.set_ybound(lower=min2, upper=max2)
        normalized = colors.Normalize(vmin=min_time, vmax=max_time)
        # end of setup
        for i in range(len(wells)):
            well_df = self.well_info[wells[i]]
            if (parameter1 not in well_df.columns) or (parameter2 not in well_df.columns):
                print(f"param vs param : Skipping {parameter1}vs{parameter2} for well {well}: a column is missing.")
                continue
            well_df = well_df.dropna(subset=[parameter1, parameter2])
            marker = MARKERS[i]
            if average:
                c = well_df.TimeIndex.unique() * self.dt
                c.sort()
                vals1 = [well_df[well_df.TimeIndex == time_index / self.dt][parameter1] for time_index in c]
                vals2 = [well_df[well_df.TimeIndex == time_index / self.dt][parameter2] for time_index in c]
                if absolute:
                    vals1 = [np.absolute(val) for val in vals1]
                    vals2 = [np.absolute(val) for val in vals2]
                x = [np.average(val) for val in vals1]
                y = [np.average(val) for val in vals2]
                size = 20
            else:
                size = 10
                x = well_df[parameter1]
                y = well_df[parameter2]
                c = well_df.TimeIndex * self.dt
            if absolute:
                x = np.absolute(x)
                y = np.absolute(y)
            sc = graph_ax.scatter(x, y, c=c, s=size, vmin=min_time, vmax=max_time, marker=marker, cmap="jet", label=self.shortened_well_names[wells[i]])
        colorbar = fig.colorbar(cm.ScalarMappable(norm=normalized, cmap="jet"), ax=graph_ax)
        colorbar.set_label(cb_label)
        if well == "overlayed":
            graph_ax.legend()
        graph_ax.set_box_aspect(1)
        fig.tight_layout()
        if output_folder:
            fig.savefig(os.path.join(output_folder, well + ".jpg"))
            plt.close(fig)
        else:
            plt.show()

    def make_param_vs_param_page(self, parameter1, parameter2, absolute=False, average=False):
        title = parameter1 + " vs " + parameter2
        if absolute:
            title += " absolute"
        if average:
            title += " averages"
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title, table=True)
        for i in range(self.well_amount):
            well = self.wells[i]
            image_path = os.path.join(output_folder, well + ".jpg")
            # --- Only plot if both parameters exist in this well ---
            well_df = self.well_info[well]
            if parameter1 not in well_df.columns or parameter2 not in well_df.columns:
                print(f"Skipping {parameter1} vs {parameter2} for well {well}: column(s) missing.")
                continue
            if not os.path.exists(image_path):
                self.draw_param_vs_param_graph(well, parameter1, parameter2, output_folder=output_folder,
                                               absolute=absolute, average=average)
            if os.path.exists(image_path):
                self.pdf.image(os.path.join(output_folder, well + ".jpg"), x=self.graph_x_coords[i],
                               y=self.graph_y_coords[i], w=self.graph_width)
            else:
                print(f"Skipping PDF image for {well} and {parameter1} vs {parameter2}: image not created.")

        if average:
            self.new_page(title + " overlayed")
            overlay_path = os.path.join(output_folder, "overlayed.jpg")
            if not os.path.exists(overlay_path):
                self.draw_param_vs_param_graph("overlayed", parameter1, parameter2, output_folder=output_folder,
                                               absolute=absolute, average=average)
            if os.path.exists(overlay_path):
                self.pdf.image(overlay_path, x=SINGLE_GRAPH_X, y=SINGLE_GRAPH_Y, w=SINGLE_GRAPH_WIDTH)

    def draw_wave_percentage(self, output_folder=None):
        fig, graph_ax = plt.subplots()
        graph_ax.set_title("%d%s - Percent of cells with wave properties" % (self.pdf.page_no(), chr(self.graph_counter)))
        self.graph_counter += 1
        if self.graph_counter == 123:
            self.graph_counter = 65
        graph_ax.set_xlabel("Treatment")
        graph_ax.set_ylabel("Percent of cells")
        graph_ax.set_yticks(range(0, 110, 10))
        graph_ax.set_yticklabels(range(0, 110, 10))
        full_wave_cells = np.array([0.0] * self.well_amount)
        left_sided_wave_cells = np.array([0.0] * self.well_amount)
        right_sided_wave_cells = np.array([0.0] * self.well_amount)
        no_wave_cells = np.array([0.0] * self.well_amount)
        total_cells = np.array([0.0] * self.well_amount)
        for i in range(self.well_amount):
            well_df = self.well_info[self.wells[i]]
            well_df = well_df.dropna(subset=WAVE_PARAMETERS)
            all_cells = well_df.Parent.unique()
            total_cells[i] = len(all_cells)
            for cell in all_cells:
                data = well_df[well_df.Parent == cell].iloc[0]
                if data.Velocity_Full_Width_Half_Maximum == 0:
                    no_wave_cells[i] += 1
                elif data.Velocity_Ending_Value == 0:
                    left_sided_wave_cells[i] += 1
                elif data.Velocity_Starting_Value == 0:
                    right_sided_wave_cells[i] += 1
                else:
                    full_wave_cells[i] += 1
            if full_wave_cells[i] + left_sided_wave_cells[i] + right_sided_wave_cells[i] + no_wave_cells[i] != total_cells[i]:
                print(full_wave_cells[i], left_sided_wave_cells[i], right_sided_wave_cells[i], no_wave_cells[i], total_cells[i])
                print(full_wave_cells, left_sided_wave_cells[i], right_sided_wave_cells[i], no_wave_cells, total_cells)
                raise ValueError("Unexpected issue with wave parameters - something didn't add up.")
        x = range(self.well_amount)
        rects1 = graph_ax.bar(x, 100 * full_wave_cells / total_cells, color="tab:blue", label="Full wave found")
        rects2 = graph_ax.bar(x, 100 * left_sided_wave_cells / total_cells, color="tab:orange", label="Left sided wave found", bottom=[rect.get_height() + rect.get_y() for rect in rects1])
        rects3 = graph_ax.bar(x, 100 * right_sided_wave_cells / total_cells, color="tab:red", label="Right sided wave found", bottom=[rect.get_height() + rect.get_y() for rect in rects2])
        rects4 = graph_ax.bar(x, 100 * no_wave_cells / total_cells, color="tab:green", label="No wave found", bottom=[rect.get_height() + rect.get_y() for rect in rects3])
        for i in range(self.well_amount):
            graph_ax.annotate("%d\n(%.2f" % (full_wave_cells[i], 100 * full_wave_cells[i] / total_cells[i]) + "%)",
                        xy=(rects1[i].get_x() + (rects1[i].get_width() / 2), rects1[i].get_y() + (rects1[i].get_height() / 2)), ha="center", va="center", rotation=90)
            graph_ax.annotate("%d\n(%.2f" % (left_sided_wave_cells[i], 100 * left_sided_wave_cells[i] / total_cells[i]) + "%)",
                        xy=(rects2[i].get_x() + (rects2[i].get_width() / 2), rects2[i].get_y() + (rects2[i].get_height() / 2)), ha="center", va="center", rotation=90)
            graph_ax.annotate("%d\n(%.2f" % (right_sided_wave_cells[i], 100 * right_sided_wave_cells[i] / total_cells[i]) + "%)",
                        xy=(rects3[i].get_x() + (rects3[i].get_width() / 2), rects3[i].get_y() + (rects3[i].get_height() / 2)), ha="center", va="center", rotation=90)
            graph_ax.annotate("%d\n(%.2f" % (no_wave_cells[i], 100 * no_wave_cells[i] / total_cells[i]) + "%)",
                        xy=(rects4[i].get_x() + (rects4[i].get_width() / 2), rects4[i].get_y() + (rects4[i].get_height() / 2)), ha="center", va="center", rotation=90)
        x_tick_labels = [self.shortened_well_names[well] for well in self.wells]
        graph_ax.set_xticks(x)
        graph_ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
        graph_ax.set_box_aspect(1)
        graph_ax.legend(bbox_to_anchor=(1,0.5), loc="center left")
        fig.tight_layout()
        if output_folder:
            fig.savefig(os.path.join(output_folder, "wave_percentage.jpg"))
            plt.close(fig)
        else:
            plt.show()

    def make_wave_pages(self):
        title = "Wave properties - all cells where a wave was found"
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title)
        if not os.path.exists(os.path.join(output_folder, "wave_percentage.jpg")):
            self.draw_wave_percentage(output_folder)
        self.pdf.image(os.path.join(output_folder, "wave_percentage.jpg"), x=EIGHT_GRAPH_X[0], y=EIGHT_GRAPH_Y[0], w=EIGHT_GRAPH_WIDTH)
        for i in range(7):
            parameter = WAVE_PARAMETERS[i]
            if not os.path.exists(os.path.join(output_folder, parameter + ".jpg")):
                self.draw_average_barplot(parameter, output_folder=output_folder, wave="all", alt_name=parameter, per_cell=True)
            self.pdf.image(os.path.join(output_folder, parameter + ".jpg"), x=EIGHT_GRAPH_X[i+1], y=EIGHT_GRAPH_Y[i+1], w=EIGHT_GRAPH_WIDTH)
        title = "Wave properties - only double sided waves"
        self.new_page(title)
        self.pdf.image(os.path.join(output_folder, "wave_percentage.jpg"), x=EIGHT_GRAPH_X[0], y=EIGHT_GRAPH_Y[0], w=EIGHT_GRAPH_WIDTH)
        for i in range(7):
            parameter = WAVE_PARAMETERS[i]
            alt_name = parameter + "_double"
            if not os.path.exists(os.path.join(output_folder, alt_name + ".jpg")):
                self.draw_average_barplot(parameter, output_folder=output_folder, wave="double", alt_name=alt_name, per_cell=True)
            self.pdf.image(os.path.join(output_folder, alt_name + ".jpg"), x=EIGHT_GRAPH_X[i+1], y=EIGHT_GRAPH_Y[i+1], w=EIGHT_GRAPH_WIDTH)
        title = "Wave properties - only left sided waves"
        self.new_page(title)
        self.pdf.image(os.path.join(output_folder, "wave_percentage.jpg"), x=SIX_GRAPH_X[0], y=SIX_GRAPH_Y[0], w=FOUR_GRAPH_WIDTH)
        for i in range(5):
            parameter = LEFT_WAVE_PARAMETERS[i]
            alt_name = parameter + "_left"
            if not os.path.exists(os.path.join(output_folder, alt_name + ".jpg")):
                self.draw_average_barplot(parameter, output_folder=output_folder, wave="left", alt_name=alt_name, per_cell=True)
            self.pdf.image(os.path.join(output_folder, alt_name + ".jpg"), x=SIX_GRAPH_X[i+1], y=SIX_GRAPH_Y[i+1], w=FOUR_GRAPH_WIDTH)
        title = "Wave properties - only right sided waves"
        self.new_page(title)
        self.pdf.image(os.path.join(output_folder, "wave_percentage.jpg"), x=SIX_GRAPH_X[0], y=SIX_GRAPH_Y[0], w=FOUR_GRAPH_WIDTH)
        for i in range(5):
            parameter = RIGHT_WAVE_PARAMETERS[i]
            alt_name = parameter + "_right"
            if not os.path.exists(os.path.join(output_folder, alt_name + ".jpg")):
                self.draw_average_barplot(parameter, output_folder=output_folder, wave="right", alt_name=alt_name, per_cell=True)
            self.pdf.image(os.path.join(output_folder, alt_name + ".jpg"), x=SIX_GRAPH_X[i+1], y=SIX_GRAPH_Y[i+1], w=FOUR_GRAPH_WIDTH)

    def make_displacement_page(self):
        title = "Displacement information"
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title)
        if "Track_Displacement_Z" in self.parameters:
            x_coords = FIVE_GRAPH_X
            y_coords = FIVE_GRAPH_Y
            displacement_params = DISPLACEMENT_PARAMS
        else:
            x_coords = FOUR_GRAPH_X
            y_coords = FOUR_GRAPH_Y
            displacement_params = DISPLACEMENT_PARAMS[:-1]
        for i in range(len(displacement_params)):
            parameter = displacement_params[i]
            if not os.path.exists(os.path.join(output_folder, parameter + ".jpg")):
                self.draw_average_barplot(parameter, output_folder=output_folder, alt_name=parameter, per_cell=True)
            self.pdf.image(os.path.join(output_folder, parameter + ".jpg"), x=x_coords[i], y=y_coords[i], w=FOUR_GRAPH_WIDTH)

    def make_motility_page(self):
        title = "Cell motility properties"
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title)
        for i in range(4):
            parameter = MOTILITY_PARAMS[i]
            if not os.path.exists(os.path.join(output_folder, parameter + ".jpg")):
                self.draw_average_barplot(parameter, output_folder=output_folder, alt_name=parameter, per_cell=True)
            self.pdf.image(os.path.join(output_folder, parameter + ".jpg"), x=FOUR_GRAPH_X[i], y=FOUR_GRAPH_Y[i], w=FOUR_GRAPH_WIDTH)

    def draw_msd(self, well, output_folder=None):
        fig, graph_ax = plt.subplots()
        well_name = self.shortened_well_names[well]
        graph_ax.set_title("%d%s - " % (self.pdf.page_no(), chr(self.graph_counter)) + well_name)
        self.graph_counter += 1
        if self.graph_counter == 123:
            self.graph_counter = 65
        graph_ax.set_xlabel("Tau [%d minutes each]" % self.dt)
        graph_ax.set_ylabel(r"MSD [$\mu$m]")
        graph_ax.set_ylim(top=self.max_msd)
        graph_ax.set_xlim(right=self.max_tau)
        msd_dicts = self.msd_info[well]["msd_dicts"]
        for msd_dict in msd_dicts:
            graph_ax.plot(list(msd_dict.keys()), list(msd_dict.values()))
        graph_ax.set_box_aspect(1)
        fig.tight_layout()
        if output_folder:
            fig.savefig(os.path.join(output_folder, well + ".jpg"))
            plt.close(fig)
        else:
            plt.show()

    def draw_average_msd(self, output_folder=None):
        fig, graph_ax = plt.subplots()
        graph_ax.set_title("%d%s - " % (self.pdf.page_no(), chr(self.graph_counter)) + "Average MSD per well")
        self.graph_counter += 1
        if self.graph_counter == 123:
            self.graph_counter = 65
        graph_ax.set_xlabel("Tau [%d minutes each]" % self.dt)
        graph_ax.set_ylabel(r"Average MSD [$\mu$m]")
        for well in self.wells:
            well_name = self.shortened_well_names[well]
            average_dict = self.msd_info[well]["average"]
            graph_ax.plot(average_dict.keys(), average_dict.values(), label=well_name)
        graph_ax.set_box_aspect(1)
        graph_ax.legend()
        fig.tight_layout()
        if output_folder:
            fig.savefig(os.path.join(output_folder, "average_msds.jpg"))
            plt.close(fig)
        else:
            plt.show()

    def make_MSD_pages(self):
        self._get_msd_info()
        title = "MSD properties"
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title)
        for i in range(6):
            parameter = MSD_PARAMS[i]
            if not os.path.exists(os.path.join(output_folder, parameter + ".jpg")):
                self.draw_average_barplot(parameter, output_folder=output_folder, alt_name=parameter, per_cell=True)
            self.pdf.image(os.path.join(output_folder, parameter + ".jpg"), x=SIX_GRAPH_X[i], y=SIX_GRAPH_Y[i], w=FOUR_GRAPH_WIDTH)
        self.new_page("MSD graphs per well", table=True)
        for i in range(self.well_amount):
            well = self.wells[i]
            if not os.path.exists(os.path.join(output_folder, well + ".jpg")):
                self.draw_msd(well, output_folder=output_folder)
            self.pdf.image(os.path.join(output_folder, well + ".jpg"), x=self.graph_x_coords[i], y=self.graph_y_coords[i], w=self.graph_width)
        self.new_page("Average MSD")
        if not os.path.exists(os.path.join(output_folder, "average_msds.jpg")):
            self.draw_average_msd(output_folder=output_folder)
        self.pdf.image(os.path.join(output_folder, "average_msds.jpg"),
                       x=SINGLE_GRAPH_X, y=SINGLE_GRAPH_Y, w=SINGLE_GRAPH_WIDTH)

    def draw_cluster_analysis(self, output_folder=None):
        avg_df = pd.DataFrame(index=self.wells, columns=self.parameters)
        avg_df.drop(columns=CLUSTER_DROP_FIELDS, inplace=True, errors="ignore")
        for well in self.wells:
            well_df = self.well_info[well]
            for parameter in CLUSTER_ID_FIELDS:
                if parameter in self.parameters and parameter in well_df.columns:
                    parameter_array = well_df[parameter].dropna()
                    avg_df.loc[well, parameter] = np.average(parameter_array)
            cells = well_df.Parent.unique()
            well_df = well_df.set_index("Parent")
            for parameter in CLUSTER_CELL_FIELDS + CLUSTER_WAVE_FIELDS:
                if parameter in self.parameters and parameter in well_df.columns:
                    values = np.array([0.0] * len(cells))
                    for i in range(len(cells)):
                        try:
                            values[i] = well_df.loc[cells[i], parameter].iloc[0]
                        except AttributeError:
                            values[i] = well_df.loc[cells[i], parameter]
                    if parameter in CLUSTER_WAVE_FIELDS:
                        values = values[values.nonzero()]
                    values = pd.Index(values).dropna()
                    avg_df.loc[well, parameter] = np.average(values)

        # Drop columns and rows with any non-finite values
        avg_df = avg_df.apply(pd.to_numeric, errors='coerce')
        avg_df = avg_df.replace([np.inf, -np.inf], np.nan)
        avg_df = avg_df.dropna(axis=1, how='any')  # Drop columns with any NaN
        avg_df = avg_df.dropna(axis=0, how='any')  # Drop rows with any NaN
        if avg_df.shape[0] < 2 or avg_df.shape[1] < 2:
            print("Not enough data for clustering after dropping NaNs.")
            return
        scaler = StandardScaler(with_std=True)
        avg_df = pd.DataFrame(scaler.fit_transform(avg_df), columns=avg_df.columns, index=[self.shortened_well_names[w] for w in avg_df.index])
        linkaged_pca = linkage(avg_df, "ward")
        s = sns.clustermap(data=avg_df, row_linkage=linkaged_pca, cmap=sns.color_palette("coolwarm", n_colors=256),
                           vmin=-2, vmax=2, figsize=(30, 15),
                           cbar_kws = dict(use_gridspec=False))
        #s.ax_heatmap.set_xlabel("Parameters", fontsize=25, fontweight='bold')
        #s.ax_heatmap.set_ylabel("Well", fontsize=25, fontweight='bold')
        #s.cax.set_yticklabels(s.cax.get_yticklabels());
        #pos = s.ax_heatmap.get_position();
        #cbar = s.cax
        #cbar.set_position([0.02, pos.bounds[1], 0.02, pos.bounds[3]]);
        s.ax_heatmap.set_xticklabels(s.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
        if output_folder:
            plt.savefig(os.path.join(output_folder, "clustermap.jpg"))
            plt.close()
        else:
            plt.suptitle("%d - Cluster analysis" % self.pdf.page_no(), fontweight='bold', fontsize=30)
            plt.show()

    def make_cluster_page(self):
        title = "Cluster Analysis"
        output_folder = os.path.join(self.output_path, title)
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        self.new_page(title)
        if not os.path.exists(os.path.join(output_folder, "clustermap.jpg")):
            self.draw_cluster_analysis(output_folder=output_folder)
        self.pdf.image(os.path.join(output_folder, "clustermap.jpg"), x=0, y=30, w=WORKING_WIDTH, h=WORKING_HEIGHT)

    def load_dicts(self, output_path):
        try:
            for dict_name in DICTS:
                dict_file = os.path.join(output_path, dict_name + ".temp")
                with open(dict_file, "r") as f:
                    temp_dict = json.load(f)
                for well in self.wells:
                    try:
                        self.__dict__[dict_name][well] = temp_dict[well]
                    except KeyError:
                        pass
        except FileNotFoundError:
            return

    def dump_dicts(self, output_path):
        for dict_name in DICTS:
            dict_file = os.path.join(output_path, dict_name + "_" + self.exp_name + "_" + self.exp_sub_name + ".temp")
            try:
                with open(dict_file, "r") as f:
                    temp_dict = json.load(f)
                temp_dict.update(self.__dict__[dict_name])
                with open(dict_file, "w") as f:
                    json.dump(temp_dict, f, default=lambda x: x.__dict__)
            except FileNotFoundError:
                with open(dict_file, "w") as f:
                    json.dump(self.__dict__[dict_name], f, default=lambda x: float(x) if type(x) == np.float64 else x.__dict__)

    def create_output(self, output_path, param_graphs=PARAM_GRAPHS, param_pair_graphs=PARAM_PAIR_GRAPHS):
        pr = cProfile.Profile()
        pr.enable()
        print(self.exp_name + " graphs:")
        #self.load_dicts(output_path)
        self.create_pdf_file()
        already_run = self.make_output_folders(output_path)
        if already_run == "ALL":
            print("Already ran all this.")
            return "Done"
        print("\tMaking first pages...")
        self.make_first_page()
        self.make_second_page()
        self.make_third_page()
        if self.scratch:
            self.make_scratch_pages()
        self.make_sunplot_page()
        for parameter in param_graphs.keys():
            if parameter in self.parameters:
                print("\tMaking %s graphs..." % parameter)
                if self.dimensions == 2:
                    if "y_pos_time" in param_graphs[parameter]:
                        self.make_y_pos_time_page(parameter)
                    if "log_y_pos_time" in param_graphs[parameter]:
                        self.make_y_pos_time_page(parameter, log=True)
                    if "absolute_y_pos_time" in param_graphs[parameter]:
                        self.make_y_pos_time_page(parameter, absolute=True)
                    if "y_pos_time_scaled" in param_graphs[parameter] and "absolute_y_pos_time" in param_graphs[parameter]:
                        self.make_y_pos_time_page(parameter, absolute=True, scaled=True)
                if "average" in param_graphs[parameter]:
                    self.make_average_page(parameter)
                if "absolute_average" in param_graphs[parameter]:
                    self.make_average_page(parameter, absolute=True)
                if self.scratch == True:
                    if "layers" in param_graphs[parameter]:
                        self.make_layers_page(parameter)
                    if "absolute_layers" in param_graphs[parameter]:
                        self.make_layers_page(parameter, absolute=True)
                    if "layers_scaled" in param_graphs[parameter]:
                        self.make_layers_page(parameter, scaled=True)
                    if "absolute_layers_scaled" in param_graphs[parameter]:
                        self.make_layers_page(parameter, absolute=True, scaled=True)
        for parameter_pair in param_pair_graphs.keys():
            parameter1, parameter2 = parameter_pair
            if parameter1 in self.parameters and parameter2 in self.parameters:
                print("\tMaking %s vs %s graphs..." % (parameter1, parameter2))
                self.make_param_vs_param_page(parameter1, parameter2)
                if "absolute" in param_pair_graphs[parameter_pair]:
                    absolute = True
                    self.make_param_vs_param_page(parameter1, parameter2, absolute=True)
                else:
                    absolute = False
                if "average" in param_pair_graphs[parameter_pair]:
                    self.make_param_vs_param_page(parameter1, parameter2, average=True, absolute=absolute)
        print("\tMaking last pages...")
        self.make_wave_pages()
        self.make_displacement_page()
        self.make_motility_page()
        print("\tMaking MSD page...")
        self.make_MSD_pages()
        print("\tMaking cluster page...")
        self.make_cluster_page()
        print("\tFinal steps...")
        self.pdf.output(os.path.join(output_path, self.exp_name + "_report.pdf"))
        #self.dump_dicts(output_path)

        average_df.to_excel(output_path + "\\average_parameter_values.xlsx")

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        log_path = os.path.join(r'\\metlab25\d\pybatch\logs\ori_logs', time.strftime("%Y%m%d%H%M%S_cprofile.log", time.localtime()))
        with open(log_path, "w") as f:
            f.write(s.getvalue())

def parse_xls2(xls2_path, scratch, dt):
    xls2 = xlrd.open_workbook(xls2_path).sheet_by_index(0)
    experiments = []
    exp_names = xls2.col_values(1, start_rowx=4, end_rowx=19)
    path_list = xls2.col_values(1, start_rowx=32, end_rowx=35)
    protocol_file, incucyte_files, imaris_xls_files = path_list
    for i in range(len(exp_names)):
        exp_name = exp_names[i]
        if exp_name not in ["NNN0", ""]:
            if exp_names.count(exp_name) > 1:
                exp_name += "_%d" % i
            row_values = xls2.row_values(4+i, start_colx=2, end_colx=27)
            exp_sub_name = row_values[0]
            wells = [well for well in row_values[1:] if well != "NNN0"]
            experiments.append(Batch_Experiment(exp_name, exp_sub_name, wells, scratch, protocol_file, incucyte_files, imaris_xls_files, dt=dt))
    return experiments


def load_from_pyobject(summary_table, exp_name, exp_sub_name, wells, scratch=False, dt=45):
    experiment = Batch_Experiment(exp_name, exp_sub_name, wells, scratch, dt=dt)
    experiment.load_wells_from_summary_table(summary_table)
    return experiment


def load_info_old(xls2_path=r"\\metlab24\d\orimosko\yossi_chemo_exps\EXP120\xls2\2_YLEXP120_ExperimentLayoutV1.xlsm",
                                            summary_table_path=r"E:\orimosko\pybatch_output\YL120_summary_table.xlsx",
                                            scratch=True,
                                            dt=45):
    """
    This function loads info from experiments with the old xls2 and summary table formats
    """
    experiments = parse_xls2(xls2_path, scratch, dt)
    summary_table = pd.read_excel(summary_table_path)
    summary_table.drop(columns="Unnamed: 0", inplace=True)
    for experiment in experiments:
        experiment.xls2_path = xls2_path
        experiment.load_wells_from_summary_table(summary_table)
    return experiments


def main(xls2_path=None, summary_table_path=None):
    if xls2_path and summary_table_path:
        experiments = load_info_old(xls2_path, summary_table_path)


########### MAYBE TAKE FUNCTIONS OUT OF CLASS #############

#exps = load_info_old()
#for exp in exps:
#    exp.create_output(r"E:\orimosko\pybatch_output\exp120\graphs")
#exp = load_from_pyobject(summary_table, "The effect of Siltuximab on group cell motility", 'G.0',
#                         ['YL120CHR1B02SK00CON1NNN0NNN0NNN0NNN0WH00NNN0', 'YL120CHR1B04SK00SIL5NNN0NNN0NNN0NNN0WH00NNN0', 'YL120CHR1B03SK00SIL3NNN0NNN0NNN0NNN0WH00NNN0'])




