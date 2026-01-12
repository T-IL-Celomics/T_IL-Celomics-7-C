import pandas as pd
import os
import xlrd
from shutil import copyfile
import re
from collections import OrderedDict
import math
import time
import json
import pickle
from pybatch_objects import *
from batch_calculations import *
#from SessionState import *
import streamlit as st
from streamlit.hashing import _CodeHasher
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server
from copy import deepcopy
import win32com.client as win32
import io
import pstats

###################################################################################################

# Streamlit things

class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
            "loaded": False
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            self._state["is_rerun"] = True
            self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


class Pybatch_State(object):

    def __init__(self, stage=1, experiments=[None], skip_rename=False, skip_summary_table=False, run=False, wherewasi=0,
                 drop_infos={}, create_one_pickle=False, pickle_path=""):
        self.stage = stage
        self.experiments = experiments
        self.skip_rename = skip_rename
        self.skip_summary_table = skip_summary_table
        self.run = run
        self.wherewasi = wherewasi
        self.drop_infos = drop_infos
        self.create_one_pickle = create_one_pickle
        self.pickle_path = pickle_path

    def load_dict(self, state_dict):
        self.__dict__ = state_dict

    def copy(self):
        return Pybatch_State(self.stage, self.experiments, self.skip_rename, self.skip_summary_table, self.run,
                             self.wherewasi, self.drop_infos, self.create_one_pickle, self.pickle_path)

    def __eq__(self, other):
        if type(other) == type(self):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


@st.cache
def get_time_and_session_id():
    return time.localtime(), get_report_ctx().session_id


def read_state(log_file):
    with open(log_file, "r") as f:
        state_dict = json.load(f)
    state = Pybatch_State()
    state.load_dict(state_dict)
    return state


def write_state(log_file, state):
    with open(log_file, "w") as f:
        json.dump(state, f, default=lambda x: x.__dict__)



###################################################################################################


FIRST_PARENT = 10 ** 9

DESIRED_FIELDS = ["Area",
                  "Acceleration",
                  "Acceleration_OLD",
                  "Acceleration_X",
                  "Acceleration_Y",
                  "Acceleration_Z",
                  "Coll",
                  "Coll_CUBE",
                  "Confinement_Ratio",
                  "Directional_Change",
                  "Directional_Change_OLD",
                  "Directional_Change_X",
                  "Directional_Change_Y",
                  "Directional_Change_Z",
                  "Overall_Displacement",
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
                  "IntensityCenterCh3",
                  "IntensityMaxCh1",
                  "IntensityMaxCh2",
                  "IntensityMaxCh3",
                  "IntensityMeanCh1",
                  "IntensityMeanCh2",
                  "IntensityMeanCh3",
                  "IntensityMedianCh1",
                  "IntensityMedianCh2",
                  "IntensityMedianCh3",
                  "IntensitySumCh1",
                  "IntensitySumCh2",
                  "IntensitySumCh3",
                  "Instantaneous_Angle",
                  "Instantaneous_Angle_OLD",
                  "Instantaneous_Angle_X",
                  "Instantaneous_Angle_Y",
                  "Instantaneous_Angle_Z",
                  "Instantaneous_Speed",
                  "Instantaneous_Speed_OLD",
                  "Mean_Speed",
                  "Linearity_of_Forward_Progression",
                  "Mean_Curvilinear_Speed",
                  "Mean_Straight_Line_Speed",
                  "Current_MSD_1",
                  "Final_MSD_1",
                  "MSD_Linearity_R2_Score",
                  "MSD_Brownian_Motion_BIC_Score",
                  "MSD_Brownian_D",
                  "MSD_Directed_Motion_BIC_Score",
                  "MSD_Directed_D",
                  "MSD_Directed_v2",
                  "Sphericity",
                  "Total_Track_Displacement",
                  "Track_Displacement_Length",
                  "Track_Displacement_X",
                  "Track_Displacement_Y",
                  "Track_Displacement_Z",
                  "Velocity",
                  "Velocity_X",
                  "Velocity_Y",
                  "Velocity_Z",
                  "Eccentricity",
                  "Eccentricity_A",
                  "Eccentricity_B",
                  "Eccentricity_C",
                  "Min_Distance",
                  "Velocity_Full_Width_Half_Maximum",
                  "Velocity_Time_of_Maximum_Height",
                  "Velocity_Maximum_Height",
                  "Velocity_Ending_Value",
                  "Velocity_Ending_Time",
                  "Velocity_Starting_Value",
                  "Velocity_Starting_Time",
                  "TimeIndex",
                  "x_Pos",
                  "y_Pos",
                  "z_Pos",
                  "Parent",
                  "dt",
                  "ID",
                  "Experiment"]

GRAPH_OPTIONS = ["y_pos_time", "absolute_y_pos_time", "average", "absolute_average", "layers",
                 "layers_scaled", "absolute_layers", "absolute_layers_scaled"]
PAIR_GRAPH_OPTIONS = ["average", "absolute"]

OLD_FIELDS = ["Area", "Acceleration", "Coll", "Confinement_Ratio", "Directional_Change", "Displacement", "Displacement2", "Ellip_Ax_B_X", "Ellip_Ax_B_Y",
              "Ellip_Ax_C_X", "Ellip_Ax_C_Y", "EllipsoidAxisLengthB", "EllipsoidAxisLengthC", "Ellipticity_oblate", "Ellipticity_prolate", "Instantaneous_Angle",
              "Instantaneous_Speed", "Linearity_of_Forward_Progression", "Mean_Curvilinear_Speed", "Mean_Straight_Line_Speed", "MSD", "Sphericity",
              "Track_Displacement_Length", "Track_Displacement_X", "Track_Displacement_Y", "Velocity", "Velocity_X", "Velocity_Y", "Eccentricity",
              "Velocity_Full_Width_Half_Maximum", "Velocity_Time_of_Maximum_Height", "Velocity_Maximum_Height", "Velocity_Ending_Value", "Velocity_Ending_Time",
              "Velocity_Starting_Value", "Velocity_Starting_Time", "TimeIndex", "x_Pos", "y_Pos", "Parent", "dt", "ID", "Experiment"]

LOG_PATH = r"\\metlab24\d\jeries\pybatch\logs"

PLATE_LETTERS = ["B", "C", "D", "E", "F", "G"]
PLATE_NUMBERS = range(2, 12)

####### IMARIS PARSING #######

def check_imaris_version(imaris_path, imaris_version=None):
    try:
        test_file = pd.read_excel(imaris_path, sheet_name="Overall", skipfooter=10000)
    except xlrd.compdoc.CompDocError:
        excel_app = win32.Dispatch('Excel.Application')
        wb = excel_app.Workbooks.open(imaris_path)
        wb.Save()
        excel_app.quit()
        test_file = pd.read_excel(imaris_path, sheet_name="Overall", skipfooter=10000)
    if "Overall" in test_file.columns:
        apparent_imaris_version = 8
    else:
        apparent_imaris_version = 7
    if imaris_version == None:
        return apparent_imaris_version
    elif apparent_imaris_version != imaris_version:
        raise ValueError("You said (or based on previous folders) imaris was version %d, but it seems to be version %d.\nPlease start over." % (imaris_version, apparent_imaris_version))
    return imaris_version


def check_dimensions(imaris_df, dimensions=None):
    z_positions = imaris_df["Position"]["Position Z"]
    if len(set([round(z, 2) for z in z_positions])) < 6:
        apparent_dimensions = 2
    else:
        apparent_dimensions = 3
    if dimensions == None:
        return apparent_dimensions
    elif apparent_dimensions != dimensions:
        raise ValueError("You said (or based on previous folders) experiment has %d dimensions, but it seems to have %d.\nPlease start over." % (dimensions, apparent_dimensions))
    return dimensions

#check function
def check_dt(imaris_df, dt=None):
    FIRST_PARENT = imaris_df["Acceleration"]["Parent"].iloc[0]
    first_parent_duration = round(imaris_df["TrackDuration"].loc[FIRST_PARENT][0])
    if imaris_df["TrackDuration"].loc[FIRST_PARENT][1] == "m":
        first_parent_duration *= 60
    elif imaris_df["TrackDuration"].loc[FIRST_PARENT][1] == "h":
        first_parent_duration *= 3600
    first_parent_time_indexes = imaris_df["TimeIndex"][imaris_df["TimeIndex"]["Parent"] == FIRST_PARENT]
    first_parent_time_indexes = list(first_parent_time_indexes[first_parent_time_indexes.columns[0]])
    
    apparent_dt = round(
        (first_parent_duration / (max(first_parent_time_indexes) - min(first_parent_time_indexes))) / 60, 2)
    if dt == None:
        return apparent_dt
    elif apparent_dt != dt:
        raise ValueError(
            "You said (or based on previous folders) dt was %d, but it seems to be %d.\nPlease start over." % (
            dt, apparent_dt))
    return dt


def get_imaris_df(imaris_path, imaris_version):
    try:
        if imaris_version == 7:
            imaris_df = OrderedDict(pd.read_excel(imaris_path, sheet_name=None))
        elif imaris_version == 8:
            imaris_df = OrderedDict(pd.read_excel(imaris_path, sheet_name=None, header=1))
            for k in imaris_df.keys():
                imaris_df[k].rename(columns={"TrackID": "Parent"}, inplace=True)
    except xlrd.compdoc.CompDocError:
        excel_app = win32.Dispatch('Excel.Application')
        wb = excel_app.Workbooks.open(imaris_path)
        wb.Save()
        excel_app.quit()
        if imaris_version == 7:
            imaris_df = OrderedDict(pd.read_excel(imaris_path, sheet_name=None))
        elif imaris_version == 8:
            imaris_df = OrderedDict(pd.read_excel(imaris_path, sheet_name=None, header=1))
            for k in imaris_df.keys():
                imaris_df[k].rename(columns={"TrackID": "Parent"}, inplace=True)


    #Renames columns to match current MATLAB convention:
    for _ in range(len(imaris_df)):
        old_key, value = imaris_df.popitem(last=False)
        if old_key == "Ellipticity (oblate)":
            imaris_df["Ellipticity_oblate"] = value
        elif old_key == "Ellipticity (prolate)":
            imaris_df["Ellipticity_prolate"] = value
        else:
            imaris_df[old_key.replace(" ","").replace("=","").replace("^","").replace("-","_")] = value
    for key in imaris_df.keys():
        try:
            imaris_df[key].set_index("ID", inplace=True)
        except KeyError:
            pass
    return imaris_df


def get_all_parents(imaris_df, dimensions, dt, imaris_file, get_intensity=False, skip_limit=1):

    all_parents = []
    all_time_ids = []
    cell_nums = imaris_df["Acceleration"]["Parent"].unique().tolist()
    FIRST_PARENT = cell_nums[0]
    #Get most info
    for parent_id in cell_nums:
        parent = Cell_Info(imaris_df, parent_id, dimensions, dt, imaris_file, skip_limit, get_intensity)
        parent.get_parent_info()
        parent.get_info_per_id()
        parent.get_final_calculations()
        all_parents.append(parent)
        all_time_ids = list(set(all_time_ids + [id_info.TimeIndex for id_info in parent.info_per_id]))
    return all_parents, all_time_ids


def calculate_collectivity_distance(all_parents, all_time_ids):
    for time_index in all_time_ids:
        same_time_ids = []
        for parent in all_parents:
            for i in range(len(parent.info_per_id)):
                if i > time_index:
                    break
                elif parent.info_per_id[i].TimeIndex == time_index and "Instantaneous_Speed" in parent.info_per_id[i].keys():
                    same_time_ids.append(parent.info_per_id[i])
                    break
        if len(same_time_ids) != 1:
            for id_info in same_time_ids:
                other_ids = same_time_ids.copy()
                other_ids.remove(id_info)
                id_info.get_min_distance(other_ids)
                if time_index != all_time_ids[0]: #starting from time 2 because at time 1 none have speed
                    id_info.get_collectivity(other_ids)
                    id_info.get_collectivity(other_ids, cube=True)
    return all_parents


def get_info_from_imaris(imaris_file, imaris_df, dimensions, dt, get_intensity=False, desired_fields=DESIRED_FIELDS):
    start_time = time.time()
    all_parents, all_time_ids = get_all_parents(imaris_df, dimensions, dt, imaris_file, get_intensity)
    parent_end = time.time()
    print("Done with parents, up to here took %d seconds" % round(parent_end - start_time, 3))
    all_parents = calculate_collectivity_distance(all_parents, all_time_ids)
    print("Collectivity takes %d seconds" % round(time.time() - parent_end, 3))
    output_info = []
    single_skips, double_skips = 0, 0
    for parent in all_parents:
        parent_info, parent_single_skips, parent_double_skips = parent.return_info(desired_fields)
        output_info += parent_info
        single_skips += parent_single_skips
        double_skips += parent_double_skips
    print("Total %d seconds. Creating output..." % round(time.time() - start_time, 3))
    return output_info, single_skips, double_skips


def draw_drop_report(ax3, not_dropped, firsts, single_skips, double_skips, acceleration_dropped, msd_dropped, ellipticity_dropped, wave_dropped, other_drops):
    if ax3 == None:
        fig, ax3 = plt.subplots()
    sizes = [not_dropped, firsts, single_skips, double_skips, acceleration_dropped, msd_dropped, ellipticity_dropped, wave_dropped, other_drops]
    labels = ["Remaining Values", "First Occurrences", "After single skips", "After double skips", "Missing acceleration",
              "Missing MSD", "<issing ellipticity", "Missing wave info", "Other drops"]
    pops = []
    for i in range(len(sizes)):
        if sizes[i] == 0:
            pops = [i] + pops
    for pop in pops:
        sizes.pop(pop)
        labels.pop(pop)
    explode = tuple([0.1] + [0] * (len(sizes) - 1))
    ax3.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%")


def create_output_df(imaris_info, single_skips, double_skips, desired_fields=DESIRED_FIELDS, dropna=True,
                     return_drops=False):
    output_df = pd.DataFrame(imaris_info)
    if dropna:
        drop_info = {}
        drop_text = ""
        cell_num = len(set(output_df.Parent))
        initial_len = len(output_df)
        drop_text += "Found %d total data points from %d cells.\n" % (initial_len, cell_num)
        drop_info["Total data points"] = initial_len
        output_df.dropna(subset=["Instantaneous_Speed"], inplace=True)
        old_len = len(output_df)
        speed_dropped = initial_len - old_len
        # if speed_dropped != cell_num + single_skips + double_skips:
        #     print(initial_len, old_len, cell_num, single_skips, double_skips)
        #     raise RuntimeError("Something went wrong while dropping values, this shouldn't happen. Call Ori 0507479922.")
        drop_text += "We dropped %d data points without Instantaneous_Speed, making up %.2f percent of all data points.\n" % (speed_dropped, (round(100 * speed_dropped / initial_len, 2)))
        drop_text += "(%d first occurrences, %d after a single skip and %d after a double skip)\n" % (cell_num, single_skips, double_skips)
        drop_info["First occurrences"] = cell_num
        drop_info["After single skips"] = single_skips
        drop_info["After double skips"] = double_skips
        output_df.dropna(subset=["Acceleration"], inplace=True)
        new_len = len(output_df)
        acceleration_dropped = old_len - new_len
        old_len = new_len
        drop_text += "We dropped another %d data points without Acceleration, making up %.2f percent of all data points.\n" % (acceleration_dropped, round(100 * acceleration_dropped / initial_len, 2))
        drop_info["Missing acceleration"] = acceleration_dropped
        output_df.dropna(subset=["Final_MSD_1"], inplace=True)
        new_len = len(output_df)
        msd_dropped = old_len - new_len
        if msd_dropped > 0:
            old_len = new_len
            drop_text += "We dropped another %d data points without valid MSD parameters (meaning the cell they belonged to had less than 7 data points), making up %.2f percent of all data points.\n" \
                % (msd_dropped, round(100 * msd_dropped / initial_len, 2))
            drop_info["Missing MSD"] = msd_dropped
        if "Eccentricity" in output_df.keys():
            output_df.dropna(subset=["Eccentricity"], inplace=True)
        else:
            output_df.dropna(subset=["Eccentricity_A", "Eccentricity_B", "Eccentricity_C"], inplace=True)
        new_len = len(output_df)
        ellipticity_dropped = old_len - new_len
        if ellipticity_dropped > 0:
            old_len = new_len
            drop_text += "We dropped another %d data points without Ellipticity information (because of Imaris), making up %.2f percent of all data points.\n" \
                % (ellipticity_dropped, round(100 * ellipticity_dropped / initial_len, 2))
            drop_info["Missing ellipticity"] = ellipticity_dropped
        output_df.dropna(subset=["Velocity_Full_Width_Half_Maximum"], inplace=True)
        new_len = len(output_df)
        wave_dropped = old_len - new_len
        if wave_dropped > 0:
            old_len = new_len
            drop_text += "We dropped another %d data points without wave information (meaning the cell they belonged to had less than 3 valid speed values), making up %.2f percent of all data points.\n" \
                % (wave_dropped, round(100 * wave_dropped / initial_len, 2))
            drop_info["Missing wave information"] = wave_dropped
        count = dict(output_df.count())
        other_drops = 0
        for field in list(count.keys()):
            if count[field] < new_len:
                output_df.dropna(subset=[field], inplace=True)
                new_len = len(output_df)
                dropped = old_len - new_len
                old_len = new_len
                drop_text += "Unexpectedly dropped %d data points because %s was NaN. CHECK THIS OUT.\n" % (dropped, field)
                other_drops += dropped
                drop_info["Missing " + field] = dropped
        output_df.reset_index(inplace=True, drop=True)
        drop_text += "Altogether, we kept %d data points, meaning %.2f percent of all data points." % (new_len, round(100 * new_len / initial_len, 2))
        output_df = output_df.reindex(columns=desired_fields)
        output_df.dropna(axis="columns", inplace=True, how="all")
        drop_info["Remaining data points"] = len(output_df)
        if return_drops:
            return output_df, drop_info
        else:
            print(drop_text)
    return output_df


def load_single_pybatch_df(imaris_path, imaris_version=None, dimensions=None, dt=None):
    """
    Function used for testing, doesn't have any validations or protections..
    """
    print("Loading " + imaris_path)
    imaris_version = check_imaris_version(imaris_path, imaris_version)
    imaris_df = get_imaris_df(imaris_path, imaris_version)
    dimensions = check_dimensions(imaris_df, dimensions)
    dt = check_dt(imaris_df, dt)
    imaris_info, single_skips, double_skips = get_info_from_imaris(imaris_path.split(".")[0].split("\\")[-1], imaris_df, dimensions, dt, desired_fields=DESIRED_FIELDS + ["MSDs"])
    output_df = create_output_df(imaris_info, single_skips, double_skips, desired_fields=DESIRED_FIELDS + ["MSDs"])
    return output_df


def single_folder_imaris_parser(rename_dir, output_dir, dimensions=None, dt=None, imaris_version=None):
    startrow = 0
    checked_version = False
    checked_dimensions_dt = False
    if os.path.isfile(rename_dir):
        imaris_files = [os.path.basename(rename_dir)]
        rename_dir = os.path.dirname(rename_dir)
    else:
        imaris_files = os.listdir(rename_dir)
    exp_name = imaris_files[0][:5]
    for i in range(len(imaris_files)):
        imaris_file = imaris_files[i]
        try:
            print("Loading " + imaris_file)
            if imaris_file[:5] != exp_name:
                raise ValueError("There seems to be more than one experiment in the rename folder.")
            output_file = os.path.join(output_dir, exp_name + "_summary_table.xlsx")
            imaris_path = os.path.join(rename_dir, imaris_file)
            if checked_version == False:
                imaris_version = check_imaris_version(imaris_path, imaris_version)
                checked_version = True # Erase this if you want script to check version of every file within the same folder
            imaris_df = get_imaris_df(imaris_path, imaris_version)
            if checked_dimensions_dt == False:
                dimensions = check_dimensions(imaris_df, dimensions)
                dt = check_dt(imaris_df, dt)
                checked_dimensions_dt = True # Erase this if you want script to check dimensions and dt of every file within the same folder
            imaris_info, single_skips, double_skips = get_info_from_imaris(imaris_file.split(".")[0], imaris_df, dimensions, dt)
            output_df = create_output_df(imaris_info, single_skips, double_skips)
            if startrow == 0:
                with pd.ExcelWriter(output_file, mode="w", engine="openpyxl") as writer:
                    output_df.to_excel(writer)
                startrow = len(output_df) + 1
            else:
                with pd.ExcelWriter(output_file, mode="a", engine="openpyxl") as writer:
                    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
                    output_df.to_excel(writer, sheet_name="Sheet1", header=False, startrow=startrow)
                startrow += len(output_df)
            print("Done with %d out of %d for this experiment." % (i, len(imaris_files)))
        except:
            print(imaris_file, "FAILED TO WRITE TO SUMMARY TABLE!")
            raise
    return dimensions, dt, imaris_version


def imaris_parser(rename_dirs, output_dir, dimensions=None, dt=None, imaris_version=None):
    if type(rename_dirs) == str:
        rename_dirs = [rename_dirs]
    for rename_dir in rename_dirs:
        dimensions, dt, imaris_version = single_folder_imaris_parser(rename_dir, output_dir, dimensions, dt, imaris_version)
    return "Done"


#######################################################################################################################################


st.set_option('deprecation.showfileUploaderEncoding', False)

PAGES = ["Renaming", "Summary table creation", "Batch calculations", "Advanced design", "Running"]

cell_finder = re.compile("[^a-zA-Z0-9]([A-Z][0-9][0-9]?)[^a-zA-Z0-9]")
well_pattern = re.compile("\w\w\d\d\d\w\w\w\d\w\d\d")


class Xls1FileError(Exception):
    pass


class Xls2FileError(Exception):
    pass


def parse_xls1(imaris_dir, xls1_path):
    rename_file = xlrd.open_workbook(xls1_path).sheet_by_index(0)
    initials = rename_file.cell_value(1, 0)
    exp_num = rename_file.cell_value(1, 1)
    date = rename_file.cell_value(1, 2)
    channel = rename_file.cell_value(1, 3)
    plate_num = rename_file.cell_value(1, 4)
    exp_type = rename_file.cell_value(1, 5)
    row_names = rename_file.col_values(0)
    col_names = rename_file.row_values(3)
    cell_info = {}
    cell_paths = {}
    for f_name in os.listdir(imaris_dir):
        cell = cell_finder.findall(f_name)
        if len(cell) > 1:
            raise ValueError("Issue with Imaris file names - found more than one cell name: %s." % ", ".join(cell))
        elif len(cell) == 0:
            raise ValueError("Issue with Imaris file names - could not find a correct cell name.")
        cell = cell[0]
        indexes = [(x, col_names.index(int(cell[1:]))) for x in range(row_names.index(cell[0]), row_names.index(cell[0]) + row_names.count(cell[0]))]
        cell_name = cell[0] + cell[1:].zfill(2)
        cell_paths[f_name] = cell_name
        treatments = [rename_file.cell_value(*index) for index in indexes]
        cell_info[cell_name] = treatments
    renaming_info = {"initials": initials,
                     "exp_num": exp_num,
                     "date": date,
                     "channel": channel,
                     "plate_num": plate_num,
                     "exp_type": exp_type,
                     "cell_info": cell_info,
                     "cell_paths": cell_paths}
    return renaming_info


def parse_xls2(xls2_path):
    xls2 = xlrd.open_workbook(xls2_path).sheet_by_index(0)
    batch_experiments_info = []
    exp_names = xls2.col_values(1, start_rowx=4, end_rowx=19)
    path_list = xls2.col_values(1, start_rowx=32, end_rowx=35)
    for i in range(len(exp_names)):
        exp_name = exp_names[i]
        if exp_name not in ["NNN0", ""]:
            if exp_names.count(exp_name) > 1:
                exp_name += "_%d" % i
            row_values = xls2.row_values(4+i, start_colx=2)#, end_colx=38)
            exp_sub_name = row_values[0]
            wells = [well for well in row_values[1:] if re.match(well_pattern, well)]
            batch_experiments_info.append({"exp_name": exp_name, "exp_sub_name": exp_sub_name, "wells": wells,
                                           "protocol_file": path_list[0], "incucyte_files": path_list[1],
                                           "imaris_xls_files": path_list[2], "design": "auto", "table_info": {},
                                           "manual_graph_choice": False, "param_graphs": {}, "param_pair_graphs": {}})
    if len(wells) == 0:
        raise Xls2FileError("Had trouble reading wells from xls2 file - check that they are in the correct format.")
    return batch_experiments_info


def renaming(imaris_dir, renaming_info, rename_dir):
    prefix = renaming_info["initials"] + renaming_info["exp_num"] + renaming_info["date"] + renaming_info["channel"] + renaming_info["plate_num"]
    suffix = renaming_info["exp_type"]
    counter = 0
    counter_jump = 1 / len(os.listdir(imaris_dir))
    rename_progress = st.progress(counter)
    for f_name in os.listdir(imaris_dir):
        with st.spinner("Renaming %s..." % f_name):
            rename_progress.progress(counter)
            extension = "." + f_name.split(".")[-1]
            cell_name = renaming_info["cell_paths"][f_name]
            old_file = os.path.join(imaris_dir, f_name)
            new_file = os.path.join(rename_dir, prefix + cell_name + "".join(renaming_info["cell_info"][cell_name]) + suffix + extension)
            if os.path.isfile(old_file):
                copyfile(old_file, new_file)
            else:
                raise ValueError("Couldn't find file:", old_file)
            counter += counter_jump
        st.success("Done with %s" % f_name)
    rename_progress.progress(100)
    return


def create_drop_page(imaris_file, drop_info):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(8.27, 11.69)
    fig.suptitle(imaris_file, size=16, wrap=True)
    if type(drop_info) == str:
        ax = fig.add_subplot(frame_on=False)
        ax.set_axis_off()
        ax.text(0.5, 0.5, drop_info, size=14, wrap=True, ha="center", va="center")
    else:
        gs = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.1, figure=fig)
        ax2 = fig.add_subplot(gs[0, :], frame_on=False)
        ax2.set_axis_off()
        ax2.table([[key, drop_info[key]] for key in list(drop_info.keys())], cellColours=[["c", "w"]]*len(drop_info), cellLoc="left", transform=ax2.transAxes, loc="center")
        ax3 = fig.add_subplot(gs[1:, :])
        ax3.pie(list(drop_info.values())[1:], explode=tuple([0] * (len(drop_info) - 2) + [0.1]), labels=list(drop_info.keys())[1:], autopct="%1.1f%%")
    return fig


def create_summary_table(rename_dir, output_file, drop_report_path, initial_session_id, state, whereami,
                         dimensions=None, dt=None, imaris_version=None, get_intensity=False):
    if not os.path.exists(os.path.dirname(output_file)):
        os.mkdir(os.path.dirname(output_file))
    checked_version = False
    checked_dimensions_dt = False
    imaris_files = os.listdir(rename_dir)
    counter = 0
    counter_jump = 1 / len(imaris_files)
    summary_progress = st.progress(counter)
    exp_name = imaris_files[0][:5]
    #printed_info = False
    dropped_output_files = []
    for imaris_file in imaris_files:
        if imaris_file[:5] != exp_name:
            raise ValueError("There seems to be more than one experiment in the rename folder.")
        imaris_path = os.path.join(rename_dir, imaris_file)
        imaris_df = None
        if checked_version == False:
            imaris_version = check_imaris_version(imaris_path, imaris_version)
            checked_version = True # Erase this if you want script to check version of every file within the same folder
        if checked_dimensions_dt == False:
            imaris_df = get_imaris_df(imaris_path, imaris_version)
            dimensions = check_dimensions(imaris_df, dimensions)
            dt = check_dt(imaris_df, dt)
            checked_dimensions_dt = True # Erase this if you want script to check dimensions and dt of every file within the same folder
        if whereami > state.wherewasi:
            with st.spinner("Working on %s..." % imaris_file):
                #if checked_dimensions_dt and checked_version and not printed_info:
                #    printed_info = True
                #    st.success("Your experiment was run on Imaris %d, has %d dimension and a dt of %d minutes" % (imaris_version, dimensions, dt))
                summary_progress.progress(counter)
                try:
                    if imaris_df == None:
                        imaris_df = get_imaris_df(imaris_path, imaris_version)
                    imaris_info, single_skips, double_skips = get_info_from_imaris(imaris_file.split(".")[0], imaris_df,
                                                                                   dimensions, dt,
                                                                                   get_intensity=get_intensity)
                    output_df, drop_info = create_output_df(imaris_info, single_skips, double_skips, return_drops=True)
                    if imaris_file not in state.drop_infos.keys():
                        state.drop_infos[imaris_file] = drop_info
                    fig = create_drop_page(imaris_file, drop_info)
                    st.text("check out this drop report while you wait!")
                    st.write(fig)
                    plt.close(fig)
                    file_name = os.path.splitext(output_file)[0] + "_" + os.path.splitext(imaris_file)[0] + ".xlsx"
                    dropped_output_files.append(file_name)
                    with pd.ExcelWriter(file_name, mode="w", engine="openpyxl") as writer:
                        output_df.to_excel(writer)
                    full_output_df = pd.DataFrame(imaris_info)
                    full_output_df = full_output_df.reindex(columns=DESIRED_FIELDS)
                    full_output_df.dropna(axis="columns", how="all", inplace=True)
                    full_file = os.path.splitext(file_name)[0] + "_FULL.xlsx"
                    with pd.ExcelWriter(full_file, mode="w", engine="openpyxl") as writer:
                        full_output_df.to_excel(writer)
                except Exception as e:
                    error_msg = "FAILED TO WRITE %s TO SUMMARY TABLE!\n" % imaris_file + str(e)
                    if type(e) == xlrd.compdoc.CompDocError:
                        error_msg += "\nThe Imaris file is corrupted - open the xls file in excel and 'save as', "\
                                     "overwriting the current file. Then try again. If this doesn't work, contact Ori "\
                                     "- 0507479922 "
                    st.error(error_msg)
                    print(error_msg)
                    state.drop_infos[imaris_file] = error_msg
                    raise
            state.wherewasi = whereami
        whereami += 1
        st.success("Done with %s" % imaris_file)
        counter += counter_jump
        summary_progress.progress(100)
        write_state(log_file, state)
    with st.spinner("merging files and creating TASC pickles..."):
        if whereami > state.wherewasi:
            skip = False
            dirname = os.path.join(os.path.dirname(output_file), "TASC_pickles")
            if os.path.exists(dirname):
                if len(os.listdir(dirname)) == 2:
                    skip = True
            else:
                os.mkdir(dirname)
            if not skip:
                if len(dropped_output_files) != len(imaris_files):
                    dropped_output_files = [os.path.join(os.path.dirname(output_file),
                                                         os.path.splitext(output_file)[0] + "_" +
                                                         os.path.splitext(imaris_file)[0] + ".xlsx")
                                            for imaris_file in imaris_files]
                summary_df = pd.concat([pd.read_excel(dropped_file) for dropped_file in dropped_output_files],
                                       ignore_index=True)
                summary_df.drop(columns="Unnamed: 0", inplace=True)
                if len(summary_df) <= 1048575:  # max excel sheet size
                    with pd.ExcelWriter(output_file, mode="w", engine="openpyxl") as writer:
                        summary_df.to_excel(writer)
                else:
                    st.warning("Too much data for one sheet. Could not write to summary_table.xlsx. "
                               "All the data exists in the pickle file.")
                    #for i in range(1, int(np.ceil(len(summary_df) / 1048575)) + 1):
                    #    with pd.ExcelWriter(os.path.splitext(output_file)[0] + "_part%d.xlsx" % i,
                    #                        mode="w", engine="openpyxl") as writer:
                    #        summary_df[(i - 1) * 1048575:i * 1048575].to_excel(writer)
                summary_df.to_pickle(os.path.join(dirname, "rawdatagraph.pickle"))
                summary_df.drop(columns=["TimeIndex", "x_Pos", "y_Pos", "z_Pos", "Parent", "dt", "ID"], inplace=True,
                                errors="ignore")
                summary_df.to_pickle(os.path.join(dirname, "rawdata.pickle"))
                state.wherewasi = whereami
        whereami += 1
        st.success("Done Merging")
        write_state(log_file, state)
    if whereami > state.wherewasi:
        with PdfPages(drop_report_path) as pdf:
            for imaris_file in imaris_files:
                fig = create_drop_page(imaris_file, state.drop_infos[imaris_file])
                pdf.savefig(fig)
                plt.close(fig)
        write_state(log_file, state)
        state.wherewasi = whereami
    whereami += 1
    return imaris_version, dimensions, dt, whereami


def make_rename_dir(imaris_dir):
    rename_dir = os.path.join(os.path.split(imaris_dir)[0], "renaming")
    return rename_dir


def make_drop_report_path(summary_table_path, exp_serial_num):
    drop_report_path = os.path.join(summary_table_path, "drop_report_exp%d.pdf" % exp_serial_num)
    return drop_report_path


def validate_xls1(xls1_path):
    xls1_file = xlrd.open_workbook(xls1_path).sheet_by_index(0)
    headers = xls1_file.row_values(0, end_colx=6)
    if (headers[0] != "Initials" or
        headers[1] != "exp. Num" or
        headers[2] != "date(DDMMYY)" or
        headers[3] != "Channel" or
        headers[4] != "plate number" or
        (headers[5] != "Exp Type" and headers[5] != "Experiment Type") or
        "" in xls1_file.row_values(1, end_colx=6)):
        raise Xls1FileError("Problem found in first two rows")
    if (xls1_file.row_values(3, end_colx=12) != ["Location", "Sub Localization", 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0] or
        xls1_file.col_values(0, start_rowx=4, end_rowx=40) != ["B"]*6 + ["C"]*6 + ["D"]*6 + ["E"]*6 + ["F"]*6 + ["G"]*6):
        raise Xls1FileError("Problem found in table headers")
    for row in range(4, 40):
        for col in range(2, 12):
            if len(xls1_file.cell_value(row, col)) != 4:
                raise Xls1FileError("Values in table were not 4 letters long")
    return True


def validate_xls2(xls2_path):
    xls2_file = xlrd.open_workbook(xls2_path).sheet_by_index(0)
    if xls2_file.cell(1, 0).value != "Experiment Layout" or not (xls2_file.row_values(3, end_colx=27) == ["Exp Num", "Exp Name", "Exp Sub Name", "1", "2", "3", "4", "5",
                                                        "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"] or
                                                                xls2_file.row_values(3, end_colx=27) == ["Exp Num", "Exp. Name", "Exp Sub Name", "1", "2", "3", "4", "5",
                                                        "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"]):
        raise Xls2FileError("Doesn't seem to be an xls2 file - 'Experiment Layout' table isn't right")
    if xls2_file.cell(19, 0).value != "Experiment Structure Path" or not (xls2_file.row_values(20, start_colx=1, end_colx=3) == ['Exp Name', 'Exp Path'] or
                                                                          xls2_file.row_values(20, start_colx=1, end_colx=3) == ['Exp. Name', 'Exp Path']):
        raise Xls2FileError("Problem with 'Experiment Structure Path' table")
    if xls2_file.col_values(0, start_rowx=32, end_rowx=36) != ['Protocol file', 'IncuCyte files', 'Imaris xls files']:
        raise Xls2FileError("Problem with file information (end of file)")
    return True


def validate_summary_table(summary_table_path):
    return True


def get_folder(folder_description, file_type, key=None, value=""):
    path = st.text_input("Enter the %s path here (local or network location)" % folder_description, key=key, value=value).replace("\"", "")
    if path == "":
        st.text("Waiting for %s path..." % folder_description)
    else:
        try:
            files = os.listdir(path)
            if file_type == "empty":
                if os.listdir(path) == []:
                    return path
                else:
                    st.warning(folder_description.capitalize() + " has files in it! Try again with an empty folder or clear it out.")
            else:
                if files == []:
                    st.warning(folder_description.capitalize() + " is empty! Try again.")
                elif True not in [file_type in f.split(".")[-1] for f in files]:
                    st.warning(folder_description.capitalize() + " doesn't contain %s files! Try again." % file_type)
                else:
                    return path
        except FileNotFoundError:
            if file_type == "empty":
                if path[-1] == "\\":
                    path = path[:-1]
                if os.path.exists(os.path.split(path)[0]) or os.path.basename(path) == "batch_output":
                    return path
                else:
                    st.warning("Invalid path. Make sure the path you entered exists and try again.")
            st.warning("Couldn't find folder! Try again.")
    return ""


def get_file_path(file_description, file_type, new_file=False, key=None, validation_func=None, value=""):
    file_path = st.text_input("Write the %s path here (local or network location)" % file_description, key=key, value=value).replace("\"", "")
    if file_path == "":
        st.text("Waiting for %s path..." % file_description)
    else:
        if file_type not in file_path.split(".")[-1] or len(file_path.split(".")[-1]) not in [3, 4]:
            st.warning("File given should be a %s file, try again." % file_type)
        else:
            if new_file == False:
                if os.path.exists(file_path):
                    try:
                        if validation_func(file_path):
                            st.text("Valid %s, please continue." % file_description)
                            return file_path
                    except Exception as e:
                        st.warning("Invalid %s - %s" % (file_description, str(e)))
                else:
                    st.warning("Couldn't find the file! Check your path and try again.")
            else:
                if os.path.exists(file_path):
                    st.warning("A file with that path already exists. Delete it or try another path.")
                else:
                    return file_path
    return ""


def make_experiment():
    return {"imaris_dir": "",
            "rename_dir": "",
            "xls1_path": "",
            "renaming_info": {"initials": "", "exp_num": "", "date": "", "channel": "", "plate_num": "", "exp_type": "",
                              "cell_info": {}, "cell_paths": {}},
            "summary_table_path": "",
            "xls2_path": "",
            "scratch": False,
            "batch_experiments_info": [None],
            "batch_path": "",
            "drop_report_path": "",
            "full": True,
            "imaris_version": 0,
            "dt": 0,
            "dimensions": 0,
            "cell_names": [],
            "get_intensity": False,
            "output_edit": False,
            "protocol_file": "",
            "incucyte_files": "",
            "imaris_xls_files": ""}

def renaming_section(experiments, exp_serial_num):
    another = False
    next_experiments = [None]
    if experiments == [None]:
        exp = make_experiment()
    else:
        exp = experiments[0]
        if len(experiments) > 1:
            another = True
            next_experiments = experiments[1:]
    st.header("Renaming experiment #%d" % exp_serial_num)
    st.text("The following section will rename all Imaris excel files according to your renaming info or xls1 file.")

    #get imaris files
    imaris_pass = False
    exp["imaris_dir"] = get_folder("Imaris excel folder", "xls", key="imaris_exp%d" % exp_serial_num, value=exp["imaris_dir"])
    if exp["imaris_dir"]:
        if exp["rename_dir"] == "":
            exp["rename_dir"] = make_rename_dir(exp["imaris_dir"])
        imaris_pass = True

    #get renaming info
    renaming_info_pass = False
    all_good = True
    if imaris_pass:
        info_source = st.radio("How will you provide renaming information?", ["manual input", "xls1 file"], index=0 if exp["xls1_path"] == "" else 1, key="imaris_exp%d" % exp_serial_num)
        if info_source == "xls1 file":
            exp["xls1_path"] = get_file_path("xls1 file", "xls", key="xls1_exp%d" % exp_serial_num, validation_func=validate_xls1, value=exp["xls1_path"])
            if exp["xls1_path"]:
                exp["renaming_info"] = parse_xls1(exp["imaris_dir"], exp["xls1_path"])
        elif info_source == "manual input":
            exp["renaming_info"]["initials"] = st.text_input("Enter your initials:", max_chars=2, key="imaris_exp%d" % exp_serial_num, value=exp["renaming_info"]["initials"])
            if len(exp["renaming_info"]["initials"]) < 2 and len(exp["renaming_info"]["initials"]) != 0:
                st.warning("Initials must be exactly 2 characters")
                all_good = False
            exp["renaming_info"]["exp_num"] = st.text_input("Enter your experiment number:", max_chars=3, key="imaris_exp%d" % exp_serial_num, value=exp["renaming_info"]["exp_num"]).zfill(3)
            if (len(exp["renaming_info"]["exp_num"]) < 3 or not exp["renaming_info"]["exp_num"].isdigit()) and len(exp["renaming_info"]["exp_num"]) != 0:
                st.warning("Experiment number must be exactly 3 characters, all numbers")
                all_good = False
            exp["renaming_info"]["date"] = st.text_input("Enter the date you ran the IncuCyte experiment (DDMMYY):", max_chars=6, key="imaris_exp%d" % exp_serial_num, value=exp["renaming_info"]["date"])
            if (len(exp["renaming_info"]["date"]) < 6 or not exp["renaming_info"]["date"].isdigit()) and len(exp["renaming_info"]["date"]) != 0:
                st.warning("Date must be exactly 6 characters, all numbers")
                all_good = False
            exp["renaming_info"]["channel"] = st.radio("Choose your channel:", ["CHR", "CHG"],
                                                             index=1 if exp["renaming_info"]["channel"] == "CHG" else 0,
                                                             key="channel%d" % exp_serial_num)
            if exp["renaming_info"]["channel"] == "":
                all_good = False
            exp["renaming_info"]["plate_num"] = st.text_input("Enter your plate number:", max_chars=1, key="imaris_exp%d" % exp_serial_num, value=exp["renaming_info"]["plate_num"])
            if (len(exp["renaming_info"]["plate_num"]) != 1 or not exp["renaming_info"]["plate_num"].isdigit()) and len(exp["renaming_info"]["plate_num"]) != 0:
                st.warning("Plate number must be 1 digit")
                all_good = False
            exp["renaming_info"]["exp_type"] = st.text_input("Enter your experiment type:", max_chars=4, key="imaris_exp%d" % exp_serial_num, value=exp["renaming_info"]["exp_type"])
            if len(exp["renaming_info"]["exp_type"]) < 4 and len(exp["renaming_info"]["exp_type"]) != 0:
                st.warning("Experiment type must be exactly 4 characters")
                all_good = False
            if exp["renaming_info"]["cell_info"] == {}:
                for f_name in os.listdir(exp["imaris_dir"]):
                    cell = cell_finder.findall(f_name)
                    if len(cell) != 1:
                        raise ValueError("Issue with Imaris file names - could not find the correct cell name.")
                    cell = cell[0]
                    cell_name = cell[0] + cell[1:].zfill(2)
                    exp["renaming_info"]["cell_paths"][f_name] = cell_name
                    exp["renaming_info"]["cell_info"][cell_name] = ["", "", "NNN0", "NNN0", "NNN0", "NNN0"]
            cell_names = list(exp["renaming_info"]["cell_info"].keys())
            cell_names.sort()
            for cell_name in cell_names:
                st.subheader(cell_name)
                exp["renaming_info"]["cell_info"][cell_name][0] = st.text_input("Enter the cell line of well %s (4 characters):" % cell_name,
                                                                                max_chars=4, key="imaris_exp%d%s" % (exp_serial_num, cell_name),
                                                                                value=exp["renaming_info"]["cell_info"][cell_name][0])
                if 0 < len(exp["renaming_info"]["cell_info"][cell_name][0]) < 4:
                    st.warning("Cell line must be exactly 4 characters")
                    all_good = False
                another_treatment = [False] * 4
                for i in range(1, 6):
                    if i == 1 or another_treatment[i-2]:
                        exp["renaming_info"]["cell_info"][cell_name][i] = st.text_input("Enter treatment #%d of well %s (4 characters):" % (i, cell_name),
                                                                                        max_chars=4, key="imaris_exp%d%s" % (exp_serial_num, cell_name),
                                                                                        value=exp["renaming_info"]["cell_info"][cell_name][i])
                        if 0 < len(exp["renaming_info"]["cell_info"][cell_name][i]) < 4:
                            st.warning("Each treatment must be exactly 4 characters")
                            all_good = False
                        elif len(exp["renaming_info"]["cell_info"][cell_name][i]) == 4 and exp["renaming_info"]["cell_info"][cell_name][i] != "NNN0":
                            another_treatment[i-1] = st.checkbox("add another treatment?", key="another_treament%s%d" % (cell_name, i), value=another_treatment[i-1])
                if list(exp["renaming_info"]["cell_info"].values()).count(exp["renaming_info"]["cell_info"][cell_name]) > 1 and \
                        exp["renaming_info"]["cell_info"][cell_name] != ["", "", "NNN0", "NNN0", "NNN0", "NNN0"]:
                    st.warning("You already gave this info for a different cell")
        if "" not in exp["renaming_info"].values() and ["", "", "NNN0", "NNN0", "NNN0", "NNN0"] not in exp["renaming_info"]["cell_info"].values() and all_good:
            renaming_info_pass = True

    #get rename folder
    rename_pass = False
    if imaris_pass and renaming_info_pass:
        exp["rename_dir"] = get_folder("renaming folder", "empty", key="rename_exp%d" % exp_serial_num, value=exp["rename_dir"])
        if exp["rename_dir"]:
            rename_pass = True

    if renaming_info_pass and imaris_pass and rename_pass:
        another = st.checkbox("Add another experiment.", key="another_exp%d" % exp_serial_num, value=another)
        if another:
            return [exp] + renaming_section(next_experiments, exp_serial_num+1)
        else:
            return [exp]
    return [None]


def skipped_rename_section(experiments, exp_serial_num):
    another = False
    next_experiments = [None]
    if experiments == [None]:
        exp = make_experiment()
    else:
        exp = experiments[0]
        if len(experiments) > 1:
            another = True
            next_experiments = experiments[1:]
    st.header("Renamed files of experiment #%d" % exp_serial_num)
    st.text("In the following section enter the location of your renamed files in order to create summary tables.")
    exp["rename_dir"] = get_folder("rename folder", "xls", value=exp["rename_dir"])
    if exp["rename_dir"]:
        another = st.checkbox("Add another experiment.", key="another_exp%d" % exp_serial_num, value=another)
        if another:
            return [exp] + skipped_rename_section(next_experiments, exp_serial_num+1)
        else:
            return [exp]
    return [None]


def skipped_summary_table_section(experiments, exp_serial_num):
    another = False
    next_experiments = [None]
    if experiments == [None]:
        exp = make_experiment()
    else:
        exp = experiments[0]
        if len(experiments) > 1:
            another = True
            next_experiments = experiments[1:]
    st.header("Summary table for experiment #%d" % exp_serial_num)
    st.text("In the following section enter the location of your summary table file.")
    exp["summary_table_path"] = get_folder("results folder for summary table", "xls", key="sum_exp%d" % exp_serial_num, value=exp["summary_table_path"])
    if exp["summary_table_path"]:
        exp["drop_report_path"] = make_drop_report_path(exp["summary_table_path"], exp_serial_num)
        another = st.checkbox("Add another experiment.", key="another_exp%d" % exp_serial_num, value=another)
        if another:
            return [exp] + skipped_summary_table_section(next_experiments, exp_serial_num+1)
        else:
            return [exp]
    return [None]


def summary_table_section(exp, exp_serial_num):
    st.header("Summary table for experiment #%d" % exp_serial_num)
    exp["get_intensity"] = st.checkbox("Include intensity information", key="intensity%d" % exp_serial_num, value=exp["get_intensity"])
    st.text("In the following section enter the desired path for your output files.")
    st.warning("The files for this experiment came from the following path:\n" + exp["imaris_dir"] if exp["imaris_dir"] != "" else exp["rename_dir"])
    exp["summary_table_path"] = get_folder("output (summary table)", "empty", key="sum_exp%d" % exp_serial_num,
                                           value=exp["summary_table_path"] if exp["summary_table_path"] != "" else os.path.join(os.path.dirname(exp["rename_dir"]), "output"))
    if exp["summary_table_path"]:
        exp["drop_report_path"] = make_drop_report_path(exp["summary_table_path"], exp_serial_num)
    return exp


def batch_exp_info_subsection(batch_exps, exp, exp_serial_num, batch_exp_serial_num):
    another = False
    next_batch_exps = [None]
    if batch_exps == [None]:
        batch_exp_info = {"exp_name": "", "exp_sub_name": "", "wells": [], "protocol_file": exp["protocol_file"],
                          "incucyte_files": exp["incucyte_files"], "imaris_xls_files": exp["imaris_xls_files"],
                          "design": "auto", "table_info": {}, "manual_graph_choice": False, "param_graphs": {},
                          "param_pair_graphs": {}}
    else:
        batch_exp_info = batch_exps[0]
        if len(batch_exps) > 1:
            another = True
            next_batch_exps = batch_exps[1:]
    st.header("Info for batch experiment #%d in experiment %d" % (batch_exp_serial_num, exp_serial_num))
    batch_exp_info["exp_name"] = st.text_input("Enter the batch experiment name:",
                                               key="batch_name%d%d" % (exp_serial_num, batch_exp_serial_num),
                                               value=batch_exp_info["exp_name"])
    batch_exp_info["exp_sub_name"] = st.text_input("Enter the batch experiment subname:",
                                                   key="batch_subname%d%d" % (exp_serial_num, batch_exp_serial_num),
                                                   value=batch_exp_info["exp_sub_name"])
    cell_names = exp["cell_names"]
    if not cell_names:
        cell_names = list(exp["renaming_info"]["cell_info"].keys())
        if not cell_names:
            if exp["rename_dir"]:
                cell_names = [f[15:18] for f in os.listdir(exp["rename_dir"])]
        if not cell_names:
            cell_names = [f[29:32] for f in os.listdir(exp["summary_table_path"]) if "summary_table_" in f and "_FULL" in f]
        cell_names.sort()
        exp["cell_names"] = cell_names
    with st.form(key="wellform%d%d" % (exp_serial_num, batch_exp_serial_num)):
        batch_exp_info["wells"] = st.multiselect("Choose which wells are part of the batch experiment:", cell_names,
                                                 default=batch_exp_info["wells"],
                                                 key="batch_wells%d%d" % (exp_serial_num, batch_exp_serial_num))
        st.form_submit_button("update")
    plate_table = pd.DataFrame([[letter + str(num).zfill(2) for num in PLATE_NUMBERS] for letter in PLATE_LETTERS],
                               columns=PLATE_NUMBERS, index=PLATE_LETTERS)
    styler = plate_table.style.apply(lambda x: ["background-color: yellow" if v in batch_exp_info["wells"] else "" \
                                                for v in x], axis=0)
    st.table(styler)
    if batch_exp_info["exp_name"] and batch_exp_info["exp_sub_name"] and batch_exp_info["wells"] and \
            batch_exp_info["protocol_file"] and batch_exp_info["incucyte_files"] and batch_exp_info["imaris_xls_files"]:
        another = st.checkbox("Add another batch experiment.",
                              key="another_exp%d%d" % (exp_serial_num, batch_exp_serial_num), value=another)
        if another:
            return [batch_exp_info] + batch_exp_info_subsection(next_batch_exps, exp, exp_serial_num,
                                                                batch_exp_serial_num+1)
        else:
            return [batch_exp_info]
    return [None]


def batch_section(exp, exp_serial_num):
    st.header("Batch inforamtion for experiment #%d" % exp_serial_num)
    st.text("In the following section enter the batch experiment inforamtion and the desired path for your batch analysis output.")
    #get experiment info
    exp["batch_path"] = get_folder("batch output", "empty", key="batch_exp%d" % exp_serial_num,
                                   value=exp["batch_path"] if exp["batch_path"] != "" else
                                   os.path.join(exp["summary_table_path"], "batch_output"))
    if exp["renaming_info"]["exp_type"].upper() == "WH00":
        exp["scratch"] = True
    elif exp["renaming_info"]["exp_type"].upper() != "":
        exp["scratch"] = False
    else:
        try:
            file_list = os.listdir(exp["rename_dir"])
        except FileNotFoundError:
            file_list = []
        if not file_list:
            file_list = os.listdir(exp["summary_table_path"])
        if file_list[0].split(".")[0][-4:].upper() == "WH00":
            exp["scratch"] = True
    exp["scratch"] = st.checkbox("Does the experiment have a scratch?", value=exp["scratch"],
                                 key="scratch_exp%d" % exp_serial_num)
    exp["full"] = st.checkbox("Run batch calculations on full summary table (no drops)", key="full%d" % exp_serial_num,
                              value=exp["full"])
    if exp["full"] == False:
        st.warning("You will get less information by running calculations on the partial summary table!")
    info_source = st.radio("How will you provide batch analysis information?", ["manual input", "xls2 file"],
                           index=0 if exp["xls2_path"] == "" else 1, key="imaris_exp%d" % exp_serial_num)
    if info_source == "xls2 file":
        exp["xls2_path"] = get_file_path("xls2 file", "xls", key="xls2_exp%d" % exp_serial_num,
                                         validation_func=validate_xls2, value=exp["xls2_path"])
        if exp["xls2_path"]:
            exp["batch_experiments_info"] = parse_xls2(exp["xls2_path"])
    elif info_source == "manual input":
        exp["protocol_file"] = st.text_input("Enter the protocol file path:", key="batch_protocol%d" % exp_serial_num,
                                             value=exp["protocol_file"])
        exp["incucyte_files"] = st.text_input("Enter the Incucyte files folder path:",
                                              key="batch_incucyte%d" % exp_serial_num,
                                              value=exp["incucyte_files"])
        exp["imaris_xls_files"] = st.text_input("Enter the Imaris xls files folder path:",
                                                key="batch_imaris%d" % exp_serial_num,
                                                value=exp["imaris_xls_files"] if exp["imaris_xls_files"] else exp["imaris_dir"])
        exp["batch_experiments_info"] = batch_exp_info_subsection(exp["batch_experiments_info"], exp, exp_serial_num, 1)
        if None not in exp["batch_experiments_info"]:
            exp_names = [batch_exp_info["exp_name"] for batch_exp_info in exp["batch_experiments_info"]]
            for exp_name in exp_names:
                if exp_names.count(exp_name) > 1:
                    st.warning("Two experiments have the same name: %s.\nYou must fix this before moving on" % exp_name)
                    st.stop()
    return exp


def get_basic_table_info(exp, batch_exp, skip_rename, skip_summary_table):
    basic_table_info = {"column_treatments": [], "column_names": [], "row_treatments": [], "row_names": [],
                       "cell_names": [well[-3:] for well in batch_exp["wells"]], "treatments": [], "valid": False}
    if exp["renaming_info"]["cell_info"] != {} and not skip_rename:
        batch_exp["treatments_dict"] = exp["renaming_info"]["cell_info"].copy()
    else:
        if not skip_summary_table:
            file_names = [f[15:46].replace("NNN0", "") for f in os.listdir(exp["rename_dir"])
                          if f[15:18] in basic_table_info["cell_names"]]
        else:
            relevant_files = [f for f in os.listdir(exp["summary_table_path"]) if
                              "summary_table_" in f and "_FULL" in f]
            file_names = [f[29:60].replace("NNN0", "") for f in relevant_files
                          if f[29:32] in basic_table_info["cell_names"]]
            file_names = list(set(file_names))
        batch_exp["treatments_dict"] = {f[:3]: [f[i:i + 4] for i in range(3, len(f), 4)] for f in file_names}
    batch_exp["treatments_dict"] = {k: batch_exp["treatments_dict"][k] for k in batch_exp["treatments_dict"]
                                    if k in basic_table_info["cell_names"]}
    treatment_lists = list(batch_exp["treatments_dict"].values())
    all_treatments = []
    for treat_list in treatment_lists:
        for treat in treat_list:
            if "CON" not in treat and treat != "CTRL" and False in [treat in t_list for t_list in treatment_lists]:
                all_treatments.append(treat)
    all_treatments = list(set(all_treatments))
    groups = [[all_treatments[0]], []]
    for treat in all_treatments[1:]:
        single_placed = False
        for i in range(2):
            same_group = groups[i]
            other_group = groups[i - 1]
            for second_treat in same_group:
                if second_treat != treat:
                    common_count = [treat in treat_list and second_treat in treat_list for treat_list in
                                    treatment_lists].count(True)
                    if common_count == 0:  # means the treatments are not combined (belong to same group)
                        if not single_placed:
                            same_group.append(treat)
                            single_placed = True
                    elif common_count == 1:  # means the treatments are combined once (belong to other groups)
                        if not single_placed:
                            other_group.append(treat)
                            single_placed = True
                    elif common_count > 1:  # means the treatments are combined more than once (belong to same group)
                        same_group.append(treat + second_treat)
    groups.sort(key=lambda x: len(x), reverse=True)
    basic_table_info["treatments"] = groups[0] + groups[1]
    basic_table_info["column_treatments"] = groups[0]
    basic_table_info["row_treatments"] = groups[1]
    return basic_table_info


def auto_table_info_selection(exp, exp_serial_num, batch_exp, batch_exp_serial_num, skip_rename, skip_summary_table):
    st.subheader("Automatic table information:")
    if batch_exp["table_info"] == {}:
        batch_exp["table_info"] = get_basic_table_info(exp, batch_exp, skip_rename, skip_summary_table)
    elif batch_exp["table_info"]["column_treatments"] == []:
        batch_exp["table_info"] = get_basic_table_info(exp, batch_exp, skip_rename, skip_summary_table)
    batch_exp["table_info"]["column_treatments"].sort()
    if len(batch_exp["table_info"]["column_names"]) != len(batch_exp["table_info"]["column_treatments"]):
        batch_exp["table_info"]["column_names"] = batch_exp["table_info"]["column_treatments"].copy()
    batch_exp["table_info"]["row_treatments"].sort()
    if len(batch_exp["table_info"]["row_names"]) != len(batch_exp["table_info"]["row_treatments"]):
        batch_exp["table_info"]["row_names"] = batch_exp["table_info"]["row_treatments"].copy()
    st.subheader("Preview")
    zeroes_preview = pd.DataFrame(0, index=["Control"]+batch_exp["table_info"]["row_names"],
                                  columns=["Control"]+batch_exp["table_info"]["column_names"])
    preview = zeroes_preview.copy()
    for cell_name in batch_exp["table_info"]["cell_names"]:
        temp_preview = zeroes_preview.copy()
        treatment_list = batch_exp["treatments_dict"][cell_name]
        col_found = False
        row_found = False
        for treat in treatment_list:
            for col_treat in batch_exp["table_info"]["column_treatments"]:
                if treat in col_treat:
                    temp_preview[col_treat] += 1
                    col_found = True
            for row_treat in batch_exp["table_info"]["row_treatments"]:
                if treat in row_treat:
                    temp_preview.loc[row_treat] += 1
                    row_found = True
        if not col_found:
            temp_preview["Control"] += 1
        if not row_found:
            temp_preview.loc["Control"] += 1
        preview.loc[temp_preview.max(axis=1).idxmax(), temp_preview.max(axis=0).idxmax()] = cell_name
    st.table(preview)
    if 0 in preview.values:
        st.error("Couldn't automatically fill table. Try manually creating the table or just use auto layout.")
        batch_exp["table_info"] = {}
    else:
        batch_exp["table_info"]["valid"] = True
        st.subheader("Column names")
        for i in range(len(batch_exp["table_info"]["column_names"])):
            batch_exp["table_info"]["column_names"][i] = st.text_input("how should we call %s?" %
                                                                       batch_exp["table_info"]["column_treatments"][i],
                                                                       key="co_name%d%d%d" % (i, exp_serial_num,
                                                                                              batch_exp_serial_num),
                                                                       value=batch_exp["table_info"]["column_names"][i])
        treatments = [treat for treat in batch_exp["table_info"]["treatments"]
                      if treat not in batch_exp["table_info"]["column_treatments"]]
        st.subheader("Row names")
        for i in range(len(batch_exp["table_info"]["row_names"])):
            batch_exp["table_info"]["row_names"][i] = st.text_input("how should we call %s?" %
                                                                    batch_exp["table_info"]["row_treatments"][i],
                                                                    key="co_name%d%d%d" % (i, exp_serial_num,
                                                                                           batch_exp_serial_num),
                                                                    value=batch_exp["table_info"]["row_names"][i])
        batch_exp["table_info"]["cell_order"] = []
        for row in preview.values.tolist():
            batch_exp["table_info"]["cell_order"] += row
    if not batch_exp["table_info"]["valid"]:
        st.error("Fix your info before moving on - or we will use auto layout.")
    return batch_exp


def manual_table_info_selection(exp, exp_serial_num, batch_exp, batch_exp_serial_num, skip_rename, skip_summary_table):
    st.subheader("Manual table information:")
    total_len = len(batch_exp["wells"])
    col_options = [i for i in range(2, int(np.ceil(total_len / 2))) if total_len % i == 0]
    if not col_options:
        st.error("The number of wells couldn't fit in any size table, all cells of the table must be full.")
        batch_exp["table_info"] = {}
        return batch_exp
    if batch_exp["table_info"] == {}:
        batch_exp["table_info"] = {"column_treatments": [], "column_names": [], "row_treatments": [], "row_names": [],
                                   "cell_names": [well[-3:] for well in batch_exp["wells"]], "treatments": [],
                                   "valid": True}
        batch_exp["table_info"]["cell_order"] = batch_exp["table_info"]["cell_names"]
        batch_exp["table_info"]["cell_order"].sort()
    elif len(batch_exp["table_info"]["column_names"]) not in col_options:
        batch_exp["table_info"] = {"column_treatments": [], "column_names": [], "row_treatments": [], "row_names": [],
                                   "cell_names": [well[-3:] for well in batch_exp["wells"]], "treatments": [],
                                   "valid": True}
        batch_exp["table_info"]["cell_order"] = batch_exp["table_info"]["cell_names"]
        batch_exp["table_info"]["cell_order"].sort()
    col_num = st.selectbox("How many columns should the table have?", col_options,
                           index=col_options.index(max(len(batch_exp["table_info"]["column_names"]), min(col_options))),
                           key="columns%d%d" % (exp_serial_num, batch_exp_serial_num))
    row_num = total_len // col_num
    st.info("Table will have %d columns, %d rows." % (col_num, row_num))
    st.subheader("Table design")
    if len(batch_exp["table_info"]["column_names"]) != col_num:
        batch_exp["table_info"]["column_names"] = ["col%d" % i for i in range(1, col_num+1)]
    if len(batch_exp["table_info"]["row_names"]) != row_num:
        batch_exp["table_info"]["row_names"] = ["row%d" % i for i in range(1, row_num+1)]
    cols = st.beta_columns(col_num + 1)
    for k in range(col_num + 1):
        with cols[k]:
            if k == 0:
                st.text_input("")
                for j in range(row_num):
                    batch_exp["table_info"]["row_names"][j] = st.text_input("",
                                                                            value=batch_exp["table_info"]["row_names"][j],
                                                                            key="row%d%d%d" % (exp_serial_num,
                                                                                               batch_exp_serial_num, j))
            else:
                i = k - 1
                batch_exp["table_info"]["column_names"][i] = st.text_input("",
                                                                           value=batch_exp["table_info"]["column_names"][i],
                                                                           key="column%d%d%d" % (exp_serial_num,
                                                                                                 batch_exp_serial_num, i))
                for j in range(row_num):
                    batch_exp["table_info"]["cell_order"][i + (j * col_num)] = \
                        st.text_input("", value=batch_exp["table_info"]["cell_order"][i + (j * col_num)], max_chars=3,
                                      key="cell%d%d%d%d" % (exp_serial_num, batch_exp_serial_num, i, j)).upper()
    batch_exp["table_info"]["valid"] = True
    for cell in batch_exp["table_info"]["cell_order"]:
        if cell not in batch_exp["table_info"]["cell_names"]:
            st.warning("Invalid entry - well %s does not exist in experiment" % cell)
            break
            batch_exp["table_info"]["valid"] = False
        elif len(cell) < 3:
            st.warning("Invalid entry - every cell must be 3 characters long (B03, etc.)")
            break
            batch_exp["table_info"]["valid"] = False
        elif batch_exp["table_info"]["cell_order"].count(cell) > 1:
            st.warning("Invalid entry - cell %s was entered more than once" % cell)
            break
            batch_exp["table_info"]["valid"] = False
    if not batch_exp["table_info"]["valid"]:
        st.error("Fix your info before moving on - or we will use auto layout.")
    return batch_exp


def batch_graph_selection(exp, exp_serial_num, batch_exp, batch_exp_serial_num):
    st.subheader("Graph selection:")
    graph_options = GRAPH_OPTIONS
    if not exp["scratch"]:
        graph_options = [option for option in graph_options if "layer" not in option]
    all_fields = DESIRED_FIELDS
    if not exp["get_intensity"]:
        all_fields = [field for field in all_fields if "Intensity" not in field]
    all_fields.sort()
    if batch_exp["param_graphs"] == {} or batch_exp["param_pair_graphs"] == {}:
        for i in range(len(all_fields)):
            param = all_fields[i]
            batch_exp["param_graphs"][param] = []
            for param2 in all_fields[i+1:]:
                param_pair = param + "_+_" + param2
                batch_exp["param_pair_graphs"][param_pair] = []
    for graph_type in graph_options:
        current_params = [k for k in batch_exp["param_graphs"].keys() if graph_type in batch_exp["param_graphs"][k]]
        desired_params = st.multiselect("Which parameters do you want a %s graph for?" % graph_type, all_fields,
                                        default=current_params,
                                        key="%s%d%d" % (graph_type, exp_serial_num, batch_exp_serial_num))
        new_params = [v for v in desired_params if v not in current_params]
        for param in new_params:
            batch_exp["param_graphs"][param].append(graph_type)
            batch_exp["param_graphs"][param] = list(set(batch_exp["param_graphs"][param]))
    current_param_pairs = [k.split("_+_") for k in batch_exp["param_pair_graphs"].keys()
                           if batch_exp["param_pair_graphs"][k] != []]
    for i in range(len(current_param_pairs) + 1):
        if i == len(current_param_pairs):
            param1_index = 0
            param2_index = 1
            absolute_val = False
            add_val = False
        else:
            param1_index = all_fields.index(current_param_pairs[i][0])
            param2_index = all_fields.index(current_param_pairs[i][1])
            absolute_val = "absolute" in batch_exp["param_pair_graphs"]["_+_".join(current_param_pairs[i])]
            add_val = True
        param1 = st.selectbox("Which parameter do you want to compare?", all_fields, index=param1_index,
                              key="param1%d%d%d" % (exp_serial_num, batch_exp_serial_num, i))
        param2 = st.selectbox("What do you want to compare it with?", [f for f in all_fields if f != param1],
                              index=param2_index - 1, key="param2%d%d%d" % (exp_serial_num, batch_exp_serial_num, i))
        absolute = st.checkbox("Do you want to compare the absolute values as well?", value=absolute_val,
                               key="abs%d%d%d" % (exp_serial_num, batch_exp_serial_num, i))
        add = st.checkbox("Is this parameter pair ready to be added to graph list?", value=add_val,
                          key="add%d%d%d" % (exp_serial_num, batch_exp_serial_num, i))
        if add:
            params = [param1, param2]
            params.sort()
            param_pair = "_+_".join(params)
            batch_exp["param_pair_graphs"][param_pair] = ["average"]
            if absolute:
                batch_exp["param_pair_graphs"][param_pair].append("absolute")
    return batch_exp


def output_batch_subsection(exp, exp_serial_num, batch_exp, batch_exp_serial_num, skip_rename, skip_summary_table):
    st.subheader("Advanced output editing for batch experiment #%d: %s, %s" % (batch_exp_serial_num, batch_exp["exp_name"],
                                                                       batch_exp["exp_sub_name"]))
    layout_options = ["auto", "auto_table", "manual_table"]
    batch_exp["design"] = st.radio("How should the report's layout be designed?", layout_options,
                                   index=layout_options.index(batch_exp["design"]),
                                   key="design%d%d" % (exp_serial_num, batch_exp_serial_num))
    if batch_exp["design"] == "auto_table":
        batch_exp = auto_table_info_selection(exp, exp_serial_num, batch_exp, batch_exp_serial_num,
                                              skip_rename, skip_summary_table)
    if batch_exp["design"] == "manual_table":
        batch_exp = manual_table_info_selection(exp, exp_serial_num, batch_exp, batch_exp_serial_num,
                                                skip_rename, skip_summary_table)
    batch_exp["manual_graph_choice"] = st.checkbox("Do you want to manually select graphs for the report?",
                                                   value=batch_exp["manual_graph_choice"],
                                                   key="manual_graph%d%d" % (exp_serial_num, batch_exp_serial_num))
    if batch_exp["manual_graph_choice"]:
        batch_exp = batch_graph_selection(exp, exp_serial_num, batch_exp, batch_exp_serial_num)
    return batch_exp


def output_section(exp, exp_serial_num, skip_rename, skip_summary_table):
    st.header("Advanced output info for experiment #%d" % exp_serial_num)
    st.text("This experiment's output will be in %s" % exp["batch_path"])
    exp["output_edit"] = st.checkbox("Would you like to edit the output for the report?",
                                     value=exp["output_edit"], key="output_edit%d" % exp_serial_num)
    if exp["output_edit"]:
        exp["batch_experiments_info"] = [output_batch_subsection(exp, exp_serial_num, batch_exp,
                                                                 exp["batch_experiments_info"].index(batch_exp) + 1,
                                                                 skip_rename, skip_summary_table)
                                         for batch_exp in exp["batch_experiments_info"]]
    return exp


def pybatch_sum2pd(paths, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    summary_df = pd.concat([pd.read_excel(path) for path in paths], ignore_index=True)
    assert len(summary_df) > 0, "Summary table is empty. Please check your source file and try again."
    summary_df.drop(columns=["Unnamed: 0", "Parent_OLD"], inplace=True, errors="ignore")
    summary_df.to_pickle(os.path.join(output_path, "rawdatagraph.pickle"))
    summary_df.drop(columns=["TimeIndex", "x_Pos", "y_Pos", "z_Pos", "Parent", "dt", "ID"], inplace=True,
                    errors="ignore")
    summary_df.to_pickle(os.path.join(output_path, "rawdata.pickle"))
    return "DONE"


def rename_page(state):
    st.header("Renaming stage")
    state.skip_rename = st.checkbox("Skip rename stage (select this if you already have renamed excel files)", value=state.skip_rename)
    if state.skip_rename:
        st.warning("If you have a summary table ready, skip to the next page and select \"skip summary table stage\" as well.")
        if st.button("Next stage"):
            state.stage = 2
        state.experiments = skipped_rename_section(state.experiments, 1)
    else:
        state.experiments = renaming_section(state.experiments, 1)
    if None not in state.experiments:
        all_rename_dirs = [exp["rename_dir"] for exp in state.experiments]
        if len(set(all_rename_dirs)) != len(all_rename_dirs):
            for rename_dir in set(all_rename_dirs):
                if all_rename_dirs.count(rename_dir) > 1:
                    st.error("You entered the rename dir %s for more than one experiment." % rename_dir)
        else:
            if st.button("Next stage", key="twice"):
                state.stage = 2
            st.success("You're all done here, you may move on to the next section or click the box to add an experiment")


def summary_table_page(state):
    st.header("Summary table stage")
    if state.skip_rename:
        state.skip_summary_table = st.checkbox("Skip summary table stage (select this if you already have a summary table)", value=state.skip_summary_table)
        if state.skip_summary_table:
            state.experiments = skipped_summary_table_section(state.experiments, 1)
    if None in state.experiments and state.skip_summary_table == False:
        st.warning("Must have valid renaming info before you can start this stage, please go back or choose to skip both stages.")
    else:
        if state.skip_summary_table == False:
            state.experiments = [summary_table_section(exp, state.experiments.index(exp) + 1) for exp in state.experiments]
        if None not in state.experiments:
            if "" not in [exp["summary_table_path"] for exp in state.experiments]:
                success_msg = "You're all done here, you may move on to the next section"
                if state.skip_summary_table:
                    success_msg += " or click the box to add an experiment"
                if st.button("Next stage"):
                    state.stage = 3
                st.success(success_msg)


def batch_page(state):
    st.header("Batch calculations stage")
    if None in state.experiments:
        st.warning("Must have valid summary table info before you can start this stage, please go back.")
    else:
        state.experiments = [batch_section(exp, state.experiments.index(exp) + 1) for exp in state.experiments]
        if True not in [exp["batch_path"] == "" or None in exp["batch_experiments_info"] for exp in state.experiments]:
            if st.button("Next stage"):
                state.stage = 4
            if st.button("Run now", key="batch"):
                state.stage = 5
            st.success("You're all done here, click 'Next stage' to design your output or 'Run now' to use the "
                       "default design and go directly to the run page.")


def output_page(state):
    st.header("Advanced output design stage.")
    st.text("Make sure you know what you're doing, or just go on to the run page.")
    if None in state.experiments:
        st.warning("Must have valid summary table info before you can start this stage, please go back.")
    elif True in [exp["batch_path"] == "" or None in exp["batch_experiments_info"] for exp in state.experiments]:
        st.warning("Must have valid batch calculation info before you can start this stage, please go back.")
    else:
        state.experiments = [output_section(exp, state.experiments.index(exp) + 1, state.skip_rename,
                                            state.skip_summary_table) for exp in state.experiments]
        ready = 0
        for exp in state.experiments:
            if not exp["output_edit"]:
                ready += 1
            else:
                exp_ready = 0
                for batch_exp in exp["batch_experiments_info"]:
                    if (batch_exp["design"] == "auto" or ("table" in batch_exp["design"]
                                                          and batch_exp["table_info"]["valid"]) and
                        (batch_exp["manual_graph_choice"] is False or (batch_exp["param_graphs"] is not {}
                                                                       or batch_exp["param_pair_graphs"] is not {}))):
                        exp_ready += 1
                if exp_ready == len(exp["batch_experiments_info"]):
                    ready += 1
        if ready == len(state.experiments):
            if st.button("Run now", key="output"):
                state.stage = 5
            st.success("You're all done here, click 'Run now' to go to the run page.")


def run_page(state, log_file, initial_session_id):
    whereami = 1
    if state.run == False:
        st.header("About to run, please double check all fields:")
    else:
        st.header("Running all stages, this will take a while.")
        if get_report_ctx().session_id != initial_session_id:
            st.warning("Whoops! Streamlit refreshed the session. Don't worry, all info was saved and we've picked up where we left off.")
            st.write("Already ran %d steps, so we'll start from step #%d" % (state.wherewasi, state.wherewasi + 1))
    for exp in state.experiments:
        st.header("Experiment #%d:" % (state.experiments.index(exp) + 1))
        if state.skip_rename == False:
            st.write("Renaming:")
            st.write("Imaris dir: " + exp["imaris_dir"])
            st.write("Your renamed files will be in: " + exp["rename_dir"])
            if state.run:
                if whereami > state.wherewasi:
                    if not os.path.exists(exp["rename_dir"]):
                        os.mkdir(exp["rename_dir"])
                    renaming(exp["imaris_dir"], exp["renaming_info"], exp["rename_dir"])
                    st.success("Done renaming")
                    state.wherewasi = whereami
                    write_state(log_file, state)
                whereami += 1
        else:
            st.write("Skipped renaming.")
            st.write("Your renamed files are in: " + exp["rename_dir"])
        if exp["summary_table_path"] == "":
            if state.run == False:
                st.warning("The script will only run up to the renaming stage. If you wish to change that, go back to the summary table page.")
        else:
            if state.skip_summary_table == False:
                st.write("Summary table creation:")
                st.write("Your summary table will be in: " + exp["summary_table_path"])
                if state.run == False:
                    exp["drop_report_path"] = get_file_path("drop report", "pdf", new_file=True, key=state.experiments.index(exp), value=exp["drop_report_path"])
                    st.write("(You can change the drop report path)")
                if state.run:
                    imaris_version, dimensions, dt, whereami = create_summary_table(exp["rename_dir"], os.path.join(exp["summary_table_path"], "summary_table.xlsx"), exp["drop_report_path"],
                                                                                                     initial_session_id, state, whereami, get_intensity=exp["get_intensity"])
                    if [exp["imaris_version"], exp["dimensions"], exp["dt"]] == [0,0,0]:
                        exp["imaris_version"] = int(imaris_version)
                        exp["dimensions"] = int(dimensions)
                        exp["dt"] = int(dt)
                    st.success("Done creating the summary table.\nYour experiment was run on Imaris %d, has %d "
                               "dimensions and a dt of %d minutes"
                               % (exp["imaris_version"], exp["dimensions"], exp["dt"]))
                    st.write("Saved drop report to " + exp["drop_report_path"])
            else:
                st.write("Skipped summary table creation")
                st.write("Your summary table is in: " + os.path.join(exp["summary_table_path"], "summary_table.xlsx"))
            if exp["batch_path"] == "":
                if state.run == False:
                    st.warning("The script will only run up to the summary table creation stage. If you wish to change that, go back to the batch page.")
            else:
                st.write("Batch calculations:")
                st.write("Your batch analysis results will be in: " + exp["batch_path"])
                st.write("The batch experiment names are: " + ", ".join([batch_exp_info["exp_name"] for batch_exp_info in exp["batch_experiments_info"]]))
                if state.run:
                    for batch_exp_info in exp["batch_experiments_info"]:
                        if whereami > state.wherewasi:
                            with st.spinner("Running batch calculations and creating graphs and reports for %s..." % batch_exp_info["exp_name"]):
                                if exp["full"] == True:
                                    if state.skip_summary_table:
                                        summary_table_excel = os.path.join(exp["summary_table_path"],
                                                                           [f for f in os.listdir(exp["summary_table_path"]) if "summary_table" in f and "_FULL" in f][0])
                                        summary_table = pd.read_excel(summary_table_excel, nrows=1)
                                        exp["dt"] = int(summary_table.dt[0])
                                else:
                                    summary_table_excel = os.path.join(exp["summary_table_path"], "summary_table.xlsx")
                                    summary_table = pd.read_excel(summary_table_excel)
                                    summary_table.drop(columns="Unnamed: 0", inplace=True)
                                    if state.skip_summary_table:
                                        exp["dt"] = int(summary_table.dt[0])
                                batch_exp = Batch_Experiment(batch_exp_info["exp_name"], batch_exp_info["exp_sub_name"],
                                                             batch_exp_info["wells"], exp["scratch"],
                                                             batch_exp_info["protocol_file"],
                                                             batch_exp_info["incucyte_files"],
                                                             batch_exp_info["imaris_xls_files"], dt=exp["dt"],
                                                             design=batch_exp_info["design"],
                                                             table_info=batch_exp_info["table_info"],
                                                             exp_rename_dir=exp["rename_dir"])
                                if exp["full"] == True:
                                    batch_exp.load_wells_from_summary_folder(exp["summary_table_path"])
                                else:
                                    batch_exp.load_wells_from_summary_table(summary_table_excel)
                                if batch_exp_info["manual_graph_choice"]:
                                    param_pair_graphs = {tuple(k.split("_+_")):
                                                             batch_exp_info["param_pair_graphs"][k]
                                                         for k in batch_exp_info["param_pair_graphs"].keys()
                                                         if batch_exp_info["param_pair_graphs"][k] != []}
                                    batch_exp.create_output(exp["batch_path"],
                                                            param_graphs=batch_exp_info["param_graphs"],
                                                            param_pair_graphs=param_pair_graphs)
                                else:
                                    batch_exp.create_output(exp["batch_path"])
                            state.wherewasi = whereami
                            write_state(log_file, state)
                        st.success("Done with %s!" % batch_exp_info["exp_name"])
                        whereami += 1
                    st.success("Done creating graphs!")
    if state.run == False:
        if len(state.experiments) > 1 and "" not in [exp["summary_table_path"] for exp in state.experiments]:
            st.subheader("Pickles for TASC")
            st.text("We will automatically create pickles for each experiment, and you can always create pickles later "
                    "with summary_table_creator_tool.ipynb")
            state.create_one_pickle = st.checkbox("Would you like to create unified TASC pickles for all experiments "
                                                  "in this run?", value=state.create_one_pickle)
            if state.create_one_pickle:
                if state.pickle_path == "":
                    state.pickle_path = os.path.join(os.path.commonpath([exp["summary_table_path"]
                                                                         for exp in state.experiments]),
                                                     "unified_TASC_pickles")
                state.pickle_path = get_folder("Enter path for unified TASC pickles:", "empty", value=state.pickle_path)
        st.success("If you have gone over everything and it seems right, please click here:")
        state.run = st.button("Run now")
    else:
        if state.create_one_pickle:
            pybatch_sum2pd([os.path.join(exp["summary_table_path"], "summary_table.xlsx") for exp in state.experiments],
                           state.pickle_path)
        st.success("ALL DONE. Feel free to review everything, and simply exit the browser when you're ready. Everything should be saved.")
        print("======================================================\nDONE")
        st.balloons()
        if st.button("DONE"):
            return


def main(log_file, state, initial_session_id):
    old_state = deepcopy(state)
    st.header("Running PyBatch")
    if state.run == False:
        st.sidebar.title("Navigation")
    if state.stage != 5:
        state.stage = PAGES[:-1].index(st.sidebar.radio("Enter parameters for:", PAGES[:-1], index=state.stage - 1)) + 1
        if None not in state.experiments and (state.skip_rename == False or "" not in [exp["summary_table_path"] for exp in state.experiments]):
            if st.sidebar.button("Run now"):
                state.stage = 5
        if state.stage == 1:
            rename_page(state)
        elif state.stage == 2:
            summary_table_page(state)
        elif state.stage == 3:
            batch_page(state)
        elif state.stage == 4:
            output_page(state)
    elif state.stage == 5:
        if state.run == False:
            state.stage = PAGES.index(st.sidebar.radio("Enter parameters for:", PAGES, index=state.stage - 1)) + 1
        run_page(state, log_file, initial_session_id)
    if state != old_state:
        write_state(log_file, state)
        session = _get_session()
        session.request_rerun()

run_start_time, initial_session_id = get_time_and_session_id()
log_file = os.path.join(LOG_PATH, time.strftime("%Y%m%d%H%M%S.log", run_start_time))
if os.path.exists(log_file):
    state = read_state(log_file)
else:
    state = Pybatch_State()
main(log_file, state, initial_session_id)