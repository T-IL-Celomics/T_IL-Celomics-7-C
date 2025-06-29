import pandas as pd
import os
import xlrd
from shutil import copyfile
import streamlit as st
import re
from collections import OrderedDict
import math
import time
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np


FIRST_PARENT = 10 ** 9

PARENT_SHEETS = ["TrackVolumeMean", "TrackDisplacementLength"]
PARENT_ZIR_SHEETS = [["Track", "Displacement"]] #can delete
ID_SHEETS = ["Area", "Displacement2", "Ellipticity_oblate", "Ellipticity_prolate", "Sphericity"]
ELLIP_LEN_SHEET = "EllipsoidAxisLength"
INTENSITY_SHEETS = ["IntensityCenterCh%d", "IntensityMaxCh%d", "IntensityMeanCh%d", "IntensityMedianCh%d", "IntensitySumCh%d"]

def msd_brownian_motion_func(x, D):
    return 4 * D * x

def msd_directed_motion_func(x, D, v):
    return (4 * D * x) + ((v ** 2) * (x ** 2))

def msd_exponent_func(x, D, exp):
    return D * (x ** exp)


class Cell_Info(object):

    def __init__(self, imaris_df, parent_id, dimensions, dt, imaris_file, skip_limit, get_intensity=False):
        """
        Gets basic info regarding the Parent Cell.
        """
        self.imaris_df = imaris_df
        self.dimensions = dimensions
        self.Experiment = imaris_file
        self.axes = ["A", "B", "C"][-dimensions:]
        self.zirim = ["X", "Y", "Z"][:dimensions]
        self.id_nums = list(imaris_df["Area"][imaris_df["Area"]["Parent"] == parent_id].index)
        self.Parent = parent_id
        self.Parent_OLD = parent_id + 1 - FIRST_PARENT
        self.dt = dt
        self.skip_limit = skip_limit
        self.intensity = get_intensity

    def keys(self):
        return self.__dict__.keys()

    def _set_value_from_sheet(self, sheet, id_num):
        self.__setattr__(sheet, self.imaris_df[sheet].loc[id_num][0])

    def _set_zir_values_from_sheet(self, sheet_words, id_num): #can delete
        sheet_list = [sheet_words+[zir] for zir in self.zirim]
        for sheet in sheet_list:
            self.__setattr__("_".join(sheet), self.imaris_df["".join(sheet[:-1])].loc[id_num, " ".join(sheet)])

    def _get_average_value(self, attribute):
        attribute_list = [id_info.__getattribute__(attribute) for id_info in self.info_per_id if attribute in id_info.keys()]
        try:
            return sum(attribute_list) / len(attribute_list)
        except ZeroDivisionError:
            return None

    def _calculate_BIC(self, x, y_obs, func, func_args):
        n = len(x)
        rss = sum([(y_obs[i] - func(x[i], *func_args)) ** 2 for i in range(n)])
        bic = n * math.log(rss/n) + len(func_args) * math.log(n)
        return bic

    def _get_MSD(self, tau, id_list):
        max_index = len(id_list)
        delta_squared_list = []
        for first_index in range(max_index - 1):
            first_id = id_list[first_index]
            for second_index in range(first_index + 1, max_index):
                second_id = id_list[second_index]
                if second_id.TimeIndex - first_id.TimeIndex == tau:
                    delta_squared_sum = sum([(first_id.positions[i] - second_id.positions[i]) ** 2 for i in range(self.dimensions)])
                    delta_squared_list.append(delta_squared_sum)
                    break
                elif second_id.TimeIndex - first_id.TimeIndex < tau:
                    continue
                else:
                    break
        if delta_squared_list == []:
            return None
        return sum(delta_squared_list) / len(delta_squared_list)

    def _make_MSD_calculations(self, id_list=None, test_mode=False, calculate_up_to_div=2):
        if id_list == None:    
            id_list = self.info_per_id.copy()
        MSDs = {}
        for tau in range(1, int((id_list[-1].TimeIndex - id_list[0].TimeIndex) / calculate_up_to_div) + 1):
            msd = self._get_MSD(tau, id_list)
            if msd != None:
                MSDs[tau] = msd
        if test_mode:
            return MSDs
        if len(MSDs) < 3 or 1 not in MSDs.keys():
            return
        self.Final_MSD_1 = id_list[-1].Current_MSD_1
        #self.MSD_OLD = sum(MSDs)
        self.MSDs = MSDs
        tau_values = list(MSDs.keys())
        msd_values = list(MSDs.values())
        max_tau = tau_values[-1]
        msd_slope, msd_intercept, r_value, p_value, stderr = linregress(x=tau_values, y=msd_values)
        self.MSD_Linearity_R2_Score = r_value ** 2
        try:
            (D,), pcov = curve_fit(msd_brownian_motion_func, tau_values, msd_values)
            self.MSD_Brownian_D = D
            self.MSD_Brownian_Motion_BIC_Score = self._calculate_BIC(tau_values, msd_values, msd_brownian_motion_func, [D])
        except:
            self.MSD_Brownian_D = 0
            self.MSD_Brownian_Motion_BIC_Score = 0
        try:
            (D, v), pcov = curve_fit(msd_directed_motion_func, tau_values, msd_values, bounds=[(0, 1), (np.inf, np.inf)])
            self.MSD_Directed_D = D
            self.MSD_Directed_v2 = v**2
            self.MSD_Directed_Motion_BIC_Score = self._calculate_BIC(tau_values, msd_values, msd_directed_motion_func, [D, v])
        except:
            self.MSD_Directed_D = 0
            self.MSD_Directed_v2 = 0
            self.MSD_Directed_Motion_BIC_Score = 0
        try:
            (D, exp), pcov = curve_fit(msd_exponent_func, tau_values, msd_values, bounds=[(0, 0), (np.inf, np.inf)])
            self.MSD_Constant_D = D
            self.MSD_Exponent = exp
        except:
            self.MSD_Constant_D = 0
            self.MSD_Exponent = 0

    def _get_half_velocity_index(self, max_index, half_max_avg_speed, avg_speeds, avg_indexes, speeds_we_averaged, window_size, direction):
        if direction == "Back":
            start = max_index - 1
            stop = -1
            jump = -1
        elif direction == "Forward":
            start = max_index + 1
            stop = len(avg_speeds)
            jump = 1
        for index in range(start, stop, jump):
            if avg_speeds[index] <= half_max_avg_speed:
                for i in range(window_size)[::jump]: # [::jump] reverses direction if direction is "Back"
                    if speeds_we_averaged[index][i] <= half_max_avg_speed:
                        return avg_indexes[index][i]
        return "Not Found"

    def _get_wave_info(self, window_size=3):
        all_speeds = [id_info.Instantaneous_Speed if "Instantaneous_Speed" in id_info.keys() else None for id_info in self.info_per_id]
        no_none_indexes = []
        no_none_speeds = []
        for index in range(len(all_speeds)):
            if all_speeds[index] != None:
                no_none_indexes.append(index)
                no_none_speeds.append(all_speeds[index])
        if len(no_none_speeds) < window_size:
            return
        avg_indexes = []
        avg_speeds = []
        speeds_we_averaged = []
        for i in range(len(no_none_speeds) - window_size + 1):
            speeds_to_average = no_none_speeds[i:i+window_size]
            avg_indexes.append(no_none_indexes[i:i+window_size])
            avg_speeds.append(sum(speeds_to_average) / window_size)
            speeds_we_averaged.append(speeds_to_average)
        max_avg = max(avg_speeds)
        if avg_speeds.count(max_avg) > 1:
            print("Max average speed appeared more than once, this shouldn't happen. Call Ori 0507479922")
        max_index = avg_speeds.index(max_avg)
        max_avg_indexes = avg_indexes[max_index]
        max_speed = max(speeds_we_averaged[max_index])
        max_speed_index = avg_indexes[max_index][speeds_we_averaged[max_index].index(max_speed)]
        half_max_avg_speed = max_avg / 2
        self.Velocity_Maximum_Height = max_speed
        self.Velocity_Time_of_Maximum_Height = self.info_per_id[max_speed_index].TimeIndex * self.dt
        t0_index = self._get_half_velocity_index(max_index, half_max_avg_speed, avg_speeds, avg_indexes, speeds_we_averaged, window_size, direction="Back")
        t1_index = self._get_half_velocity_index(max_index, half_max_avg_speed, avg_speeds, avg_indexes, speeds_we_averaged, window_size, direction="Forward")
        #fill in values normally if none are missing (wave reached half max on both sides)
        self.Velocity_Full_Width_Half_Maximum = 0
        if t0_index != "Not Found":
            self.Velocity_Starting_Value = self.info_per_id[t0_index].Instantaneous_Speed
            self.Velocity_Starting_Time = self.info_per_id[t0_index].TimeIndex * self.dt
            self.Velocity_Full_Width_Half_Maximum += self.Velocity_Time_of_Maximum_Height - self.Velocity_Starting_Time
            good_half_index = t0_index
        else:
            self.Velocity_Starting_Value = 0
            self.Velocity_Starting_Time = 0
        if t1_index != "Not Found":
            self.Velocity_Ending_Value = self.info_per_id[t1_index].Instantaneous_Speed
            self.Velocity_Ending_Time = self.info_per_id[t1_index].TimeIndex * self.dt
            self.Velocity_Full_Width_Half_Maximum += self.Velocity_Ending_Time - self.Velocity_Time_of_Maximum_Height
            good_half_index = t1_index
        else:
            self.Velocity_Ending_Value = 0
            self.Velocity_Ending_Time = 0
        """
        THIS PART "CREATES" DOUBLE SIDED WAVES! the "sections above and should be fixed if we return to forcing double sided waves
        #fill in all zeroes if half of max speed was never reached
        if missing == 2:
            self.Velocity_Starting_Value = 0
            self.Velocity_Starting_Time = 0
            self.Velocity_Ending_Value = 0
            self.Velocity_Ending_Time = 0
            self.Velocity_Full_Width_Half_Maximum = 0
        #fill in missing side as mirror of other side if only one half max speed was found
        elif missing == 1:
            if good_half_index > max_speed_index:
                self.Velocity_Starting_Time = self.Velocity_Time_of_Maximum_Height - (self.Velocity_Ending_Time - self.Velocity_Time_of_Maximum_Height)
                self.Velocity_Starting_Value = self.Velocity_Ending_Value
            else:
                self.Velocity_Ending_Time = self.Velocity_Time_of_Maximum_Height + (self.Velocity_Time_of_Maximum_Height - self.Velocity_Starting_Time)
                self.Velocity_Ending_Value = self.Velocity_Starting_Value
            self.Velocity_Full_Width_Half_Maximum = (self.Velocity_Ending_Time - self.Velocity_Time_of_Maximum_Height) * 2
        #calculate FWHM normally if both sides were found
        else:
            self.Velocity_Full_Width_Half_Maximum = self.Velocity_Ending_Time - self.Velocity_Starting_Time
            #self.Velocity_Full_Width_Half_Maximum_OLD = (self.Velocity_Ending_Time - self.Velocity_Starting_Time) / self.dt
        """

    def _calculate_distance(self, first_info, second_info):
        return sum([(first_info.positions[i] - second_info.positions[i]) ** 2 for i in range(self.dimensions)]) ** 0.5

    def get_parent_info(self):
        """
        Gets info from the constant sheets, plus extra fields requiring calculation which are entered manually.
        """
        try:
            for sheet in PARENT_SHEETS:
                self._set_value_from_sheet(sheet, self.Parent)
            for sheet in PARENT_ZIR_SHEETS:
                self._set_zir_values_from_sheet(sheet, self.Parent)
            #self.Track_Displacement_Length = self.imaris_df["TrackDisplacementLength"].loc[self.Parent, "Value"]
            #self.Track_Displacement_Length_OLD = sum([self.__getattribute__("Track_Displacement_" + zir)**2 for zir in self.zirim])**0.5 #can delete
        except Exception:
            print("Problem with cell %d" % self.Parent)
            raise

    def get_info_per_id(self):
        self.info_per_id = []
        #First run
        for id_num in self.id_nums:
            id_info = Id_Info(self, id_num)
            id_info.get_id_info()
            id_info.Current_MSD_1 = self._get_MSD(1, self.info_per_id.copy() + [id_info])
            self.info_per_id.append(id_info)


        #Second run for calculations that require next/all ids - NOT RUNNING NOW - RUN IF YOU NEED MEAN_SPEED
        #for i in range(1, len(self.info_per_id) - 1):
        #    self.info_per_id[i].get_secondary_info(self.info_per_id[i+1])

    def get_final_calculations(self):
        """
        Calculations run after all id_info is full.
        """
        #Displacement was calculated differently in matlab, it would've looked like this:
        #self.Displacement = sqrt(x_PosFinal.^2 + y_PosFinal.^2 + z_PosFinal.^2) - sqrt(x_PosInitial.^2 + y_PosInitial.^2 + z_PosInitial.^2) #not python, obviously
        self.Overall_Displacement = self._calculate_distance(self.info_per_id[-1], self.info_per_id[0]) #Net Displacement
        self.Total_Track_Displacement = sum([id_info.Displacement_From_Last_Id for id_info in self.info_per_id if "Displacement_From_Last_Id" in id_info.keys()])
        #self.Confinement_Ratio_OLD = self.Displacement / self.Track_Displacement_Length
        if self.Total_Track_Displacement == 0:
            self.Confinement_Ratio = np.nan
        else:
            self.Confinement_Ratio = self.Overall_Displacement / self.Total_Track_Displacement
        self.Mean_Curvilinear_Speed = self._get_average_value("Instantaneous_Speed")
        self.Mean_Straight_Line_Speed = self.Overall_Displacement / ((self.info_per_id[-1].TimeIndex - self.info_per_id[0].TimeIndex) * self.dt)
        try:
            self.Linearity_of_Forward_Progression = self.Mean_Straight_Line_Speed / self.Mean_Curvilinear_Speed
        except TypeError:
            pass
        self._make_MSD_calculations()
        self._get_wave_info()

    def return_info(self, desired_fields):
        single_skips = 0
        double_skips = 0
        id_dict_list = []
        parent_keys = [field for field in desired_fields if field in self.__dict__.keys()]
        parent_dict = {k: self.__getattribute__(k) for k in parent_keys}
        for id_info in self.info_per_id:
            if id_info.skip == 2:
                single_skips += 1
            elif id_info.skip == 3:
                double_skips += 1
            id_dict = parent_dict.copy()
            for k in id_info.__dict__.keys():
                if k in desired_fields and k not in parent_keys:
                    id_dict[k] = id_info.__getattribute__(k)
            id_dict_list.append(id_dict)
        return id_dict_list, single_skips, double_skips


class Id_Info(Cell_Info):

    def __init__(self, parent, id_num):
        """
        A class for a cell at a given timepoint, inherits all features from Parent.
        """
        self.ID = id_num
        self.imaris_df = parent.imaris_df
        self.dimensions = parent.dimensions
        self.axes = parent.axes
        self.zirim = parent.zirim
        self.dt = parent.dt
        self.skip_limit = parent.skip_limit
        self.TimeIndex = self.imaris_df["Area"].loc[self.ID, "Time"]
        self.cell_info = parent
        self.skip = 0
        self.intensity = parent.intensity
        try:
            self.last_id_info = parent.info_per_id[-1]
        except IndexError:
            self.last_id_info = False
            pass
        
    def _calculate_eccentricity(self, axis1, axis2):
        short_axis, long_axis = min([axis1, axis2]), max([axis1, axis2])
        return (1 - (short_axis/long_axis)**2)**0.5

    def _get_ellipsoid_info(self):
        try:
            for axis in self.axes:
                self._set_value_from_sheet(ELLIP_LEN_SHEET + axis, self.ID)
                for zir in self.zirim:
                    self.__setattr__("Ellip_Ax_%s_%s" % (axis, zir), self.imaris_df["EllipsoidAxis"+axis].loc[self.ID, "Ellipsoid Axis %s %s" % (axis, zir)])
        except KeyError:
            return
        #Eccentricity
        #self.Eccentricity_OLD = self._calculate_eccentricity(short_axis=(self.imaris_df[ELLIP_LEN_SHEET+"A"].loc[self.ID, "Value"] + self.__getattribute__(ELLIP_LEN_SHEET+"B")) / 2,
        #                                                      long_axis=self.__getattribute__(ELLIP_LEN_SHEET+"C"))
        if self.dimensions == 2:
            self.Eccentricity = self._calculate_eccentricity(self.__getattribute__(ELLIP_LEN_SHEET+"B"), self.__getattribute__(ELLIP_LEN_SHEET+"C"))
        elif self.dimensions == 3:
            self.Eccentricity_A = \
                self._calculate_eccentricity(self.__getattribute__(ELLIP_LEN_SHEET+"A"), 
                                             (self.__getattribute__(ELLIP_LEN_SHEET+"B") ** 2 + self.__getattribute__(ELLIP_LEN_SHEET+"C") ** 2) ** 0.5)
            self.Eccentricity_B = \
                self._calculate_eccentricity(self.__getattribute__(ELLIP_LEN_SHEET+"B"), 
                                             (self.__getattribute__(ELLIP_LEN_SHEET+"A") ** 2 + self.__getattribute__(ELLIP_LEN_SHEET+"C") ** 2) ** 0.5)
            self.Eccentricity_C = \
                self._calculate_eccentricity(self.__getattribute__(ELLIP_LEN_SHEET+"C"), 
                                             (self.__getattribute__(ELLIP_LEN_SHEET+"A") ** 2 + self.__getattribute__(ELLIP_LEN_SHEET+"B") ** 2) ** 0.5)

    def _get_intensity_info(self):
        for i in range(1, 4):
            try:
                for intensity_param in INTENSITY_SHEETS:
                    self._set_value_from_sheet(intensity_param % i, self.ID)
            except KeyError:
                return

    def _get_positions(self):
        self.positions = []
        for zir in self.zirim:
            pos = zir.lower() + "_Pos"
            position = self.imaris_df["Position"].loc[self.ID, "Position " + zir]
            self.__setattr__(pos, position)
            self.positions.append(position)

    def _get_velocity_acceleration_speed(self):
        time_delta = (self.TimeIndex - self.last_id_info.TimeIndex) * self.dt / 60 #hours
        self.velocities = []
        for zir in self.zirim:
            pos = zir.lower() + "_Pos"
        #Velocity
            try: #This will only work if the previous id had ALL the necessary info. It does not work with partial info or skip info-less ids.
                vel = "Velocity_"+zir
                velocity = (self.__getattribute__(pos) - self.last_id_info.__getattribute__(pos)) / time_delta
                self.__setattr__(vel, velocity)
                self.velocities.append(velocity)
        #Acceleration
                accel = "Acceleration_"+zir                                
                self.__setattr__(accel, (self.__getattribute__(vel) - self.last_id_info.__getattribute__(vel)) / time_delta)
            except AttributeError: #no velocity or position info on previous cell. 
                pass
        try:
            self.Instantaneous_Speed = sum([vel ** 2 for vel in self.velocities]) ** 0.5 # USED TO BE CALLED VELOCITY
            self.Acceleration = (self.Instantaneous_Speed - self.last_id_info.Instantaneous_Speed) / time_delta 
            self.Acceleration_OLD = sum([self.__getattribute__("Acceleration_" + zir) ** 2 for zir in self.zirim]) ** 0.5
        except AttributeError: #missing info regarding velocity or acceleration.
            pass
        if self.last_id_info.last_id_info and self.last_id_info.skip <= self.skip_limit:
            self.Instantaneous_Speed_OLD = sum([(self.positions[i] - self.last_id_info.last_id_info.positions[i]) ** 2 for i in range(self.dimensions)]) ** 0.5

    def _get_inst_angle(self):
        if self.dimensions == 2:
            self.Instantaneous_Angle = math.atan2(self.y_Pos - self.last_id_info.y_Pos, self.x_Pos - self.last_id_info.x_Pos)
        elif self.dimensions == 3:
            self.Instantaneous_Angle_OLD = math.atan2(self.y_Pos - self.last_id_info.y_Pos, self.x_Pos - self.last_id_info.x_Pos)
            self.Instantaneous_Angle_X = \
                math.atan2(((self.y_Pos - self.last_id_info.y_Pos) ** 2 + (self.z_Pos - self.last_id_info.z_Pos) ** 2) ** 0.5, (self.x_Pos - self.last_id_info.x_Pos))
            self.Instantaneous_Angle_Y = \
                math.atan2(((self.x_Pos - self.last_id_info.x_Pos) ** 2 + (self.z_Pos - self.last_id_info.z_Pos) ** 2) ** 0.5, (self.y_Pos - self.last_id_info.y_Pos))
            self.Instantaneous_Angle_Z = \
                math.atan2(((self.x_Pos - self.last_id_info.x_Pos) ** 2 + (self.y_Pos - self.last_id_info.y_Pos) ** 2) ** 0.5, (self.z_Pos - self.last_id_info.z_Pos))

    def _get_directional_change(self):
        if self.dimensions == 2:
            self.Directional_Change = self.Instantaneous_Angle - self.last_id_info.Instantaneous_Angle
        elif self.dimensions == 3:
            self.Directional_Change_OLD = self.Instantaneous_Angle_OLD - self.last_id_info.Instantaneous_Angle_OLD
            self.Directional_Change_X = self.Instantaneous_Angle_X - self.last_id_info.Instantaneous_Angle_X
            self.Directional_Change_Y = self.Instantaneous_Angle_Y - self.last_id_info.Instantaneous_Angle_Y
            self.Directional_Change_Z = self.Instantaneous_Angle_Z - self.last_id_info.Instantaneous_Angle_Z

    def get_id_info(self):
        """
        Gets info from the constant sheets, plus extra fields requiring calculation which are entered manually.
        """
        skip_ellipticity = False
        for sheet in ID_SHEETS:
            try:
                self._set_value_from_sheet(sheet, self.ID)
            except KeyError:
                if "Ellipticity" not in sheet:
                    print(self.ID, "was missing", sheet)
                else:
                    skip_ellipticity = True
                pass
        if not skip_ellipticity:
            self._get_ellipsoid_info()
        if self.intensity:
            self._get_intensity_info()
        self._get_positions()
        if self.last_id_info:
            self.Displacement_From_Last_Id = self._calculate_distance(self.last_id_info, self)
            self.skip = self.TimeIndex - self.last_id_info.TimeIndex
            if self.skip <= self.skip_limit:
                self._get_velocity_acceleration_speed()
                try:
                    self._get_inst_angle()
                    self._get_directional_change()
                except AttributeError:
                    pass

    def _get_mean_speed(self, next_id_info):
        # Can improve maybe by making this a weighted average of the deltas from current time index
        time_delta = (next_id_info.TimeIndex - self.last_id_info.TimeIndex) * self.dt / 60 
        distance = self._calculate_distance(next_id_info, self.last_id_info)
        mean_speed = distance / time_delta
        return mean_speed

    def get_secondary_info(self, next_id_info):
        """
        Gets info that is based on the all id info. Currently is run on every id starting from the second position (1) and ending with the second to last position
        """
        if self.skip <= self.skip_limit and next_id_info.TimeIndex - self.TimeIndex <= self.skip_limit:
            self.Mean_Speed = self._get_mean_speed(next_id_info)

    def get_min_distance(self, same_time_ids):
        distances = [self._calculate_distance(self, id_info) for id_info in same_time_ids]
        self.Min_Distance = min(distances)

    def get_collectivity(self, same_time_ids, radius=200, cube=False):
        if cube == False:
            relevant_ids = [id_info for id_info in same_time_ids if self._calculate_distance(self, id_info) <= radius]
            attribute = "Coll"
        elif cube == True:
            relevant_ids = [id_info for id_info in same_time_ids if False not in [abs(id_info.positions[i] - self.positions[i]) <= radius for i in range(self.dimensions)]]
            attribute = "Coll_CUBE"
        if relevant_ids == []:
            self.__setattr__(attribute, 0)
            return
        cosang_sum = 0
        amount = 0
        for id_info in relevant_ids:
            try:
                top_part = sum([(self.velocities[i] * id_info.velocities[i]) for i in range(self.dimensions)])
                bottom_part = (sum([self.velocities[i] ** 2 for i in range(self.dimensions)]) *\
                               sum([id_info.velocities[i] ** 2 for i in range(self.dimensions)])) ** 0.5
                cosang = top_part / bottom_part
                cosang_sum += cosang
                amount += 1
            except Exception:
                pass
        try:
            cosang_avg = cosang_sum / amount
            self.__setattr__(attribute, cosang_avg)
        except ZeroDivisionError:
            self.__setattr__(attribute, 0)
            return
        if self.__getattribute__(attribute) == 0:
            raise Exception("Collectivity came out zero - this shouldn't happen. Call Ori 0507479922")

    def return_all(self):
        to_return = self.cell_info.__dict__.copy()
        to_return.update(self.__dict__)
        return to_return

