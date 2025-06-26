import pandas as pd

FIRST_PARENT = 10 ** 9


MATLAB_PATH = r"\\metlab24\d\orimosko\yossi_chemo_exps\EXP120\results\Cluster Analysis\SummaryTables\all\Summary Table Single Cell all for graphs.xlsx"
RENAME_PATH = r"\\metlab24\d\orimosko\yossi_chemo_exps\EXP120\renaming\YL120031215CHR1B02SK00CON1NNN0NNN0NNN0NNN0WH00.xls"
EXP = "YL120031215CHR1B02SK00CON1NNN0NNN0NNN0NNN0WH00"
COMPARE_FIELDs = [("Area"), ("Acceleration"), ("Coll", "Coll_CUBE"), ("Confinement_Ratio"), ("Directional_Change"), ("Ellip_Ax_B_X"), ("Ellip_Ax_B_Y"), ("Ellip_Ax_C_X"), ("Ellip_Ax_C_Y"), ("EllipsoidAxisLengthB"), ("EllipsoidAxisLengthC"), ("Ellipticity_oblate"), ("Ellipticity_prolate"), ("Instantaneous_Angle"), ("Instantaneous_Speed"), ("Linearity_of_Forward_Progression"), ("Mean_Curvilinear_Speed"), ("Mean_Straight_Line_Speed"), ("MSD"), ("Sphericity"), ("Track_Displacement_Length"), ("Track_Displacement_X"), ("Track_Displacement_Y"), ("Velocity"), ("Velocity_X"), ("Velocity_Y"), ("Eccentricity"), ("Velocity_Full_Width_Half_Maximum"), ("Velocity_Time_of_Maximum_Height"), ("Velocity_Maximum_Height"), ("Velocity_Ending_Value"), ("Velocity_Ending_Time"), ("Velocity_Starting_Value"), ("Velocity_Starting_Time"), ("TimeIndex"), ("x_Pos"), ("y_Pos"), ("Parent"), ("dt"), ("ID"), ("Experiment")]

def matlab_opener(path, exp):
	full_excel = pd.read_excel(path)
	exp_excel = full_excel[full_excel["Experiment"] == exp]
	return exp_excel


def pybatch_runner(path, exp, version=7, dimensions=2, dt=45):
	imaris_df = get_imaris_df(path, version)
	py_df = get_info_from_imaris(exp, imaris_df, dimensions, dt)
	#py_df = pd.DataFrame(imaris_info)
	#py_df.dropna(inplace=True)
	#py_df.reset_index(inplace=True, drop=True)
	return imaris_info


def both_finder(mat_df, py_df):
	#Create empty dfs with correct headers
	only_pybatch = py_df[py_df["Parent"] == 0]
	only_matlab = mat_df[mat_df["Parent"] == 0]
	both = {"pybatch": only_pybatch, "matlab": only_matlab}
	for p in set(list(py_df["Parent_OLD"]) + list(mat_df["Parent"])):
		mat_parent = mat_df[mat_df["Parent"] == p]
		py_parent = py_df[py_df["Parent_OLD"] == p]
		if mat_parent.empty and py_parent.empty:
			raise Exception("Parent didn't actually exist, something is wrong.")
		if mat_parent.empty:
			only_pybatch = pd.concat([only_pybatch, py_parent]).reset_index(drop=True)
		elif py_parent.empty:
			only_matlab = pd.concat([only_matlab, mat_parent]).reset_index(drop=True)
		else:
			py_times = list(py_parent["TimeIndex"])
			mat_times = [t+1 for t in mat_parent["TimeIndex"]]
			for t in set(py_times + mat_times):
				if t not in py_times:
					only_matlab = pd.concat([only_matlab, mat_parent[mat_parent["TimeIndex"] == t-1]]).reset_index(drop=True)
				elif t not in mat_times:
					only_pybatch = pd.concat([only_pybatch, py_parent[py_parent["TimeIndex"] == t]]).reset_index(drop=True)
				else:
					both["pybatch"] = pd.concat([both["pybatch"], py_parent[py_parent["TimeIndex"] == t]]).reset_index(drop=True)
					both["matlab"] = pd.concat([both["matlab"], mat_parent[mat_parent["TimeIndex"] == t-1]]).reset_index(drop=True)
	print("Found %d values unique to matlab, %d values unique to pybatch and %d shared values" % (len(only_matlab), len(only_pybatch), len(both["pybatch"])))
	return only_pybatch, only_matlab, both


def full_list_creator(rename_path):
	sheet = pd.read_excel(rename_path, sheet_name="Time Index")
	full_list = [(sheet.loc[i].Parent, sheet.loc[i].Time) for i in range(len(sheet))]
	return full_list


def both_and_niether_finder(mat_df, py_df, full_list):
	only_pybatch = py_df[py_df["Parent"] == 0]
	only_matlab = mat_df[mat_df["Parent"] == 0]
	both = {"pybatch": only_pybatch, "matlab": only_matlab}
	niether = []
	for tup in full_list:
		parent, timeindex = tup
		mat_row = mat_df[mat_df.Parent == parent + 1 - FIRST_PARENT][mat_df.TimeIndex == timeindex - 1]
		py_row = py_df[py_df.Parent == parent][py_df.TimeIndex == timeindex]
		if len(mat_row) == 1 and len(py_row) == 1:
			both["pybatch"] = pd.concat([both["pybatch"], py_row])
			both["matlab"] = pd.concat([both["matlab"], mat_row])
		elif len(mat_row) == 1 and len(py_row) == 0:
			only_matlab = pd.concat([only_matlab, mat_row])
		elif len(mat_row) == 0 and len(py_row) == 1:
			only_pybatch = pd.concat([only_pybatch, py_row])
		elif len(mat_row) == 0 and len(py_row) == 0:
			niether.append(tup)
		else:
			print("SOMETHING UNEXPECTED HAPPENED")
	return only_pybatch, only_matlab, both, niether


def compare_field(py_df, mat_df, mat_field, alt_name=None):
	print("Working on", mat_field)
	diffs = []
	if len(py_df) != len(mat_df):
		raise Exception("Can't compare dataframes of different length")
	else:
		df_len = len(py_df)
	if alt_name != None:
		if alt_name == "_OLD":
			py_field = mat_field + "_OLD"
		py_field = alt_name
	else:
		py_field = mat_field
	for i in range(df_len):
		diffs.append(abs(py_df.loc[i][py_field] - mat_df.loc[i][mat_field]))
	avg_diff = sum(diffs) / len(diffs)
	print("The largest diff was %f, and the average was %f\n\n" % (max(diffs), avg_diff))
	return diffs


def create_compare_lists(mat_path=MATLAB_PATH, rename_path=RENAME_PATH, exp=EXP):
	print("Loading matlab results...")
	mat_df = matlab_opener(mat_path, exp)
	print("Running pybatch...")
	py_df = load_single_pybatch_df(rename_path)
	print("Finding values from both dataframes...")
	only_pybatch, only_matlab, both = both_finder(mat_df, py_df)
	print("Ready to compare fields!")
	return only_pybatch, only_matlab, both


def create_full_compare_lists(mat_path=MATLAB_PATH, rename_path=RENAME_PATH, exp=EXP):
	print("Loading full list of parents and time indexes...")
	full_list = full_list_creator(rename_path)
	print("Loading matlab results...")
	mat_df = matlab_opener(mat_path, exp)
	print("Running pybatch...")
	py_df = load_single_pybatch_df(rename_path)
	print("Finding values from both dataframes...")
	only_pybatch, only_matlab, both, niether = both_and_niether_finder(mat_df, py_df, full_list)
	print("Ready to compare fields!")
	return only_pybatch, only_matlab, both, niether



