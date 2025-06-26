from matplotlib import pyplot as plt
import os

def plot_average_msd(msd_dicts, dt=45, ylimit=None, title=None):
	fig, ax = plt.subplots()
	ax.set_xlabel("Tau (%d minutes each)" % dt)
	ax.set_ylabel("Average MSD")
	if ylimit:
		ax.set_ylim(top=ylimit)
	if title:
		ax.set_title(title + " (average)")
	timepoints = list(range(min([min(list(d.keys())) for d in msd_dicts if d not in [None, {}]]), 
							max([max(list(d.keys())) for d in msd_dicts if d not in [None, {}]]) + 1))
	avges = []
	for i in timepoints:
		temp_list = [d[i] for d in msd_dicts if d not in [None, {}] and i in d.keys()]
		try:
			avges.append(sum(temp_list) / len(temp_list))
		except ZeroDivisionError:
			print("Added None at timpoint", i)
			avges.append(None)
	ax.plot(timepoints, avges)


def plot_msd_group(msd_dicts, dt=45, plot_average=True, ylimit=None, title=None):
	fig, ax = plt.subplots()
	ax.set_xlabel("Tau (%d minutes each)" % dt)
	ax.set_ylabel("MSD")
	if ylimit:
		ax.set_ylim(top=ylimit)
	if title:
		ax.set_title(title)
	for msd in msd_dicts:
		if msd != None:
			ax.plot(list(msd.keys()), list(msd.values()))
	if plot_average:
		if ylimit == None:
			ylimit = ax.get_ylim()[1]
		plot_average_msd(msd_dicts, dt=dt, ylimit=ylimit, title=title)


def plot_msds_by_scratch(parent_list, position_to_check):
	y_pos_list = [p.info_per_id[position_to_check].y_Pos for p in parent_list]
	top = max(y_pos_list)
	bottom = min(y_pos_list)
	midline = (top + bottom) / 2
	sixth_height = (top - midline) / 3
	max_value = max([max(list(p.MSDs.values())) for p in parent_list if "MSDs" in p.__dict__.keys()])
	msds_on_scratch = [p.MSDs for p in parent_list if abs(p.info_per_id[0].y_Pos - midline) < sixth_height and "MSDs" in p.__dict__.keys()]
	print("Found %d cells around scratch" % len(msds_on_scratch))
	msds_close_to_scratch = [p.MSDs for p in parent_list if abs(p.info_per_id[0].y_Pos - midline) > sixth_height and abs(p.info_per_id[0].y_Pos - midline) < 2 * sixth_height and "MSDs" in p.__dict__.keys()]
	print("Found %d cells close to scratch" % len(msds_close_to_scratch))
	msds_far_from_scratch = [p.MSDs for p in parent_list if abs(p.info_per_id[0].y_Pos - midline) > 2 * sixth_height and "MSDs" in p.__dict__.keys()]
	print("Found %d cells far from scratch" % len(msds_far_from_scratch))
	plot_msd_group(msds_on_scratch, ylimit=max_value, title="Cells around scratch")
	plot_msd_group(msds_close_to_scratch, ylimit=max_value, title="Cells close to scratch")
	plot_msd_group(msds_far_from_scratch, ylimit=max_value, title="Cells far from scratch")


def plot_average_bic_bar(bic_dicts, ylim=None):
	fig, ax = plt.subplots()
	ax.set_xlabel("Model of Motion")
	ax.set_ylabel("Average BIC Score")
	if ylim:
		ax.set_ylim(top=ylim)
	width = 0.35
	for group in bic_dicts.keys():
		motions, bics_list = bic_dicts[group]
		bic_avgs = []
		stds = []
		for bics in bics_list:
			bic_avg = sum(bics) / len(bics)
			std = (sum([(bic - bic_avg) ** 2 for bic in bics]) / len(bics)) ** 0.5
			bic_avgs.append(bic_avg)
			stds.append(std)
		ax.bar(motions, bic_avgs, width=width, align="edge", yerr=stds, label=group)
		width *= -1
	ax.legend()
	fig.tight_layout()


def plot_dropna_barchart(output_df, desired_fields):
	counts = output_df.count()
	nans = {k: len(output_df) - counts[k] for k in list(counts.keys()) if counts[k] != len(output_df)}
	labels = list(nans.keys())
	labels.sort(key=lambda x: nans[x])
	values = list(nans.values())
	values.sort()
	fig, ax = plt.subplots()
	ax.set_ylabel("Amount of NaN values")
	bars = ax.bar(range(len(labels)), values)
	ax.set_xticks(range(len(labels)))
	ax.set_xticklabels(labels)
	plt.xticks(rotation=90)
	for bar in bars:
		height = bar.get_height()
		ax.annotate('{}'.format(height),
					xy=(bar.get_x() + bar.get_width() / 2, height),
					xytext=(0, 3),  # 3 points vertical offset
					textcoords="offset points",
					ha='center', va='bottom')
	fig.tight_layout()


if __name__ == "__main__":
	print(os.path.realpath(__file__))


