import markdown
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scikit_posthocs as sp
import statsmodels.stats.multicomp as mc
import streamlit as st
import os

parameters = ["Area", "Displacement_From_Last_Id", "Instantaneous_Speed", "Velocity_X", "Velocity_Y", "Velocity_Z", "Coll", "Coll_CUBE", "Acceleration", "Acceleration_X",
                     "Acceleration_Y", "Acceleration_Z", "Displacement2", "Directional_Change", "Directional_Change_X", "Directional_Change_Y", "Directional_Change_Z", "Volume",
                     "Ellipticity_oblate", "Ellipticity_prolate", "Eccentricity", "Eccentricity_A", "Eccentricity_B", "Eccentricity_C", "Sphericity", "EllipsoidAxisLengthB",
                     "EllipsoidAxisLengthC", "Ellip_Ax_B_X", "Ellip_Ax_B_Y", "Ellip_Ax_B_Z", "Ellip_Ax_C_X", "Ellip_Ax_C_Y", "Ellip_Ax_C_Z", "Instantaneous_Angle",
                     "Instantaneous_Angle_X", "Instantaneous_Angle_Y", "Instantaneous_Angle_Z", "Min_Distance"]


def ANOVA(summary_files,f):
    # Read Excel file (replace path with your own)

    dfs = [pd.read_excel(f) for f in summary_files]
    data = pd.concat(dfs, ignore_index=True)

    names = data['Experiment'].unique()

    # Clean up experiment names
    name_map = dict([(names[i],names[i][15:-4].replace('NNN0', '')) for i in range(len(names))])

    data['Experiment'] = data['Experiment'].replace(name_map)
    # Group by Experiment and Parent, calculate means for all available parameters
    agg_dict = {param: 'mean' for param in parameters if param in data.columns}
    grouped = data.groupby(['Experiment', 'Parent'], as_index=False).agg(agg_dict)

    tukey_dict = {}
    tukey_combined = None

    for param in agg_dict:
        anova_data = grouped[['Experiment', param]].rename(columns={param: 'Mean_Value'})
        anova_data = anova_data.dropna(subset=['Experiment', 'Mean_Value'])

        # Only proceed if there are at least 2 groups and at least some data
        if anova_data['Experiment'].nunique() < 2 or anova_data['Mean_Value'].isna().all():
            continue

        # ANOVA
        model = ols('Mean_Value ~ C(Experiment)', data=anova_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Tukey HSD
        comp = mc.MultiComparison(anova_data['Mean_Value'], anova_data['Experiment'])
        tukey_result = comp.tukeyhsd()
        tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
        tukey_df['Measurement'] = param
        tukey_dict[param] = tukey_df
        if tukey_combined is None:
            tukey_combined = tukey_df
        else:
            tukey_combined = pd.concat([tukey_combined, tukey_df], ignore_index=True)

        # Display and write results for each parameter
        st.markdown(f"### ANOVA Results ({param})")
        st.dataframe(anova_table)
        f.write(f"<h3>ANOVA Results ({param})</h3>")
        f.write(anova_table.to_html(index=True))

    # Show and write the combined table
    st.markdown("### Combined Tukey HSD Table (Both Measurements)")
    st.dataframe(tukey_combined)
    f.write("<h3>Combined Tukey HSD Table (Both Measurements)</h3>")
    f.write(tukey_combined.to_html(index=True))

    # Save all Tukey results to a multi-sheet Excel
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "tukey_results.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        for param, tukey_df in tukey_dict.items():
            tukey_df.to_excel(writer, sheet_name=param[:31], index=False)  # Excel limit: 31 chars for sheet name

    st.success(f"Tukey results saved to multi-sheet Excel: {output_file}")

    