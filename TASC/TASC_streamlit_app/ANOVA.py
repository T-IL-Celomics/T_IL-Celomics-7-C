import markdown
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scikit_posthocs as sp
import statsmodels.stats.multicomp as mc
import streamlit as st
import os



def ANOVA(summary_files,f):
    # Read Excel file (replace path with your own)

    dfs = [pd.read_excel(f) for f in summary_files]
    data = pd.concat(dfs, ignore_index=True)

    names = data['Experiment'].unique()

    # Clean up experiment names
    name_map = dict([(names[i],names[i][15:-4].replace('NNN0', '')) for i in range(len(names))])

    data['Experiment'] = data['Experiment'].replace(name_map)
    # Group by Experiment and Parent and calculate means
    grouped = data.groupby(['Experiment','Parent'], as_index=False).agg({
        'Instantaneous_Speed': 'mean',
        'Displacement2': 'mean'
    })

    # For Instantaneous Speed
    anova_data_speed = grouped[['Experiment', 'Instantaneous_Speed']].rename(
        columns={'Instantaneous_Speed': 'Mean_Value'})

    anova_data_speed = anova_data_speed.dropna(subset=['Experiment', 'Mean_Value'])
    # ANOVA for speed
    model_speed = ols('Mean_Value ~ C(Experiment)', data=anova_data_speed).fit()
    anova_table_speed = sm.stats.anova_lm(model_speed, typ=2)

    # Tukey HSD for speed
    comp_speed = mc.MultiComparison(anova_data_speed['Mean_Value'], anova_data_speed['Experiment'])
    tukey_result_speed = comp_speed.tukeyhsd()
    tukey_df_speed = pd.DataFrame(data=tukey_result_speed._results_table.data[1:],
                                  columns=tukey_result_speed._results_table.data[0])

    # For Displacement
    anova_data_disp = grouped[['Experiment', 'Displacement2']].rename(
        columns={'Displacement2': 'Mean_Value'})

    anova_data_disp = anova_data_disp.dropna(subset=['Experiment', 'Mean_Value'])

    # ANOVA for displacement
    model_disp = ols('Mean_Value ~ C(Experiment)', data=anova_data_disp).fit()
    anova_table_disp = sm.stats.anova_lm(model_disp, typ=2)

    # Tukey HSD for displacement
    comp_disp = mc.MultiComparison(anova_data_disp['Mean_Value'], anova_data_disp['Experiment'])
    tukey_result_disp = comp_disp.tukeyhsd()
    tukey_df_disp = pd.DataFrame(data=tukey_result_disp._results_table.data[1:],
                                 columns=tukey_result_disp._results_table.data[0])

    tukey_df_speed['Measurement'] = 'Instantaneous_Speed'
    tukey_df_disp['Measurement'] = 'Displacement2'

    # Concatenate the Tukey tables
    tukey_combined = pd.concat([tukey_df_speed, tukey_df_disp], ignore_index=True)

    # Show and write ANOVA results for Mean Instantaneous Speed
    st.markdown("### ANOVA Results (Mean Instantaneous Speed)")
    st.dataframe(anova_table_speed)
    f.write("<h3>ANOVA Results (Mean Instantaneous Speed)</h3>")
    f.write(anova_table_speed.to_html(index=True))

    # Show and write ANOVA results for Mean Displacement
    st.markdown("### ANOVA Results (Mean Displacement2)")
    st.dataframe(anova_table_disp)
    f.write("<h3>ANOVA Results (Mean Displacement2)</h3>")
    f.write(anova_table_disp.to_html(index=True))

    # Show and write the combined table
    st.markdown("### Combined Tukey HSD Table (Both Measurements)")
    st.dataframe(tukey_combined)
    f.write("<h3>Combined Tukey HSD Table (Both Measurements)</h3>")
    f.write(tukey_combined.to_html(index=True))

    # Get the directory where the current Python file is located
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the output file path
    output_file = os.path.join(output_dir, "tukey_combined.xlsx")

    # Write the Tukey combined table to Excel
    tukey_combined.to_excel(output_file, index=False)

    # Optional: Let user know where the file was saved
    st.success(f"Tukey combined table saved to: {output_file}")