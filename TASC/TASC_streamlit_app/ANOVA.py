import markdown
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scikit_posthocs as sp
import statsmodels.stats.multicomp as mc
import streamlit as st


def ANOVA(summary_table_file,f):
    # Read Excel file (replace path with your own)
    data = pd.read_excel(summary_table_file, sheet_name=0)

    names = data['Experiment'].unique()
    # Clean up experiment names
    name_map = dict([(names[i],names[i][15:-4].replace('NNN0', '')) for i in range(len(names))])

    data['Experiment'] = data['Experiment'].replace(name_map)

    # Group by Experiment and Parent and calculate means
    grouped = data.groupby(['Experiment', 'Parent'], as_index=False).agg({
        'Instantaneous_Speed': 'mean',
        'Displacement2': 'mean'
    })
    # You can use either mean speed or mean displacement for ANOVA; I'll use speed here
    grouped = grouped.rename(
        columns={'Instantaneous_Speed': 'Mean_Instantaneous_Speed', 'Displacement2': 'Mean_Displacement2'})

    # Use per-parent means for ANOVA (not just grand mean per experiment)
    anova_data = grouped[['Experiment', 'Mean_Instantaneous_Speed']]
    anova_data = anova_data.rename(columns={'Mean_Instantaneous_Speed': 'Mean_Value'})

    # Perform one-way ANOVA
    model = ols('Mean_Value ~ C(Experiment)', data=anova_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Show ANOVA table
    st.markdown("### ANOVA Results")
    f.write(markdown.markdown("### ANOVA Results"))
    st.dataframe(anova_table)
    f.write(anova_table.to_html(index=True))

    # Tukey HSD post-hoc test

    comp = mc.MultiComparison(anova_data['Mean_Value'], anova_data['Experiment'])
    tukey_result = comp.tukeyhsd()
    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])

    st.markdown("### Tukey HSD Post-hoc Test")
    f.write(markdown.markdown("### ANOVA Results"))
    st.dataframe(tukey_df)
    f.write(tukey_df.to_html(index=True))

