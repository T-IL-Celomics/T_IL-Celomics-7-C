import sys

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from util import *
import markdown
import ANOVA
import WD_util

st.set_page_config(layout="wide")

def run_analysis(
        path, ST, STgraph, MorphoFeatures, BpwFeatures,
        SAMPLE, expList, CellSample, expListT, title, k_cluster,
        MorphoOut, MorphoIn, combin, singleTREAT, singleCONTROL, multipleCL,
        nrows, ncols, nColor, nShades, nColorTreat, nShadesTreat,
        nColorLay, nShadesLay, figsizeEXP, figsizeTREATS, figsizeCL,
        CON, CL, wellCON, controls, HC, AE_model, model_name,f,uploaded_file):

    # Load data
    rawdata = pd.read_pickle(path + ST)
    rawdatagraph = pd.read_pickle(path + STgraph)
    rawdata.dropna(inplace=True)
    rawdatagraph.dropna(inplace=True)

    # Split experiments
    expNamesInOrder, expNamesInOrderU, expNamesInOrderUGraph, dataAll, dataAllGraph = splitingExperimentCol(rawdata,
                                                                                                            rawdatagraph)

    # Show experiment names to user
    st.markdown("### Experiment Names in Order (Unique)")
    st.write("These are the unique experiment names found in your data:")
    st.write(expNamesInOrderU)

    # Show cell counts per experiment
    val_c = dataAllGraph['Experiment'].value_counts().sort_index()[dataAllGraph['Experiment'].unique()].rename_axis(
        'Experiment').reset_index(name='Number of Cells')
    val_c.index.name = 'Index #'
    st.markdown("### Number of Cells per Experiment")
    f.write(markdown.markdown("### Number of Cells per Experiment"))
    st.dataframe(val_c)
    f.write(val_c.to_html(index=True))

    # Sorting controls by length to avoid problems later
    controls.sort(key=len, reverse=True)

    # If sampling, show sampled experiments
    if SAMPLE:
        expM, Features, dataSpec, dataSpecGraph = analysisExp(dataAll, dataAllGraph, expList, expNamesInOrderU)
        st.markdown("### Sampled Experiments (expM)")
        st.write("These are the experiments selected after sampling:")
        st.write(expM)

    # Explain sampling to the user
    st.markdown("## Sampling Cells")
    st.write(
        "If sampling is enabled, only a subset of cells from selected wells will be used. "
        "For example, CellSample=2 means every second cell track is sampled."
    )

    FigureNumber = 1

    if SAMPLE:
        st.info("Sampling is enabled. Sampling every {} cell(s) from selected wells.".format(CellSample))
        uParent = dataSpecGraph['Parent'].unique()
        ParentList = [par if par in uParent[::CellSample] else 0 for par in dataSpecGraph['Parent']]
        dataSpecGraphP = dataSpecGraph.loc[dataSpecGraph['Parent'] == ParentList].copy()
        expM = dataSpecGraphP['Experiment'].unique()
        expM = [e for e in expM]
        st.write("Sampled experiments (expM):", expM)
    else:
        st.info("Sampling is disabled. All cells from selected wells will be used.")

    # Choose the index of the experiments to analyze
    st.markdown("## Experiments to Analyze")
    st.write(
        "The following experiments will be analyzed (excluding sampled experiments if sampling is enabled):"
    )
    st.write("expListT:", expListT )

    # Run analysis
    exp, Features, dataSpec, dataSpecGraph = analysisExp(
        dataAll.copy(), dataAllGraph.copy(), expListT, expNamesInOrderU
    )

    if SAMPLE:
        dataSpecGraphN = pd.concat([dataSpecGraph, dataSpecGraphP])
        exp.extend(expM)
    else:
        dataSpecGraphN = dataSpecGraph

    # Feature selection
    if MorphoOut:
        Features = Features.drop(MorphoFeatures).copy()
    if MorphoIn:
        MorphoFeatures.append('Experiment')
        Features = MorphoFeatures

    # Show which features are being used
    st.markdown("## Features Used for Analysis")
    st.write(Features[:-1])

    # Plot histograms (these functions must be adapted for Streamlit if they use plt.show())
    st.markdown("## Histogram of Features")
    histogramData(exp, dataSpecGraphN, Features[:-1], FigureNumber)
    FigureNumber += 1

    st.markdown("## KDE Histogram of Features")
    histogramDataKDE(exp, dataSpecGraphN, Features[:-1], FigureNumber, nColor=nColor, nShades=nShades)
    FigureNumber += 1

    columnsToDrop = ['Experiment', 'ID', 'TimeIndex', 'y_Pos', 'x_Pos', 'dt', 'Parent']  # Columns to Drop
    labelsCol = 'Experiment'  # Label to Encode

    # Prepare data for analysis
    if MorphoOut:
        dataDrop = dataSpecGraphN.drop(columns=MorphoFeatures + columnsToDrop).copy()
    else:
        dataDrop = dataSpecGraphN.drop(columns=columnsToDrop).copy()
    if MorphoIn:
        dataDrop = dataSpecGraphN[MorphoFeatures[:-1]].copy()

    dataLabel = dataSpecGraphN[['Experiment', 'TimeIndex', 'dt', 'y_Pos']].copy()

    # Run TASC analysis
    pca_df, FigureNumber, kmeans_pca, labelsT, k_cluster, AE_df, pca = TASC(
        dataDrop,
        dataLabel,
        labelsCol=['Experiment', 'Treatments', 'Layers', 'TimeLayers'],
        LE=[True, False, False, False],
        title=title,
        HC=HC,
        combTreats=combin,
        LY=9, TI=3,
        k_cluster=k_cluster,
        multipleCL=multipleCL,
        singleTREAT=singleTREAT,
        FigureNumber=FigureNumber,
        nrows=nrows, ncols=ncols,
        nColor=nColor, nShades=nShades,
        nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,
        nColorLay=nColorLay, nShadesLay=nShadesLay,
        figsizeEXP=figsizeEXP, figsizeTREATS=figsizeTREATS,
        figsizeCL=figsizeCL,
        Features=Features, AE_model=AE_model,
        model_name=model_name
    )
    
    with st.spinner("Calculating Wasserstein distance and generating heatmap..."):
        wd_matrix = WD_util.wasserstein_comparing(pca_df,experiment_list= pca_df["Experiment"].unique().tolist())

    if wd_matrix.size == 0 or wd_matrix.isna().all().all():
        st.warning("No data available to plot the heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(wd_matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title("Wasserstein Distance Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
        plt.savefig("wasserstein_heatmap.png", dpi=300)
        plt.close(fig)

    # Show figures and results
    st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
    FigureNumber += 1
    Par = 'TimeIndex'
    pca_df[Par] = dataSpecGraphN[Par].values.copy()
    histByKmeans(pca_df, Par, k_cluster=k_cluster, bar_width=0.3, figsize=(15, 5), rotate=-45)

    st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
    FigureNumber += 1
    Par = 'TimeLayers'
    histByKmeans(pca_df, 'TimeLayers', k_cluster=k_cluster, bar_width=0.3, figsize=(15, 5), labels=labelsT)

    st.latex("\\color{blue}{\\Large Distribution\\ all\\ features\\ by\\ groups}")
    Groups = range(k_cluster)
    dataSpecGraphN['Groups'] = kmeans_pca['Groups'].copy()
    histogramDataKDELabels(Groups, dataSpecGraphN, Features[:-1], FigureNumber, Par='Groups', nColor=0, nShades=0)
    FigureNumber += 1

    label_handles = []
    with st.spinner("Generating Topographic k-means in PCA plot..."):
        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1

        fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=200)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        dataSpecGraphN['PC1'] = kmeans_pca['PC1'].copy()
        dataSpecGraphN['PC2'] = kmeans_pca['PC2'].copy()
        palette = sns.color_palette("hls", len(Groups))
        uPar1tmp = kmeans_pca['Groups'].unique()
        uPar1tmp.sort()
        pal = iter(sns.color_palette([palette[p] for p in [Groups.index(value) for value in uPar1tmp]]))
        for g in uPar1tmp:
            if len(kmeans_pca['PC1'].loc[kmeans_pca['Groups'] == g]) > 3:
                color = next(pal)
                sns.kdeplot(
                    x=kmeans_pca['PC1'].loc[kmeans_pca['Groups'] == g],
                    y=kmeans_pca['PC2'].loc[kmeans_pca['Groups'] == g],
                    label='Group ' + str(g), levels=5, bw=.5, ax=ax1, color=color, cut=10
                )
                ax1.scatter([], [], color=color, label='Group ' + str(g))
            else:
                next(pal)
        labels_handles1 = {label: handle for ax1 in fig1.axes for handle, label in
                           zip(*ax1.get_legend_handles_labels())}
        label_handles = labels_handles1
        fig2.legend(
            labels_handles1.values(),
            labels_handles1.keys(),
            loc='upper right', fontsize='xx-large',
            framealpha=1, edgecolor='black'
        )
        st.pyplot(fig1)
        st.pyplot(fig2)

    with st.spinner("Generating Topographic k-means in PCA (per experiment) plot..."):
        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1

        fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsizeEXP, dpi=200, sharex=True, sharey=True)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        dataSpecGraphN['PC1'] = kmeans_pca['PC1'].copy()
        dataSpecGraphN['PC2'] = kmeans_pca['PC2'].copy()
        dataEXP = dataSpecGraphN.groupby(['Experiment'])
        palette = sns.color_palette("hls", len(Groups))
        for ex, a1 in zip(exp, ax1.reshape(-1)):
            exp_df = dataEXP.get_group(ex)
            uPar1tmp = exp_df['Groups'].unique()
            uPar1tmp.sort()
            pal = iter(sns.color_palette([palette[p] for p in [Groups.index(value) for value in uPar1tmp]]))
            for g in uPar1tmp:
                if len(exp_df['PC1'].loc[exp_df['Groups'] == g]) > 3:
                    color = next(pal)
                    sns.kdeplot(
                        x=exp_df['PC1'].loc[exp_df['Groups'] == g],
                        y=exp_df['PC2'].loc[exp_df['Groups'] == g],
                        label='Group ' + str(g), levels=5, bw=.5, ax=a1, color=color, cut=10
                    )
                else:
                    next(pal)
            a1.autoscale(enable=True, tight=True)
            a1.set_title(ex)

        for a in ax1.flat:
            try:
                a.get_legend().remove()
            except Exception:
                continue
        fig2.legend(
            label_handles.values(),
            label_handles.keys(),
            loc='upper right', fontsize='xx-large',
            framealpha=1, edgecolor='black'
        )
        fig2.subplots_adjust(right=0.5)
        st.pyplot(fig1)
        st.pyplot(fig2)

    with st.spinner("Generating Topographic k-means in PCA (rotated subplots)..."):
        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1

        fig1, ax1 = plt.subplots(nrows=ncols, ncols=nrows, figsize=figsizeEXP[::-1], dpi=200, sharex=True, sharey=True)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        dataSpecGraphN['PC1'] = kmeans_pca['PC1'].copy()
        dataSpecGraphN['PC2'] = kmeans_pca['PC2'].copy()
        dataEXP = dataSpecGraphN.groupby(['Experiment'])
        palette = sns.color_palette("hls", len(Groups))
        for ex, a1 in zip(exp, ax1.reshape(-1)):
            exp_df = dataEXP.get_group(ex)
            uPar1tmp = exp_df['Groups'].unique()
            uPar1tmp.sort()
            pal = iter(sns.color_palette([palette[p] for p in [Groups.index(value) for value in uPar1tmp]]))
            for g in uPar1tmp:
                if len(exp_df['PC1'].loc[exp_df['Groups'] == g]) > 3:
                    color = next(pal)
                    sns.kdeplot(
                        x=exp_df['PC1'].loc[exp_df['Groups'] == g],
                        y=exp_df['PC2'].loc[exp_df['Groups'] == g],
                        label='Group ' + str(g), levels=5, bw=.5, ax=a1, color=color, cut=10
                    )
                else:
                    next(pal)
            a1.autoscale(enable=True, tight=True)
            a1.set_xlim(
                [pca_df['PC1'].min() + 0.1 * exp_df['PC1'].min(), pca_df['PC1'].max() - 0.1 * pca_df['PC1'].max()])
            a1.set_ylim(
                [pca_df['PC2'].min() + 0.1 * exp_df['PC2'].min(), pca_df['PC2'].max() - 0.1 * pca_df['PC2'].max()])
            a1.set_title(ex)

        for a in ax1.flat:
            try:
                a.get_legend().remove()
            except Exception:
                continue

        fig1.subplots_adjust(right=0.8)
        fig2.legend(
            label_handles.values(),
            label_handles.keys(),
            loc='upper right', fontsize='xx-large',
        )
        fig2.subplots_adjust(right=0.5)

        st.pyplot(fig1)
        st.pyplot(fig2)

    with st.spinner("Generating k-means group distributions and statistical tests..."):
        dataSpecGraphN['Treatments'] = pca_df['Treatments'].copy()
        dataTreat = dataSpecGraphN.groupby(['Treatments'])

        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1
        Par = 'Treatments'
        lb_make = LabelEncoder()
        pca_df['TreatmentsLabels'] = lb_make.fit_transform(pca_df[Par])
        uLabelT = [u for u in pca_df['Treatments'].unique()]
        histByKmeansTreats(pca_df, 'TreatmentsLabels', k_cluster=k_cluster, bar_width=0.3, figsize=(15, 5),
                                 labels=uLabelT)

        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1
        Par = 'TreatmentsLabels'
        labels = list(pca_df.groupby('Treatments').describe().index.values)
        histByKmeansTreatsLabel(pca_df, Par='TreatmentsLabels', k_cluster=k_cluster, bar_width=0.3,
                                      figsize=(15, 5), labels=labels, rotate=0)

        if not singleTREAT:
            st.markdown("**Treatments p-Value**")
            f.write(markdown.markdown("**Treatments p-Value**"))
            labels = list(pca_df.groupby('Treatments').describe().index.values)
            Par = 'Treatments'
            for expectation in CON:
                chi_square_test_tables(pca_df, labels,f, expectation=expectation, Par=Par)

        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1
        Par = 'CellLine'
        lb_make = LabelEncoder()
        pca_df['CellLineLabels'] = lb_make.fit_transform(pca_df[Par])
        labels = list(pca_df.groupby('CellLine').describe().index.values)
        histByKmeansTreatsLabel(pca_df, Par='CellLineLabels', k_cluster=k_cluster, bar_width=0.3, figsize=(15, 5),
                                      labels=labels, rotate=0)

        if multipleCL:
            for expectation in CL:
                st.markdown(f"**Cell Line p-Value {expectation}**")
                f.write(markdown.markdown(f"**Cell Line p-Value {expectation}**"))
                labels = list(pca_df.groupby(Par).describe().index.values)
                chi_square_test_tables(pca_df, labels,f, expectation=expectation, Par=Par)

        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1
        Par = 'Experiment'
        lb_make = LabelEncoder()
        pca_df['ExperimentsLabels'] = lb_make.fit_transform(pca_df[Par])
        uLabelEXP = [u for u in pca_df['Experiment'].unique()]
        histByKmeansTreats(pca_df, 'ExperimentsLabels', k_cluster=k_cluster, bar_width=0.3, figsize=(15, 5),
                                 labels=uLabelEXP, rotate=90)

        pca_df_E = pca_df.groupby('CellLine')
        if not singleTREAT:
            labelsC = list(pca_df_E.describe().index.values)
            for cl in CL:
                labelsE = list(pca_df_E.get_group(cl).groupby(Par).describe().index.values)
                expectation = [well for well in wellCON if cl in well][0]
                st.markdown(f"**Experiments p-Value {expectation}**")
                f.write(markdown.markdown(f"**Experiments p-Value {expectation}**"))
                chi_square_test_tables(pca_df, labelsE,f, expectation=expectation, Par=Par)

        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1
        labels = list(pca_df.groupby('Experiment').describe().index.values)
        histByKmeansTreatsLabel(pca_df, Par='ExperimentsLabels', k_cluster=k_cluster, bar_width=0.3,
                                      figsize=(15, 5), labels=labels, rotate=90)

        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1
        Par = 'TimeLayersLabels'
        lb_make = LabelEncoder()
        pca_df[Par] = lb_make.fit_transform(pca_df['TimeLayers'])
        labels = list(pca_df.groupby('TimeLayers').describe().index.values)
        histByKmeansTreatsLabel(pca_df, Par=Par, k_cluster=k_cluster, bar_width=0.3, figsize=(15, 5),
                                      labels=labels, rotate=0)
        Par = 'TimeLayers'
        expectation = labels[0]
        st.markdown(f"**TimeLayers p-Value {expectation}**")
        f.write(markdown.markdown(f"**TimeLayers p-Value {expectation}**"))
        chi_square_test_tables(pca_df, labels,f, expectation=expectation, Par=Par)

    controls.sort(key=len, reverse=True)
    COMB = []
    if not singleCONTROL:
        for ex in pca_df['Experiment']:
            TF = False
            for cont in controls:
                if cont in ex and not TF:
                    COMB += [cont]
                    TF = True
        pca_df['CONTROLS'] = COMB

        Par = 'Experiment'
        pca_df_C = pca_df.groupby('CONTROLS')
        for cl in controls:
            labelsE = list(pca_df_C.get_group(cl).groupby(Par).describe().index.values)
            expectation = [well for well in wellCON if cl in well][0]
            st.markdown(f"**Experiments p-Value {expectation}**")
            f.write(markdown.markdown(f"**Experiments p-Value {expectation}**"))
            chi_square_test_tables(pca_df_C.get_group(cl), labelsE,f, expectation=expectation, Par=Par)

    with st.spinner("Generating KDE plots for each treatment..."):
        for treat in uLabelT:
            st.latex(r"\color{blue}{\Large Figure\ %s}" % (treat))
            histogramDataKDELabels(
                Groups,
                dataTreat.get_group(treat),
                Features[:-1],
                FigureNumber,
                Par='Groups',
                nColor=0,
                nShades=0
            )
            FigureNumber += 1

    with st.spinner("Generating scatter plots for each experiment..."):
        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsizeEXP, dpi=200, sharex=True, sharey=True)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        dataEXP = dataSpecGraphN.groupby(['Experiment'])
        palette = sns.color_palette("hls", len(Groups))

        for ex, a in zip(exp, ax.reshape(-1)):
            exp_df = dataEXP.get_group(ex)
            uPar1tmp = exp_df['Groups'].unique()
            uPar1tmp.sort()
            pal = sns.color_palette([palette[p] for p in [Groups.index(value) for value in uPar1tmp]])

            sns.scatterplot(x='TimeIndex', y='y_Pos', hue='Groups', palette=pal, data=exp_df, ax=a)
            a.set_title(ex, fontweight='bold', fontsize=15)
            a.autoscale(enable=True, tight=True)

        labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}
        fig.tight_layout(pad=1.02)

        for a in ax.flat:
            try:
                a.get_legend().remove()
            except Exception:
                continue

        # If you have labels_handles1 from previous code, otherwise use labels_handles here
        fig2.legend(labels_handles.values(),
                    labels_handles.keys(),
                    loc='upper right', fontsize='xx-large',
                    framealpha=1, edgecolor='black')
        fig2.subplots_adjust(right=0.5)

        st.pyplot(fig)
        st.pyplot(fig2)

    with st.spinner("Generating rotated scatter plots for each experiment..."):
        st.latex(r"\color{blue}{\Large Figure\ %i}" % (FigureNumber))
        FigureNumber += 1

        fig, ax = plt.subplots(nrows=ncols, ncols=nrows, figsize=figsizeEXP[::-1], dpi=200, sharex=True, sharey=True)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        dataEXP = dataSpecGraphN.groupby(['Experiment'])
        palette = sns.color_palette("hls", len(Groups))

        for ex, a in zip(exp, ax.reshape(-1)):
            exp_df = dataEXP.get_group(ex)
            uPar1tmp = exp_df['Groups'].unique()
            uPar1tmp.sort()
            pal = sns.color_palette([palette[p] for p in [Groups.index(value) for value in uPar1tmp]])

            sns.scatterplot(x='TimeIndex', y='y_Pos', hue='Groups', palette=pal, data=exp_df, ax=a)
            a.set_title(ex, fontweight='bold', fontsize=15)
            a.autoscale(enable=True, tight=True)

        labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}
        fig.tight_layout(pad=1.02)

        for a in ax.flat:
            try:
                a.get_legend().remove()
            except Exception:
                continue

        # If you have labels_handles1 from previous code, otherwise use labels_handles here
        fig2.legend(labels_handles.values(),
                    labels_handles.keys(),
                    loc='upper right', fontsize='xx-large',
                    framealpha=1, edgecolor='black')
        fig2.subplots_adjust(right=0.5)

        st.pyplot(fig)
        st.pyplot(fig2)

    dataSpecGraphGroups = dataSpecGraphN.copy()
    dataSpecGraphGroups['Groups'] = kmeans_pca['Groups'].copy()

    st.markdown(f"$$\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}$$")
    f.write(markdown.markdown(f"Figure {FigureNumber}"))
    st.markdown("$$\\color{blue}{\\Large Descriptive\\ Table}$$")
    f.write(markdown.markdown("Descriptive Table"))
    FigureNumber += 1

    DescriptiveTable(dataSpecGraphGroups, path + title,f)

    st.markdown(f"$$\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}$$")
    f.write(markdown.markdown(f"Figure {FigureNumber}"))
    st.markdown("$$\\color{blue}{\\Large ANOVA\\ -\\ OneWay}$$")
    f.write(markdown.markdown("ANOVA - OneWay"))
    FigureNumber += 1

    with st.spinner('Running ANOVA...'):
        for col in dataSpecGraphGroups.columns:
            try:
                dataSpecGraphGroups[col] = np.float64(dataSpecGraphGroups[col])
            except:
                st.write(f"Could not convert column: {col}")

        ANOVA_TABLE(dataSpecGraphGroups, Features,f, path + title, dep='Groups')

    dataSpecGraphGroups['CellLine'] = pca_df['CellLine'].copy()
    dataSpecGraphGroupsCL = dataSpecGraphGroups.groupby('CellLine')

    if multipleCL:
        st.markdown(f"$$\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}$$")
        st.markdown("$$\\color{blue}{\\Large ANOVA\\ -\\ OneWay}$$")
        FigureNumber += 1
        for cl in CL:
            st.markdown(f"$$\\color{{blue}}{{\\Large ANOVA\\ -\\ OneWay\\ {cl}}}$$")
            with st.spinner(f'Running ANOVA for {cl}...'):
                ANOVA_TABLE(dataSpecGraphGroupsCL.get_group(cl), Features,f, path + title + ' ' + cl,
                                     dep='Groups')

    dataSpecGraphGroups = dataSpecGraphN.copy()
    dataSpecGraphGroups['Groups'] = kmeans_pca['Groups'].copy()

    if multipleCL:
        st.markdown(f"$$\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}$$")
        f.write(markdown.markdown(f"Figure {FigureNumber}"))
        st.markdown("$$\\color{blue}{\\Large Descriptive\\ Table}$$")

        FigureNumber += 1
        for cl in CL:
            st.markdown(f"$$\\color{{blue}}{{\\Large Descriptive\\ Table\\ {cl}}}$$")
            f.write(markdown.markdown(f"Descriptive Table for {cl}"))
            DescriptiveTable(dataSpecGraphGroupsCL.get_group(cl), path + title + ' ' + cl,f)

    # When the user uploads a summary table, it is passed to the ANOVA function,
    # which reads the data, performs statistical analysis, and displays the results in the app.
    st.title("ANOVA & Tukey HSD Results")

    if uploaded_file:
        with st.spinner("creating ANOVA & Tukey HSD Results ..."):
            ANOVA.ANOVA(uploaded_file,f)

    with open(path + title + 'split.pickle', 'wb') as f:
        pickle.dump([pca_df, FigureNumber, kmeans_pca, labelsT, k_cluster, AE_df,
                     dataSpecGraphN, dataEXP, Groups, Features, exp, pca], f)


st.title("TASC 2.4 Streamlit GUI")

with st.expander("Show Example Input"):
    st.markdown("""
    ```python
    # Example input for a typical analysis run:
    path = "D:\\jeries\\output\\TASC_pickles\\"
    ST = "rawdata.pickle"
    STgraph = "rawdatagraph.pickle"
    MorphoFeatures = ['Area', 'Ellip_Ax_B_X', 'Ellip_Ax_B_Y']
    BpwFeatures = ['Velocity_Full_Width_Half_Maximum', 'Velocity_Maximum_Height']
    SAMPLE = False
    expList = [16,17,18,19,20,21,22,23]
    CellSample = 2
    expListT = [0,1,2,3,4,5,6,7,8,9,10,11]
    title = "jeries_test"
    k_cluster = 3
    MorphoOut = False
    MorphoIn = False
    combin = [["NNIR"],["METR"],["GABY"]]
    singleTREAT = False
    singleCONTROL = True
    multipleCL = False
    nrows = 1
    ncols = 12
    nColor = 12
    nShades = 2
    nColorTreat = 0
    nShadesTreat = 0
    nColorLay = 3
    nShadesLay = 3
    figsizeEXP = (25,5)
    figsizeTREATS = (15,5)
    figsizeCL = (15,15)
    CON = ['NNIR']
    CL = ['293T']
    wellCON = ['AM001100425CHR2B02293TNNIRNOCOWH00']
    controls = []
    ```
    """)

# Path to pickle files
path = st.text_input("Path to pickle files", value=""
                     , placeholder="e.g. D:\\path to folder\\TASC_pickles\\")
if not path.endswith("\\"):
    path = path +"\\"

uploaded_file = st.file_uploader("Upload your summary table (.xlsx)", type=["xlsx"])

# File names
ST = st.text_input("Raw data pickle filename", value="rawdata.pickle")
STgraph = st.text_input("Raw data graph pickle filename", value="rawdatagraph.pickle")

# Features
MorphoFeatures = st.multiselect(
    "Morphological Features",
    ['Area', 'Ellip_Ax_B_X', 'Ellip_Ax_B_Y', 'Ellip_Ax_C_X', 'Ellip_Ax_C_Y',
     'EllipsoidAxisLengthB', 'EllipsoidAxisLengthC',
     'Ellipticity_oblate', 'Ellipticity_prolate',
     'Sphericity', 'Eccentricity'],
    default=['Area', 'Ellip_Ax_B_X', 'Ellip_Ax_B_Y', 'Ellip_Ax_C_X', 'Ellip_Ax_C_Y',
             'EllipsoidAxisLengthB', 'EllipsoidAxisLengthC',
             'Ellipticity_oblate', 'Ellipticity_prolate',
             'Sphericity', 'Eccentricity']
)

BpwFeatures = st.multiselect(
    "BPW Features",
    ['Velocity_Full_Width_Half_Maximum',
     'Velocity_Time_of_Maximum_Height',
     'Velocity_Maximum_Height',
     'Velocity_Ending_Value', 'Velocity_Ending_Time',
     'Velocity_Starting_Value', 'Velocity_Starting_Time'],
    default=['Velocity_Full_Width_Half_Maximum',
             'Velocity_Time_of_Maximum_Height',
             'Velocity_Maximum_Height',
             'Velocity_Ending_Value', 'Velocity_Ending_Time',
             'Velocity_Starting_Value', 'Velocity_Starting_Time']
)

# Sampling
st.markdown("**Sampling**  \n"
            "If you want to sample cells to balance well sizes, check this box.  \n"
            "- `expList`: Indices of wells to sample (comma-separated).  \n"
            "- `CellSample`: Sampling divisor (e.g., 2 for half, 3 for a third).")
SAMPLE = st.checkbox("Sample cells?", value=False)
expList = st.text_input("expList (indices to sample, comma-separated)", value="16,17,18,19,20,21,22,23")
expList = [int(x.strip()) for x in expList.split(",") if x.strip()]
CellSample = st.number_input("CellSample (sampling divisor)", min_value=1, value=2)

# Experiments to analyze (without sampling)
st.markdown("**Experiments to analyze (without sampling)**  \n"
            "Indices for the wells you wish to analyze *without* sampling.  \n"
            "You can use a comma-separated list (e.g., `0,1,2,3,4,5,6,7,8,9,10,11`) or a Python "
            "range (e.g., `range(0,12)`).")
expListT_input = st.text_input("expListT (indices to analyze, comma-separated or range)",value = "",
                               placeholder="0,1,2,3,4,5,6,7,8,9,10,11")
expListT = []
# Parse input: support both comma-separated and range(a,b)
if expListT_input.strip().startswith("range"):
    try:
        # Evaluate safely: only allow range
        expListT = list(eval(expListT_input, {"range": range}))
    except Exception:
        st.error("Invalid range format. Use e.g. range(0,12)")
        expListT = []
else:
    expListT = [int(x.strip()) for x in expListT_input.split(",") if x.strip()]

# Title and clusters
st.markdown("**Experiment title and number of clusters**  \n"
            "Choose a unique title for this run. All output files will use this as a prefix.  \n"
            "Set the number of clusters for k-means analysis.")
title = st.text_input("Experiment title",value = "", placeholder="e.g. jeries_test")
k_cluster = st.number_input("Number of clusters", min_value=1, value=3)

# Feature selection
st.markdown("**Feature selection**  \n"
            "Check to analyze only kinetics (BPW) or only morphology.  \n"
            "If you wish to run on the entire feature list, leave both unchecked.")
MorphoOut = st.checkbox("Analyze only kinetics (MorphoOut)", value=False)
MorphoIn = st.checkbox("Analyze only morphology (MorphoIn)", value=False)

# Treatments combination
st.markdown("""
**Treatments Combination**

Specify how your treatments can be combined in your experiments.  
- Each treatment name should be 4 characters long.
- Treatments that **are not combined with each other** should be placed in the **same list**.
- Treatments that **can be combined** with others should be in separate lists.

**Example:**  
If your treatments are `HGF2`, `HGF7`, `DOX1`, `PHA4`, `PHA3`:
- If `HGF2`, `HGF7`, and `DOX1` are not combined with each other, put them in the same list: `['HGF2', 'HGF7', 'DOX1']`
- If `PHA4` and `PHA3` are not combined with the others, put them in another list: `['PHA4', 'PHA3']`
- The format is: `[['HGF2', 'HGF7', 'DOX1'], ['PHA4', 'PHA3']]`

You will **not** find a well named `'HGF2HGF7'` or `'HGF2DOX1'`, but you **may** find `'HGF7PHA4'`.

**Note:**  
If you have a control well (e.g., `'CON'`), do **not** include it in any list.
""")
combin = st.text_input("Treatments combination (e.g. [[NNIR],[METR],[GABY]])", value='[["NNIR"],["METR"],["GABY"]]')
combin = eval(combin)

# Single/multiple treatment/control/cell lines
st.markdown("**Single/multiple treatment/control/cell lines**  \n"
            "- `singleTREAT`: True if only one treatment.  \n"
            "- `singleCONTROL`: True if only one control.  \n"
            "- `multipleCL`: True if multiple cell lines.")
singleTREAT = st.checkbox("Single treatment?", value=False)
singleCONTROL = st.checkbox("Single control?", value=True)
multipleCL = st.checkbox("Multiple cell lines?", value=False)

# Graph properties
st.markdown("""
**Graph Properties**

These settings control the appearance and layout of your output figures. You may need to adjust them depending on your experiment and preferences.

- **nrows, ncols**:  
  Set the number of rows and columns in your experiment figures. For example, if you have 3 cell lines and 8 treatments, use `nrows=3` and `ncols=8`.  
  Default: `nrows=0`, `ncols=1`

- **nColor, nShades**:  
  Define the number of colors and shades used in the experiment figures.    
  Default: `nColor=0`, `nShades=0`

- **nColorTreat, nShadesTreat**:  
  Number of colors and shades for treatment-specific figures. Usually, you can leave these at their default values.  
  Default: `nColorTreat=0`, `nShadesTreat=0`

- **nColorLay, nShadesLay**:  
  Number of colors and shades for y-position (layer) figures.  
  Example: `nColorLay=3`, `nShadesLay=3`

- **figsizeEXP, figsizeTREATS, figsizeCL**:  
  Set the figure size for experiment, treatment, and cell line figures as `(width, height)` tuples.  
  Example:  
  - `figsizeEXP = (40, 15)`  
  - `figsizeTREATS = (30, 15)`  
  - `figsizeCL = (15, 5)`

*Tip: You may need to revisit and adjust these settings to get the best visualization for your data.*
""")
nrows = st.number_input(
    "nrows (rows in experiment figures)",
    min_value=1,
    value=1,
    help="Number of rows in the experiment figures. For example, set nrows=3 if you have 3 cell lines."
)
ncols = st.number_input(
    "ncols (columns in experiment figures)",
    min_value=1,
    value=12,
    help="Number of columns in the experiment figures. For example, set ncols=8 if you have 8 treatments."
)
nColor = st.number_input(
    "nColor (number of colors)",
    min_value=1,
    value=12,
    help="Number of distinct colors to use in experiment figures."
)
nShades = st.number_input(
    "nShades (number of shades)",
    min_value=1,
    value=2,
    help="Number of shades per color in experiment figures."
)
nColorTreat = st.number_input(
    "nColorTreat",
    min_value=0,
    value=0,
    help="Number of colors for treatment-specific figures. Usually leave as 0 unless needed."
)
nShadesTreat = st.number_input(
    "nShadesTreat",
    min_value=0,
    value=0,
    help="Number of shades for treatment-specific figures. Usually leave as 0 unless needed."
)
nColorLay = st.number_input(
    "nColorLay",
    min_value=1,
    value=3,
    help="Number of colors for y-position (layer) figures. Usually leave as default."
)
nShadesLay = st.number_input(
    "nShadesLay",
    min_value=1,
    value=3,
    help="Number of shades for y-position (layer) figures. Usually leave as default."
)
figsizeEXP = st.text_input(
    "figsizeEXP (tuple)",
    value="(25,5)",
    help="Figure size for experiment figures as a tuple (width, height), e.g., (25, 5)."
)
figsizeEXP = eval(figsizeEXP)
figsizeTREATS = st.text_input(
    "figsizeTREATS (tuple)",
    value="(15,5)",
    help="Figure size for treatment figures as a tuple (width, height), e.g., (15, 5)."
)
figsizeTREATS = eval(figsizeTREATS)
figsizeCL = st.text_input(
    "figsizeCL (tuple)",
    value="(15,15)",
    help="Figure size for cell line figures as a tuple (width, height), e.g., (15, 15) or (15, 5)."
)
figsizeCL = eval(figsizeCL)

# Chi-squared test variables
st.markdown("""
**Chi-Squared Test Variables**

- **CON**:  
  List of control names (comma-separated).  
  Example: `CON = ['CON']` or `CON = ['CON1', 'CON2']`

- **CL**:  
  List of cell lines (comma-separated).  
  Example: `CL = ['BT54', 'MDA2', 'MCF7']` or CL = ['293T']

- **wellCON**:  
  List of well names for controls (one per line, without the 'NNN0' part).  
  Example:  
  ```
  HA033080917CHR1C02BT54CON0WH00
  HA033080917CHR1D02MDA2CON0WH00
  HA033080917CHR1F02MCF7CON0WH00
  ```

- **controls**:  
  List of specific experiment names to use as controls (one per line).  
  Example:  
  ```
  BT54HGF7
  ```
  Leave blank if not used.
""")
CON = st.text_input("CON (control names, comma-separated)",value ="", placeholder="e.g. NNIR").split(",")
CL = st.text_input("CL (cell lines, comma-separated)", value="",placeholder="e.g. 293T").split(",")
wellCON = st.text_area("wellCON (well names, one per line)",
                       value="AM001100425CHR2B02293TNNIRNOCOWH00").splitlines()
controls = st.text_area("controls (leave blank for none)", value="").splitlines()
# Sorting controls by length to avoid problems later
controls.sort(key=len, reverse=True)

# Hierarchical clustering and AE model
HC = False
AE_model = False
model_name = "d100220h122850"

# Initialize session state for run control
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Run Analysis"):
        st.session_state.analysis_running = True

with col2:
    if st.button("Stop Analysis"):
        st.session_state.analysis_running = False
        st.warning("Analysis stopped. You can rerun the analysis whenever you want.")

if st.session_state.analysis_running:
    st.info("Analysis is running...")
    with open("report_tables.html", "w") as f:
        f.write("<html><body>")
        run_analysis(
            path, ST, STgraph, MorphoFeatures, BpwFeatures,
            SAMPLE, expList, CellSample, expListT, title, k_cluster,
            MorphoOut, MorphoIn, combin, singleTREAT, singleCONTROL, multipleCL,
            nrows, ncols, nColor, nShades, nColorTreat, nShadesTreat,
            nColorLay, nShadesLay, figsizeEXP, figsizeTREATS, figsizeCL,
            CON, CL, wellCON, controls, HC, AE_model, model_name,f,uploaded_file)  # Call the analysis function from the analysis module
        f.write("</body></html>")
    # Place your analysis code here
    # Example: st.write("Parameters:", path, ST, STgraph, MorphoFeatures, ...)
    # After analysis is done, you can set:
    # st.session_state.analysis_running = False
    st.success("Analysis complete! You can rerun the analysis by clicking 'Run Analysis' again.")


