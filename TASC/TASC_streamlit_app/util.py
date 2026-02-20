import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, PowerTransformer,Normalizer, MinMaxScaler
from sklearn.mixture import GaussianMixture
import pickle
from IPython.display import display, Latex
from itertools import product, compress
from scipy.stats import *
import math
import researchpy as rp

from statsmodels.formula.api import ols
import statsmodels.api as sm
# 
import tensorflow as tf
from tensorflow.keras import backend as KTF
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from datetime import datetime
import streamlit as st


labels_handles_dic = {}

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


sns.set(style="ticks")

def get_rows_cols(feature_num):
    ncols = np.ceil(feature_num ** 0.5)
    nrows = np.ceil(feature_num / ncols)
    return int(nrows), int(ncols)

def extractSpecificExp(data,dataGraph,exp):
    dataSpec = pd.DataFrame(columns=data.columns)
    dataSpecGraph = pd.DataFrame(columns=dataGraph.columns)

    for e in exp:
        dataOne = data.loc[data['Experiment']==e,:].copy()
        dataOneGraph = dataGraph.loc[dataGraph['Experiment']==e,:].copy()
        dataSpec = pd.concat([dataSpec, dataOne], ignore_index=True)
        dataSpecGraph = pd.concat([dataSpecGraph, dataOneGraph], ignore_index=True)

    return dataSpec, dataSpecGraph
	

def splitingExperimentCol(raw, rawgraph):
    expNamesInOrder = raw['Experiment'].copy()
    expNamesInOrderU = raw['Experiment'].unique()
    expNamesInOrderUGraph = rawgraph['Experiment'].unique()
    dataAll = raw.copy()
    dataAllGraph = rawgraph.copy()
    return expNamesInOrder, expNamesInOrderU, expNamesInOrderUGraph, dataAll, dataAllGraph 

    
def splitingExperimentColH(rawgraph):
    expNamesInOrder = rawgraph['Experiment'].copy()
    expNamesInOrderU = rawgraph['Experiment'].unique()
    expNamesInOrderUGraph = rawgraph['Experiment'].unique()
    dataAll = rawgraph.copy()
    dataAllGraph = rawgraph.copy()
    return expNamesInOrder, expNamesInOrderU, expNamesInOrderUGraph, dataAll, dataAllGraph  


def extractOneExp(dataAll,dataAllGraph,exp):
    dataOne = dataAll.loc[dataAll['Experiment']==exp,:].copy()
    dataOneGraph = dataAllGraph.loc[dataAllGraph['Experiment']==exp,:].copy()
    dataOneGraphSeries = createNewDFsameIndex(['y_Pos_Start'],dataOneGraph['y_Pos'].index)
    for y in dataOneGraph['y_Pos'].index:
        Parent = dataOneGraph.loc[y,'Parent']
        listForParent = dataOneGraph.loc[dataOneGraph['Parent']==Parent,:].copy()
        yPosMinTime = listForParent.loc[listForParent['TimeIndex'] == listForParent['TimeIndex'].min(),'y_Pos']
        dataOneGraphSeries['y_Pos_Start'][y] =  yPosMinTime.get(key = yPosMinTime.index[0])
    dataOne['y_Pos_Start'] = dataOneGraphSeries['y_Pos_Start'].copy()
    return dataOne, dataOneGraph, dataOneGraphSeries
    

def createNewDFsameIndex(columns,index):
    newDF = pd.DataFrame(columns=columns,index=index)
    return newDF


def zScoreEach(data):
    for col in data.columns:
        data[col] = zscore(data[col]).astype(float)
    return data


def createHierarchicalCluster(data, title, xlabel, ylabel, cmap="RdBu_r", vmin=-2, vmax=2):
    with st.spinner("Generating hierarchical cluster..."):
        linkaged_pca = linkage(data, 'ward')
        s = sns.clustermap(
            data=data,
            row_linkage=linkaged_pca,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            figsize=(30, 15),
            cbar_kws=dict(use_gridspec=False)
        )
        plt.suptitle(title, fontweight='bold', fontsize=30)
        s.ax_heatmap.set_xlabel(xlabel, fontsize=25, fontweight='bold')
        s.ax_heatmap.set_ylabel(ylabel, fontsize=25, fontweight='bold')
        s.cax.set_yticklabels(s.cax.get_yticklabels(), fontsize=25)
        pos = s.ax_heatmap.get_position()
        cbar = s.cax
        cbar.set_position([0.02, pos.bounds[1], 0.02, pos.bounds[3]])
        s.ax_heatmap.set_xticklabels(
            s.ax_heatmap.get_xticklabels(),
            rotation=40,
            horizontalalignment='right',
            fontsize=20
        )
        st.pyplot(s.figure)
        plt.close(s.figure)
    return None


def pcaVarianceExplained(pca, NSC):
    with st.spinner("Plotting PCA variance explained..."):
        f, ax = plt.subplots(figsize=(10, 10), dpi=200)
        features = range(pca.n_components_)
        ax.bar(features, pca.explained_variance_ratio_ * 100)
        ax.set_xlabel('PCA features')
        ax.set_ylabel('Variance explained %')
        ax.set_xticks(features)
        st.write(f'There are {NSC} significant components')
        st.pyplot(f)
        plt.close(f)
    return None
	

def pcaPlot( pca, pca_df, hue, title, nColor=0, nShades=0, nColorTreat=0, nShadesTreat=0,
    nColorLay=0, nShadesLay=0, xlim_kmeans=[0,0], ylim_kmeans=[0,0]):
    with st.spinner("Generating PCA plot..."):
        sns.axes_style({'axes.spines.left': True, 'axes.spines.bottom': True,
                        'axes.spines.right': True, 'axes.spines.top': True})
        f, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200, facecolor='w', edgecolor='k')
        num_of_dep = len(hue.unique())
        sns.despine(f, left=True, bottom=True)
        if hue.name == 'Experiment' and nColor != 0:
            palette = ChoosePalette(nColor, nShades)
        elif hue.name == 'Treatments' and nColorTreat != 0:
            palette = ChoosePalette(nColorTreat, nShadesTreat)
        elif hue.name == 'Layers' and nColorLay != 0:
            palette = ChoosePalette(nColorLay, nShadesLay)
        else:
            palette = sns.color_palette("hls", num_of_dep)  # Choose color

        pca_expln_var_r = pca.explained_variance_ratio_ * 100

        s = sns.scatterplot(
            x="PC1", y="PC2", hue=hue.name, data=pca_df, ax=ax, palette=palette
        )
        handles, labels = ax.get_legend_handles_labels()
        legend_texts = ax.get_legend().texts
        ax.get_legend().remove()


        leg_title = ""
        if (len(handles) + 1) == len(labels):
            leg_title = labels[0]
            labels = labels[1:]

        plt.suptitle(title, fontweight='bold', fontsize=15)
        if xlim_kmeans != [0, 0]:
            plt.xlim(xlim_kmeans)
            plt.ylim(ylim_kmeans)
        plt.xlabel('PC1 ' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%')
        plt.ylabel('PC2 ' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%')


        if len(legend_texts) > 25:
            f.legend(handles,labels,title = leg_title,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=3, framealpha=1, edgecolor='black')
        elif len(legend_texts) > 17:
            f.legend(handles,labels,title = leg_title,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2, framealpha=1, edgecolor='black')
        else:
            f.legend(handles,labels,title = leg_title,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1, framealpha=1, edgecolor='black')

        for spine in ax.spines.values():
            spine.set_visible(True)
        st.pyplot(f)
        plt.close(f)
    return None
    
	
def pcaCalcOneExpMod(data, exp, title, whiten, n_components, nColor=0, nShades=0, FigureNumber=1):
    with st.spinner("Calculating PCA and plotting..."):
        pca = decomposition.PCA(n_components=n_components, random_state=42, whiten=whiten)
        pca.fit(data)
        pca_transformed = pca.transform(data)
        number_of_significant_components = sum(pca.explained_variance_ratio_ > 0.1)
        pca_df = pd.DataFrame(pca_transformed[:, 0:2], index=exp.index)
        pca_df.rename(columns={0: 'PC1', 1: 'PC2'}, inplace=True)
        pca_df['Experiment'] = [expNames.replace('NNN0', '') for expNames in exp]

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        pcaPlot(pca, pca_df, pca_df['Experiment'], title, nColor, nShades)
        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        pcaVarianceExplained(pca, number_of_significant_components)

    return pca_df, pca, pca_transformed
	

def pcaCalcOneExp(data, exp, title='', FigureNumber=None, nColor=0, nShades=0, show=True):
    with st.spinner("Calculating PCA and plotting..."):
        pca = decomposition.PCA(random_state=42)
        pca.fit(data)
        pca_transformed = pca.transform(data)
        number_of_significant_components = sum(pca.explained_variance_ratio_ > 0.1)
        pca_df = pd.DataFrame(pca_transformed[:, 0:2], index=exp.index)
        pca_df.rename(columns={0: 'PC1', 1: 'PC2'}, inplace=True)
        pca_df['Experiment'] = [expNames.replace('NNN0', '') for expNames in exp]
        if show:
            st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
            FigureNumber += 1
            pcaPlot(pca, pca_df, pca_df['Experiment'], title, nColor, nShades)
            st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
            FigureNumber += 1
            pcaVarianceExplained(pca, number_of_significant_components)
    return pca_df, pca, pca_transformed


def pcaCalc(data, expNamesInOrder, title, nColor=1, nShades=2):
    with st.spinner("Calculating PCA and plotting..."):
        pca = decomposition.PCA(random_state=42)
        pca.fit(data)
        pca_transformed = pca.transform(data)
        number_of_significant_components = sum(pca.explained_variance_ratio_ > 0.1)
        pca_df = pd.DataFrame(pca_transformed[:, 0:2], index=expNamesInOrder.index.values)
        pca_df.rename(columns={0: 'PC1', 1: 'PC2'}, inplace=True)
        pca_df['Experiment'] = [expNames.replace('NNN0', '') for expNames in expNamesInOrder]
        pcaPlot(pca, pca_df, pca_df['Experiment'], title, nColor, nShades)
    return None
	

def ElbowGraph(pca, pca_transformed):
    with st.spinner("Generating elbow plot for optimal K..."):
        f, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)
        number_of_significant_components = sum(pca.explained_variance_ratio_ > 0.1)
        pca_transformed_n = pca_transformed[:, 0:number_of_significant_components]
        n_clusters_i = range(2, 16)
        PC_col = ['PC' + str(x) for x in range(1, number_of_significant_components + 1)]
        kmeans_pca = pd.DataFrame(pca_transformed_n, columns=PC_col)

        distortions = []
        for i in n_clusters_i:
            kmeanModel = KMeans(n_clusters=i, random_state=0, verbose=0).fit(pca_transformed_n)
            kmeans_pca['Groups'] = kmeanModel.predict(pca_transformed_n)
            distortions.append(
                sum(np.min(cdist(pca_transformed_n, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / pca_transformed_n.shape[0]
            )

        ax.plot(n_clusters_i, distortions, 'bx-')
        ax.set_xlabel('K Clusters', fontsize=15)
        ax.set_ylabel('Distortion', fontsize=15)
        ax.set_title('Elbow Method to find the Optimal K', fontweight='bold', fontsize=25)

        st.pyplot(f)
        plt.close(f)
    return None


def kmeansPlot(k_cluster, pca_transformed, pca, dataLabel):
    with st.spinner("Running k-means clustering and plotting..."):
        number_of_significant_components = sum(pca.explained_variance_ratio_ >= 0.1)
        if number_of_significant_components < 2:
            number_of_significant_components = 2

        pca_transformed_n = pca_transformed[:, 0:number_of_significant_components]
        f, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200, facecolor='w', edgecolor='k')
        pca_expln_var_r = pca.explained_variance_ratio_ * 100
        PC_col = ['PC' + str(x) for x in range(1, number_of_significant_components + 1)]
        kmeans_pca = pd.DataFrame(pca_transformed_n, columns=PC_col, index=dataLabel.index)
        kmeanModel = KMeans(n_clusters=k_cluster, random_state=0).fit(pca_transformed_n)
        idx = np.argsort(kmeanModel.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(k_cluster)
        kmeans_pca['Groups'] = lut[kmeanModel.predict(pca_transformed_n)]
        num_of_dep = len(kmeans_pca['Groups'].unique())
        sns.despine(f, left=True, bottom=True)
        palette = sns.color_palette("hls", num_of_dep)  # Choose color
        s = sns.scatterplot(x="PC1", y="PC2", hue='Groups', data=kmeans_pca, ax=ax,
                            legend='full', palette=palette)
        plt.suptitle('K-means clustering k=' + '{0:.0f}'.format(k_cluster), fontweight='bold', fontsize=25)
        plt.xlabel('PC1 (' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%)', fontsize=15)
        plt.ylabel('PC2 (' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%)', fontsize=15)

        # splitting the legend list into few columns
        legend_texts = ax.get_legend().texts if ax.get_legend() else []
        if len(legend_texts) > 25:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=3, framealpha=1, edgecolor='black')
        elif len(legend_texts) > 17:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2, framealpha=1, edgecolor='black')
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1, framealpha=1, edgecolor='black')
        xlim_kmeans_l, xlim_kmeans_r = plt.xlim()
        ylim_kmeans_l, ylim_kmeans_r = plt.ylim()
        xlim_kmeans = [xlim_kmeans_l, xlim_kmeans_r]
        ylim_kmeans = [ylim_kmeans_l, ylim_kmeans_r]
        centers = kmeanModel.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=25)
        for spine in ax.spines.values():
            spine.set_visible(True)
        st.pyplot(f)
        plt.close(f)
    return kmeans_pca, xlim_kmeans, ylim_kmeans


def histogramData(exp, data, Features, FigureNumber):
    with st.spinner("Generating histogram, please wait..."):
        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        nrows, ncols = get_rows_cols(len(Features))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40, 30), dpi=200)
        colors = sns.color_palette("hls", len(exp))
        for par, ax in zip(Features, axes.flat):
            for label, color in zip(range(len(exp)), colors):
                vals = np.float64(data[par].loc[data['Experiment'] == exp[label]])
                ax.hist(
                    vals,
                    label=exp[label], color=color, density=True, stacked=True,
                )
                ax.set_xlabel(par,)
        fig.set_tight_layout(True)

        handles, labels = ax.get_legend_handles_labels()
        fig.set_tight_layout(False)

        fig.tight_layout(pad=1.05)
        fig.legend(handles, labels, loc='upper right', fontsize='xx-large', framealpha=1, edgecolor='black')
        plt.subplots_adjust(right=0.8, top=0.9)

        st.pyplot(fig)
        plt.close(fig)
    return None


def histogramDataKDE(exp, data, Features, FigureNumber, nColor=0, nShades=0):
    with st.spinner(f"Generating KDE histogram for Figure {FigureNumber}..."):
        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        nrows, ncols = get_rows_cols(len(Features))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25., 25.), dpi=100)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if nColor == 0:
            colors = sns.color_palette("hls", len(exp))
        else:
            colors = ChoosePalette(nColor, nShades)
        for par, ax in zip(Features, axes.flat):
            for label, color in zip(range(len(exp)), colors):
                vals = np.float64(data[par].loc[data['Experiment'] == exp[label]])
                sns.kdeplot(vals, ax=ax,
                            label=exp[label], color=color,
                            )
                ax.set_xlabel(par, fontdict={'fontsize': 15})
        labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}
        for a in axes.flat:
            try:
                a.get_legend().remove()
            except Exception:
                pass
        fig2.legend(labels_handles.values(),
                    labels_handles.keys(),
                    loc='center', fontsize='xx-large',
                    framealpha=1, edgecolor='black'
                    )
        fig.tight_layout(pad=1.01)
        fig.subplots_adjust(top=0.85)
        st.pyplot(fig)
        st.pyplot(fig2)
    return None


def AEdata(data, labelsCol, LE, shape, nrows=0, nColor=0, nShades=0,
    nColorTreat=0, nShadesTreat=0, nColorLay=0, nShadesLay=0,
    labelsLay='', figsize=(15,10), labelsTime='', AE_model=False, model_name=''):
    with st.spinner("Running AutoEncoder and plotting latent space..."):
        np.random.seed(42)
        dataNumeric = data.drop(columns=['Experiment','Treatments','Layers','TimeLayers']).copy()
        Features = dataNumeric.columns
        for par in Features:
            dataNumeric[par] = dataNumeric[par]/(dataNumeric[par].abs().max())
        X = dataNumeric.values.copy()

        if AE_model:
            input_layer = Input(shape=(shape,))
            encoding_layer1 = Dense(16, activation='elu')(input_layer)
            encoding_layer2 = Dense(4, activation='elu')(encoding_layer1)
            encoding_layer = Dense(2, activation='elu')(encoding_layer2)
            decoding_layer1 = Dense(4, activation='elu')(encoding_layer)
            decoding_layer2 = Dense(16, activation='elu')(decoding_layer1)
            decoding_layer = Dense(shape, activation='elu')(decoding_layer2)
            autoencoder = Model(input_layer, decoding_layer)
            autoencoder.compile(optimizer='adadelta', loss='mse',)
            autoencoder.fit(x=X, y=X, epochs=10)
            encoder = Model(input_layer, encoding_layer)
            now = datetime.now()
            date_time = now.strftime("d%d%m%yh%H%M%S")
            model_json_autoencoder = autoencoder.to_json()
            model_json_encoder = encoder.to_json()
            with open("model/model ae "+date_time+".json", "w") as file_json:
                file_json.write(model_json_autoencoder)
            with open("model/model e "+date_time+".json", "w") as file_json:
                file_json.write(model_json_encoder)
            autoencoder.save_weights("model/model ae "+date_time+".h5")
            encoder.save_weights("model/model e "+date_time+".h5")
            st.info("Saved AE model to disk")
        else:
            ae_json_file = open("model/model ae "+model_name+'.json', 'r')
            loaded_model_ae_json = ae_json_file.read()
            ae_json_file.close()
            e_json_file = open("model/model e "+model_name+'.json', 'r')
            loaded_model_e_json = e_json_file.read()
            e_json_file.close()
            ae_loaded_model = model_from_json(loaded_model_ae_json)
            e_loaded_model = model_from_json(loaded_model_e_json)
            ae_loaded_model.load_weights("model/model ae "+model_name+'.h5')
            e_loaded_model.load_weights("model/model e "+model_name+'.h5')
            autoencoder = ae_loaded_model
            encoder = e_loaded_model
            st.info("Loaded AE model from disk")
        encodings = encoder.predict(X)
        AE_df = pd.DataFrame(columns=['Encoder 0','Encoder 1'])
        for par, le in zip(labelsCol, LE):
            fig, ax = plt.subplots(figsize=(6,6), dpi=200)
            if par == 'Experiment' and nColor != 0:
                palette = ChoosePalette(nColor, nShades)
            elif par == 'Treatments' and nColorTreat != 0:
                palette = ChoosePalette(nColorTreat, nShadesTreat)
            elif par == 'Layers' and nColorLay != 0:
                palette = ChoosePalette(nColorLay, nShadesLay)
            else:
                num_of_dep = len(data[par].unique())
                palette = sns.color_palette("hls", num_of_dep)
            if le:
                lb_make = LabelEncoder()
                labels = lb_make.fit_transform(data[par])
                uLabel = [u.replace('NNN0','') for u in data[par].unique()]
            else:
                if par == 'Layers':
                    labels = data[par].values.copy()
                    uLabel = labelsLay
                elif par == 'TimeLayers':
                    labels = data[par].values.copy()
                    uLabel = labelsTime
                else:
                    labels = data[par].values.copy()
                    uLabel = data[par].unique().tolist()
            AE_df['Encoder 0'] = encodings[:, 0]
            AE_df['Encoder 1'] = encodings[:, 1]
            AE_df[par] = labels
            s = sns.scatterplot(x='Encoder 0', y='Encoder 1', hue=par, data=AE_df, palette=palette, ax=ax)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., framealpha=1, edgecolor='black')
            for t, l in zip(s.get_legend().texts, [par]+uLabel): t.set_text(l)
            st.pyplot(fig)
            plt.close(fig)
        AE_colorPar1_titlePar2(AE_df, Par1='', Par2='Treatments', figsize=figsize,
                                nrows=nrows, nColor=nColor, nShades=nShades,
                                nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,
                                nColorLay=nColorLay, nShadesLay=nShadesLay)
        AE_colorPar1_titlePar2(AE_df, Par1='', Par2='TimeLayers', figsize=(15,5),
                                nrows=1, nColor=nColor, nShades=nShades,
                                nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,
                                nColorLay=nColorLay, nShadesLay=nShadesLay, labelsPar2=labelsTime)
        AE_colorPar1_titlePar2(AE_df, Par1='', Par2='Layers', figsize=(15,15),
                                nrows=0, nColor=nColor, nShades=nShades,
                                nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,
                                nColorLay=nColorLay, nShadesLay=nShadesLay, labelsPar2=labelsLay)
    return AE_df
        

def ChoosePalette(nColor=0,nShades=0):
    pickle_in = open("files\\palette.pickle", "rb")
    palette = pickle.load(pickle_in)  # Choose color with shades by treatment/experiment
    palette_col = []
    for pal,i in zip(palette,range(0,len(palette))):

        if i<nColor:
            if nShades==8:
                palette_col.extend(pal[0::4])
            elif nShades==7:
                palette_col.extend(pal[0::5])
            elif nShades==6:
                palette_col.extend(pal[0::6])
            elif nShades==5:
                palette_col.extend(pal[0::7])
            elif nShades==4:
                palette_col.extend(pal[0::10])
            elif nShades==3:
                palette_col.extend(pal[0::14])
            elif nShades==2:
                palette_col.extend(pal[0::24])
            elif nShades==1:
                palette_col.extend(pal[11::50])
    
    return palette_col


# def ChoosePaletteLarge(nColor=0,nShades=0):
    # palette = sns.color_palette('hls',nColor)

def GaussianMM(pca_transformed, title='', n_components=3):
    with st.spinner("Fitting Gaussian Mixture Model and plotting..."):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(pca_transformed[:, 0:2])

        # Predictions from GMM
        labels = gmm.predict(pca_transformed[:, 0:2])
        frame = pd.DataFrame(pca_transformed[:, 0:2])
        frame['GMM'] = labels
        frame.columns = ['PC1', 'PC2', 'GMM']

        num_of_dep = len(frame['GMM'].unique())
        palette = sns.color_palette("hls", num_of_dep)  # Choose color
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)
        sns.scatterplot(x='PC1', y='PC2', hue='GMM', palette=palette, data=frame, ax=ax)
        plt.title(title)
        st.pyplot(fig)
        plt.close(fig)
    return None


def histByKmeansTreatsLabel(pca_df, Par='TreatmentsLabels', k_cluster=3, bar_width=0.2, figsize=(10,5), labels='', rotate=0):
    with st.spinner("Generating histogram by k-means groups and treatments..."):
        fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
        Treat = pca_df.groupby(Par)
        rangeLabel = [float(i) for i in list(Treat.describe().index.values)]
        rangeLabelX = [float(i) + bar_width for i in list(Treat.describe().index.values)]
        for j in list(Treat.describe().index.values):
            T = Treat.get_group(j)['Groups']
            xlabels = T.unique()
            xlabels.sort()
            N = len(xlabels)
            color = sns.color_palette('hls', k_cluster)
            xrange = range(N)
            SUM = T.value_counts().sort_index().sum()
            for i in range(k_cluster):
                Group = T.loc[T == i].value_counts().sort_index()
                if len(xlabels) == k_cluster:
                    plt.bar(
                        rangeLabel[j] + i * bar_width, Group / SUM, bar_width,
                        label='Group ' + str(i), color=color[i]
                    )
                else:
                    for e in xlabels:
                        if e not in Group:
                            Group[e] = 0
                    Group.sort_index()
                    plt.bar(
                        rangeLabel[j] + i * bar_width, Group / SUM, bar_width,
                        label='Group ' + str(i), color=color[i]
                    )
        if labels == '':
            plt.xticks(xrange + bar_width, (xlabels), rotation=rotate, fontsize=12)
        else:
            labels.sort()
            plt.xticks(rangeLabelX, (labels), rotation=rotate, fontsize=12)

        handles, legend_labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, legend_labels)) if l not in legend_labels[:i]]
        if unique:
            ax.legend(*zip(*unique), bbox_to_anchor=(1.15, 1), framealpha=1, edgecolor='black')
        plt.xlabel(Par, fontsize=15)
        plt.suptitle('each ' + Par + ' is 100%', fontsize=15)
        st.pyplot(fig)
        plt.close(fig)
    return None
   
    
def PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize, labelsPar1='', labelsPar2='',
    nrows=0, nColor=0, nShades=0, nColorTreat=0, nShadesTreat=0,
    nColorLay=0, nShadesLay=0, xlim_kmeans=[-10,10], ylim_kmeans=[-10,10]):
    with st.spinner(f"Generating PCA grid: {Par2} by {Par1}..."):
        fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='k')
        title = 'Each graph is a seperate ' + Par2 + ' and colored by ' + Par1
        num_of_dep = len(pca_df[Par1].unique())
        if Par1 == 'Experiment' and nColor != 0:
            palette = ChoosePalette(nColor, nShades)
        elif Par1 == 'Treatments' and nColorTreat != 0:
            palette = ChoosePalette(nColorTreat, nShadesTreat)
        elif Par1 == 'Layers' and nColorLay != 0:
            palette = ChoosePalette(nColorLay, nShadesLay)
        else:
            palette = sns.color_palette("hls", num_of_dep)  # Choose color
        pca_expln_var_r = pca.explained_variance_ratio_ * 100
        uPar2 = pca_df[Par2].unique()
        uPar2.sort()
        uPar1 = list(pca_df[Par1].unique())
        uPar1.sort()
        if nrows == 0:
            nrows = int(np.floor(np.sqrt(len(uPar2))))
            if nrows <= 1:
                nrows = 1
                ncols = len(uPar2)
            elif nrows ** 2 == len(uPar2):
                nrows = nrows
                ncols = nrows
            else:
                if len(uPar2) % nrows == 0:
                    ncols = int(len(uPar2) / nrows)
                else:
                    ncols = nrows + 1
                    nrows += 1
        else:
            ncols = int(len(uPar2) / nrows)
        fig, ax = plt.subplots(figsize=figsize, nrows=nrows,
                               ncols=ncols, constrained_layout=True, dpi=200)
        if type(ax) != np.ndarray:
            ax = np.array([ax])
        for i, a, t in zip(uPar2, ax.reshape(-1), range(len(uPar2))):
            df_PCA = pca_df.loc[pca_df[Par2] == i].copy()
            uPar1tmp = df_PCA[Par1].unique()
            uPar1tmp.sort()
            pal = sns.color_palette([palette[p] for p in [uPar1.index(value) for value in uPar1tmp]])
            sns.despine(fig, left=True, bottom=True)
            s = sns.scatterplot(x="PC1", y="PC2", hue=Par1, data=df_PCA, ax=a, palette=pal)
            if labelsPar2 == '':
                a.set_title(i, fontweight='bold', fontsize=15)
            else:
                a.set_title(labelsPar2[t], fontweight='bold', fontsize=15)
            a.set_xlabel('PC1 ' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%')
            a.set_ylabel('PC2 ' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%')
            a.set_xlim(xlim_kmeans)
            a.set_ylim(ylim_kmeans)
        if labelsPar1 != '':
            labels_handles = {
                lPar1: handle for ax in fig.axes for handle, label, lPar1 in zip(*ax.get_legend_handles_labels(), labelsPar1)
            }
        else:
            labels_handles = {
                label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
            }
        for a in ax.flat:
            try:
                a.get_legend().remove()
                for spine in a.spines.values():
                    spine.set_visible(True)
            except Exception:
                pass

        keys = list(labels_handles.keys())
        values = list(labels_handles.values())

        # If Layers/TimeLayers is truly a title, remove it and set as legend title
        legend_title = None
        if keys[0] in ["Layers", "TimeLayers"]:
            legend_title = keys.pop(0)
            values.pop(0)

        
        if legend_title in labels_handles_dic:
            fig2.legend(
                labels_handles_dic[legend_title][0],
                labels_handles_dic[legend_title][1],
                title=legend_title,
                loc='center', fontsize='xx-large',
                framealpha=1, edgecolor='black'
            )
        else:
            fig2.legend(
                values, keys,
                title=legend_title,
                loc='center', fontsize='xx-large',
                framealpha=1, edgecolor='black'
            )

        fig2.subplots_adjust(right=0.5, top=0.9)
        plt.suptitle(title, fontweight='bold', fontsize=25)
        st.pyplot(fig)
        st.pyplot(fig2)
        plt.close(fig)
        plt.close(fig2)
    return None


def pcaPlotLabel(pca, pca_df, hue, title, labels, nColorLay=0, nShadesLay=0, xlim_kmeans=[-10,10], ylim_kmeans=[-10,10]):
    with st.spinner("Generating PCA plot with labels..."):
        f, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200, facecolor='w', edgecolor='k')
        num_of_dep = len(hue.unique())
        sns.despine(f, left=True, bottom=True)
        if hue.name == 'Layers' and nColorLay != 0:
            palette = ChoosePalette(nColorLay, nShadesLay)
        else:
            palette = sns.color_palette("hls", num_of_dep)  # Choose color
        pca_expln_var_r = pca.explained_variance_ratio_ * 100

        s = sns.scatterplot(
            x="PC1", y="PC2", hue=hue.name, data=pca_df,
            ax=ax, palette=palette
        )
        handles, _ = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        leg_title = ""
        if (len(handles) + 1) == len(labels):
            leg_title = labels[0]
            labels = labels[1:]

        if (leg_title not in labels_handles_dic) and (leg_title != ""):
            labels_handles_dic[leg_title] = (handles,labels)
        

        f.legend(
            handles, labels,title=leg_title, bbox_to_anchor=(1.02, 1), loc=2,
            borderaxespad=0., framealpha=1, edgecolor='black'
        )
        plt.subplots_adjust(right=0.75)
        plt.suptitle(title, fontweight='bold', fontsize=15)
        plt.xlim(xlim_kmeans)
        plt.ylim(ylim_kmeans)
        plt.xlabel('PC1 ' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%')
        plt.ylabel('PC2 ' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%')
        for spine in ax.spines.values():
            spine.set_visible(True)
        st.pyplot(f)
        plt.close(f)
    return None


def printInterval(tup, unit, prec=0, mul=1.):
    label ='({:.{prec}f},{:.{prec}f}] '.format(tup[0]*mul,tup[1]*mul,prec=prec) + unit
    return label


def roundInterval(IntervalIdx, unit, prec=0, mul=1.):
    tuplesList = IntervalIdx.cat.categories.to_tuples()
    labels = list([printInterval(tuple([round(x,prec) if isinstance(x, float) \
                                        else float(x) for x in l]),unit,prec,mul) for l in tuplesList])
    return labels


# def ChiSqaureData(pca_df,Par='TreatmentsLabels',k_cluster=3,labels=''):
    # Treat = pca_df.groupby(Par)
    # chi_df = pd.DataFrame(index=range(k_cluster))
    # for j,la in zip(list(Treat.describe().index.values),labels):
        # T = Treat.get_group(j)['Groups']
        # xlabels = T.unique()
        # xlabels.sort()
        # N = len(xlabels)
        # xrange = range(N)
        # SUM = T.value_counts().sort_index().sum()
        # groups_df = []
        # for i in range(k_cluster):
            # Group = T.loc[T==i].value_counts().sort_index()
            # if len(xlabels)==k_cluster:
                # groups_df.append(Group[i])
            # else:
                # if i not in Group:
                    # Group[i] = 0
                    # groups_df.append(0)
                # else:
                    # groups_df.append(Group[i])
                # Group.sort_index()
        # chi_df[la] = groups_df

    # return chi_df
    

def chi_square_test_tables(pca_df, labels,f, expectation='CON', Par='Treatments'):

    with st.spinner("Running chi-square tests and generating table..."):
        try:
            chi_table = pd.DataFrame()
            for la in labels:
                if la != expectation:
                    # Combine data for expectation and current label
                    chi_df = pd.DataFrame(
                        pca_df[Par].loc[pca_df[Par] == expectation].values.tolist() +
                        pca_df[Par].loc[pca_df[Par] == la].values.tolist(),
                        columns=[Par]
                    )
                    chi_df['Groups'] = (
                        pca_df['Groups'].loc[pca_df[Par] == expectation].values.tolist() +
                        pca_df['Groups'].loc[pca_df[Par] == la].values.tolist()
                    )
                    # Run chi-square test
                    table, results = rp.crosstab(
                        chi_df[Par],
                        chi_df['Groups'],
                        prop='row',
                        test='chi-square'
                    )
                    pvalue = results.at[1, 'results']
                    # Fill chi_table with results
                    for i in table.columns:
                        chi_table.at[la, i[1]] = table[i].loc[la]
                    chi_table.at[la, 'N'] = pca_df[Par].loc[pca_df[Par] == la].size
                    for i in results['Chi-square test'].values:
                        chi_table.at[la, i] = results.at[
                            results['Chi-square test'].loc[results['Chi-square test'] == i].index.values.tolist()[0],
                            'results'
                        ]
            # Add expectation and total counts
            for i in table.columns:
                chi_table.at[expectation, i[1]] = table[i].loc[expectation]
            chi_table.at[expectation, 'N'] = pca_df[Par].loc[pca_df[Par] == expectation].size
            chi_table.at['All', 'N'] = chi_table['N'].sum()
            st.dataframe(chi_table)
            f.write(chi_table.to_html(index=True))
            return None
        except Exception as e:
            print(f'couldnt generate chi squared test table for {e}')

def fix_old_no_pca_var(dataDrop, dataLabel, title='',FigureNumber=None, nColor=0, nShades=0,show=False):
    standardScaler = StandardScaler(with_std=True,)
    dataDrop = pd.DataFrame(standardScaler.fit_transform(dataDrop),columns=dataDrop.columns)
    _, pca, _ = pcaCalcOneExp(dataDrop, dataLabel['Experiment'], 'PCA of '+\
                                               title,FigureNumber, nColor=nColor, nShades=nShades,show=False)
    return pca
    

def histogramDataKDELabels(Labels, data, Features, FigureNumber, Par='Experiment', nColor=0, nShades=0):
    with st.spinner("Generating KDE histogram by group..."):
        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        nrows, ncols = get_rows_cols(len(Features))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 30), dpi=200)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        if nColor == 0:
            colors = sns.color_palette("hls", len(Labels))
        else:
            colors = ChoosePalette(nColor, nShades)
        for par, ax in zip(Features, axes.flat):
            for label, color in zip(range(len(Labels)), colors):
                try:
                    vals = np.float64(data[par].loc[data[Par] == Labels[label]])
                    sns.kdeplot(vals, ax=ax, label=Labels[label], color=color)
                except Exception:
                    vals = np.float64(data[par].loc[data[Par] == Labels[label]])
                    sns.kdeplot(vals, ax=ax, label=Labels[label], color=color, bw=50)
                ax.set_xlabel(par)
        fig.set_tight_layout(True)
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.set_tight_layout(False)
        for a in axes.flat:
            try:
                a.get_legend().remove()
            except Exception:
                continue
        fig.tight_layout(pad=1.01)
        fig2.legend(handles, legend_labels, loc='upper right', fontsize='xx-large', framealpha=1, edgecolor='black')
        st.pyplot(fig)
        st.pyplot(fig2)
        plt.close(fig)
        plt.close(fig2)
    return None


def Layers(data, q, labeling, unit, prec=0, mul=1.):
    data_numeric = pd.to_numeric(data, errors='coerce')
    LayX = pd.qcut(x=data_numeric, q=q, labels=labeling)
    LayXIntervals = pd.qcut(x=data_numeric, q=q)
    labels = roundInterval(LayXIntervals, unit, prec, mul)
    return LayX, LayXIntervals, labels


def histByKmeansTreats(pca_df, Par='TimeIndex', k_cluster=3, bar_width=0.2, figsize=(10,5), labels='', rotate=0):
    with st.spinner("Generating histogram by k-means groups and treatments..."):
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        plt.title(Par + ' Histogram by group', fontsize=15)
        xlabels = pca_df[Par].unique()
        xlabels.sort()
        N = len(xlabels)
        xrange = np.arange(N)
        color = sns.color_palette('hls', k_cluster)
        for i in range(k_cluster):
            Group = pca_df[Par].loc[pca_df['Groups'] == i].value_counts().sort_index()
            if len(Group) == N:
                plt.bar( xlabels + i * bar_width, Group / Group.sum(), bar_width,label='Group ' + str(i), color=color[i])
            else:
                for e in xlabels:
                    if e not in Group:
                        Group[e] = 0
                Group.sort_index()
                plt.bar(xlabels + i * bar_width, Group / Group.sum(), bar_width,label='Group ' + str(i), color=color[i])

        plt.legend(bbox_to_anchor=(1.2, 1), framealpha=1, edgecolor='black')
        if labels == '':
            plt.xticks(xrange + bar_width, (xlabels), rotation=rotate, fontsize=12)
        else:
            labels.sort()
            plt.xticks(xlabels + bar_width, (labels), rotation=rotate, fontsize=12)
        plt.xlabel(Par + ' range', fontsize=15)
        st.pyplot(fig)
        plt.close(fig)
    return None


def histByKmeans(pca_df, Par='TimeIndex', k_cluster=3, bar_width=0.2, figsize=(10,5), labels='', rotate=0):
    with st.spinner("Generating histogram by k-means groups..."):
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        plt.title(Par + ' Histogram by group', fontsize=20)
        xlabels = pd.to_numeric(pca_df[Par].unique(),errors='coerce')
        xlabels.sort()
        xrange = np.arange(0, len(xlabels), 1)
        for i in range(k_cluster):
            Group = pca_df[Par].loc[pca_df['Groups'] == i].value_counts().sort_index()
            plt.bar(Group.index.values + i * bar_width, Group / Group.sum(), bar_width, label='Group ' + str(i))
        plt.legend(bbox_to_anchor=(1.2, 1), framealpha=1, edgecolor='black')
        if labels == '':
            plt.xticks(xlabels + bar_width, (xlabels), rotation=rotate, fontsize=12)
        else:
            plt.xticks(xlabels + bar_width, (labels), rotation=rotate, fontsize=12)
        plt.xlabel(Par + ' range', fontsize=15)
        st.pyplot(fig)
        plt.close(fig)
    return None

  
def analysisExp(dataAll, dataAllGraph, expList, expNamesInOrderU):
    with st.spinner("Analyzing experiments, please wait..."):
        # Test for valid input in expList
        try:
            exp = expNamesInOrderU[expList]
        except IndexError as e:
            bad_arg = str(e).split(" ")[1]
            e.args = tuple([
                f"You entered {bad_arg} in expList, but that was out of range. Please go back to 'Sample cells' and make sure to replace/delete it and run again."
            ]) + e.args[1:]
            raise

        dataSpec, dataSpecGraph = extractSpecificExp(dataAll, dataAllGraph, exp)

        dataSpec['Experiment'] = dataSpec['Experiment'].replace(to_replace='NNN0', value='', regex=True)
        dataSpecGraph['Experiment'] = dataSpecGraph['Experiment'].replace(to_replace='NNN0', value='', regex=True)
        Features = dataSpec.columns.copy()

        st.markdown('**The experiment(s) you chose:**')
        for e in exp:
            st.write(e.replace('NNN0',''))
        st.markdown('**The cell line(s):**')
        for e in exp:
            st.write((e.replace('NNN0',''))[18:22])
        st.markdown('**The treatments are:**')
        st.write("Number of features:", len(Features)-1)
        st.markdown("**The Features are:**")
        for i, Feature in zip(range(1, len(Features)), Features):
            st.write(f"{i}. {Feature}")

        exp = [e.replace('NNN0','') for e in exp]
        return exp, Features, dataSpec, dataSpecGraph
 
def quatileData(data, Par):
    quantile_list = [0, 0.1, .25, 0.41, .6, .85, 0.95, 1.]
    quantiles = data[Par].quantile(quantile_list)

    quantile_labels = [0, 1, 2, 3, 4, 5, 6]
    data[Par] = list(pd.qcut(
                        data[Par], 
                        q=quantile_list,       
                        labels=quantile_labels).values.copy())
    # plt.show()
    return None
    
    
def createPossibleCOMB(combin=None):
    if combin==None:
        return print('Error')
    prod = list(product(range(2), repeat=len(combin)))
    lenRange = len(prod)
    combPossible = []
    for i in range(lenRange-1):
            tmp = list(product(*compress(combin,prod[i+1])))
            if len(tmp[0])==1:
                for j in range(len(tmp)):
                    combPossible.append([tmp[j][0]])
            else:
                combPossible.append([list(elem) for elem in tmp])

    combPossible.sort(key=len)
    combPossible.reverse()
    return combPossible


def AE_colorPar1_titlePar2(AE_df, Par1='', Par2='', figsize=(15,10), labelsPar1='', labelsPar2='',
    nrows=0, nColor=0, nShades=0, nColorTreat=0, nShadesTreat=0,
    nColorLay=0, nShadesLay=0):
    with st.spinner(f"Generating AE latent space grid: {Par2} by {Par1}..."):
        title = 'Each graph is a seperate ' + Par2 + ' and colored by ' + Par1
        if Par1 != '':
            uPar1 = list(AE_df[Par1].unique())
            uPar1.sort()
            num_of_dep = len(AE_df[Par1].unique())
            if Par1 == 'Experiment' and nColor != 0:
                palette = ChoosePalette(nColor, nShades)
            elif Par1 == 'Treatments' and nColorTreat != 0:
                palette = ChoosePalette(nColorTreat, nShadesTreat)
            elif Par1 == 'Layers' and nColorLay != 0:
                palette = ChoosePalette(nColorLay, nShadesLay)
            else:
                palette = sns.color_palette("hls", num_of_dep)
        else:
            uPar1 = AE_df[Par2].unique()
            uPar1.sort()
            pal = sns.color_palette("hls", 1)
        uPar2 = AE_df[Par2].unique()
        uPar2.sort()

        if nrows == 0:
            nrows = int(np.floor(np.sqrt(len(uPar2))))
            if nrows <= 1:
                nrows = 1
                ncols = len(uPar2)
            elif nrows ** 2 == len(uPar2):
                nrows = nrows
                ncols = nrows
            else:
                if len(uPar2) % nrows == 0:
                    ncols = int(len(uPar2) / nrows)
                else:
                    ncols = nrows + 1
                    nrows += 1
        else:
            ncols = int(len(uPar2) / nrows)
        fig, ax = plt.subplots(figsize=figsize, nrows=nrows,
                               ncols=ncols, constrained_layout=True, dpi=200)
        for i, a, t in zip(uPar2, ax.reshape(-1), range(len(uPar2))):
            df_AE = AE_df.loc[AE_df[Par2] == i].copy()
            if Par1 != '':
                uPar1tmp = df_AE[Par1].unique()
                uPar1tmp.sort()
                pal = sns.color_palette([palette[p] for p in [uPar1.index(value) for value in uPar1tmp]])
                sns.despine(fig, left=True, bottom=True)
                s = sns.scatterplot(x="Encoder 0", y="Encoder 1", hue=Par1, data=df_AE, ax=a, palette=pal)
            else:
                sns.despine(fig, left=True, bottom=True)
                s = sns.scatterplot(x="Encoder 0", y="Encoder 1", data=df_AE, ax=a)
            if labelsPar2 == '':
                a.set_title(i, fontweight='bold', fontsize=15)
            else:
                a.set_title(labelsPar2[t], fontweight='bold', fontsize=15)
            a.set_xlabel('Encoder 0')
            a.set_ylabel('Encoder 1')
            a.set_xlim([
                AE_df['Encoder 0'].min() + 0.1 * AE_df['Encoder 0'].min(),
                AE_df['Encoder 0'].max() + 0.1 * AE_df['Encoder 0'].max()
            ])
            a.set_ylim([
                AE_df['Encoder 1'].min() + 0.1 * AE_df['Encoder 1'].min(),
                AE_df['Encoder 1'].max() + 0.1 * AE_df['Encoder 1'].max()
            ])

        if labelsPar1 != '':
            labels_handles = {
                lPar1: handle for ax in fig.axes for handle, label, lPar1 in zip(*ax.get_legend_handles_labels(), labelsPar1)
            }
        else:
            labels_handles = {
                label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
            }

        fig.legend(
            labels_handles.values(),
            labels_handles.keys(),
            loc="upper center",
            bbox_to_anchor=(1.2, 1),
            bbox_transform=plt.gcf().transFigure,
            framealpha=1, edgecolor='black'
        )
        if Par1 != '':
            for a in ax.flat:
                a.get_legend().remove()
        plt.suptitle(title, fontweight='bold', fontsize=25)
        st.pyplot(fig)
        plt.close(fig)
    return None

def DescriptiveTable(dataSpecGraphGroups, title,f):
    with st.spinner("Generating descriptive statistics table..."):
        # Select only numeric columns (but keep 'Groups' for grouping)
        numeric_cols = dataSpecGraphGroups.select_dtypes(include=[np.number]).columns.tolist()
        if 'Groups' not in numeric_cols and 'Groups' in dataSpecGraphGroups.columns:
            numeric_cols = ['Groups'] + numeric_cols
        # Group by 'Groups' and aggregate only numeric columns
        gb_Groups = dataSpecGraphGroups[numeric_cols].groupby(['Groups']).agg([
            'count',
            'mean',
            'std',
            ('sem', sem),
            ('ci95_hi', lambda x: (np.mean(x) + 1.96 * np.std(x) / math.sqrt(np.size(x)))),
            ('ci95_lo', lambda x: (np.mean(x) - 1.96 * np.std(x) / math.sqrt(np.size(x))))
        ])

        gb_Groups.rename(columns={
            "std": "std. Deviation",
            'count': 'N',
            'sem': 'std. Error',
            'mean': 'Mean',
            'ci95_hi': '95 confidence Interval for Mean Upper Bound',
            'ci95_lo': '95 confidence Interval for Mean Lower Bound'
        }, inplace=True)

        gb_Groups.to_csv(title + ' Descriptive Table - Groups.csv')
        st.dataframe(gb_Groups)
        f.write(gb_Groups.to_html(index=True))
    return None

def ANOVE_DESC_TABLE(dataSpecGraphGroups, Features, title,f, dep='Groups', groupList=[0,1,2]):
    with st.spinner("Generating ANOVA + Descriptive Table..."):
        st.latex(r"\color{blue}{\Large ANOVA\ Table\ feature\ per\ Group}")
        ANOVA_MI = pd.MultiIndex.from_product(
            [['Between '+dep, 'Within '+dep, 'Total'],
             ['Sum of Squares', 'df', 'Mean Sqaure', 'F', 'Sig.']]
        )
        ANOVA_df = pd.DataFrame(columns=ANOVA_MI, index=Features[:-1])
        index = pd.MultiIndex.from_product([Features[:-1], [0]], names=['Feature', 'Sig.'])
        columns = pd.MultiIndex.from_product(
            [[dep], groupList,
             ['N', 'Mean', 'Standard Deviation', 'Standard Deviation Error', '95% Upper Bound Mean', '95% Lower Bound Mean']]
        )
        ANOVA_Desc_df = pd.DataFrame(columns=columns, index=index)
        ANOVA_Desc_df = ANOVA_Desc_df.reset_index(level='Sig.')
        for par in Features[:-1]:
            model_name = ols(f"{par} ~ C({dep})", data=dataSpecGraphGroups).fit()
            ano = sm.stats.anova_lm(model_name, typ=1)
            total_row = pd.DataFrame({"df": [ano.df.sum()], "sum_sq": [ano.sum_sq.sum()]}, index=["Total"])
            ano = pd.concat([ano, total_row])
            ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
            ano.rename(columns={"sum_sq": "Sum of Squares", 'mean_sq': 'Mean Sqaure', 'PR(>F)': 'Sig.'}, inplace=True)
            ano.rename(index={'C('+dep+')': "Between "+dep, 'Residual': 'Within '+dep}, inplace=True)
            ANOVA_df.at[par] = ano.values.ravel()
            ANOVA_Desc_df.loc[par, 'Sig.'] = ANOVA_df.loc[par, 'Between '+dep]['Sig.']
            gb_Groups = dataSpecGraphGroups.groupby([dep])[par].agg([
                'count', 'mean', 'std', ('sem', sem),
                ('ci95_hi', lambda x: (np.mean(x) + 1.96 * np.std(x) / math.sqrt(np.size(x)))),
                ('ci95_lo', lambda x: (np.mean(x) - 1.96 * np.std(x) / math.sqrt(np.size(x))))
            ])
            ANOVA_Desc_df.loc[par, 'Groups'] = gb_Groups.values.ravel()
        st.dataframe(ANOVA_Desc_df)
        f.write(ANOVA_Desc_df.to_html(index=True))
        ANOVA_Desc_df.to_csv(title + ' ANOVA + Descriptive Table - ' + dep + '.csv')
    return ANOVA_Desc_df



# def ANOVA_by_Treatments(dataSpecGraphGroups,Features):
    # for par in Features[:-1]:
        # display(Latex('$\color{blue}{\Large %s}$'%(par)))
        # model_name = ols(par+' ~ C(Groups)', data=dataSpecGraphGroups).fit()
        # ano = sm.stats.anova_lm(model_name,typ=1)
        # ano = ano.append(pd.DataFrame({"df":[ano.df.sum()],"sum_sq":[ano.sum_sq.sum()]},index={"Total"}))
        # ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
        # ano.rename(columns = {"sum_sq": "Sum of Squares", 'mean_sq':'Mean Sqaure', 'PR(>F)':'Sig.'},inplace=True)
        # ano.rename(index = {"C(Groups)": "Between Groups", 'Residual':'Within Groups'},inplace=True)
        # display(ano)
    # return None

def ANOVA_TABLE(dataSpecGraphGroups, Features,f, title='', dep='Groups'):
    with st.spinner("Generating ANOVA Table..."):
        st.latex(r"\color{blue}{\Large ANOVA\ Table\ feature\ per\ Group}")
        ANOVA_MI = pd.MultiIndex.from_product(
            [['Between Groups', 'Within Groups', 'Total'],
             ['Sum of Squares', 'df', 'Mean Square', 'F', 'Sig.']]
        )
        ANOVA_df = pd.DataFrame(columns=ANOVA_MI, index=Features[:-1])
        for par in Features[:-1]:
            model_name = ols(f"{par} ~ C({dep})", data=dataSpecGraphGroups).fit()
            ano = sm.stats.anova_lm(model_name, typ=1)
            total_row = pd.DataFrame({"df": [ano.df.sum()], "sum_sq": [ano.sum_sq.sum()]}, index=["Total"])
            ano = pd.concat([ano, total_row])
            ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
            ano.rename(columns={"sum_sq": "Sum of Squares", 'mean_sq': 'Mean Square', 'PR(>F)': 'Sig.'}, inplace=True)
            ano.rename(index={'C('+dep+')': "Between "+dep, 'Residual': 'Within '+dep}, inplace=True)
            ANOVA_df.loc[par, :] = ano.values.ravel()
        ANOVA_df.dropna(axis=1, inplace=True)
        ANOVA_df.to_csv(title + ' ANOVA Table - ' + dep + '.csv')
        st.dataframe(ANOVA_df)
        f.write(ANOVA_df.to_html(index=True))
    return

def TASC(dataDrop, dataLabel, labelsCol='Experiment', LE=True,
    title='All Cell Lines', HC=False, treats=['HGF'], combTreats=[['HGF'],['PHA']],
    LY=9, TI=3, multipleCL=True, singleTREAT=False, FigureNumber=2, nrows=0, ncols=1,
    nColor=0, nShades=0, k_cluster=3,
    nColorTreat=0, nShadesTreat=0, nColorLay=0, nShadesLay=0,
    figsizeEXP=(15,40), figsizeTREATS=(15, 25),figsizeCL=(15, 25), Features='',
    AE_model=True, model_name=''):
    with st.spinner("Running TASC analysis..."):
        dataAE = dataDrop.copy()
        dataAE['Experiment'] = dataLabel['Experiment'].copy()
        # z-score data
        standardScaler = StandardScaler(with_std=True,)
        dataDrop = pd.DataFrame(standardScaler.fit_transform(dataDrop), columns=dataDrop.columns)
        if HC:
            st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
            FigureNumber += 1
            createHierarchicalCluster(
                dataDrop, title, 'Parameters', '# Cell',
                sns.diverging_palette(150, 10, n=100), vmin=-2, vmax=2
            )
        listCells = dataDrop.index.values.copy()
        pca_df, pca, pca_transformed = pcaCalcOneExp(
            dataDrop, dataLabel['Experiment'], 'PCA of ' + title, FigureNumber,
            nColor=nColor, nShades=nShades, show=True
        )
        FigureNumber += 2

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        kmeans_pca, xlim_kmeans, ylim_kmeans = kmeansPlot(
            k_cluster, pca_transformed, pca, dataLabel['Experiment']
        )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        pca_df['Time'] = dataLabel['TimeIndex'].copy()
        pcaPlot(pca, pca_df, pca_df['Time'], 'by Time points', nColor, nShades)

        pca_df['Experiment'] = dataLabel['Experiment']
        pca_df['Groups'] = kmeans_pca['Groups'].values.copy()

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        Par = 'CellLine'
        pca_df[Par] = [ex[18:18+4] for ex in pca_df['Experiment']]
        pcaPlot(
            pca, pca_df, pca_df[Par], 'by ' + Par, nColor, nShades,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        # Sign by groups/treatments
        combPossible = createPossibleCOMB(combTreats)
        COMB = []
        combPossible = [var if len(np.asarray(elem).shape) > 1 else elem for elem in combPossible for var in elem]

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1

        if combTreats is not None:
            for ex in pca_df['Experiment']:
                TF = False
                for comb in combPossible:
                    combTreat = ''.join(comb)
                    sumtreats = sum([st in ex for st in comb])
                    if len(comb) > 1 and TF is False:
                        if all(st in ex for st in comb):
                            COMB += [combTreat]
                            TF = True
                    elif TF is False and all(st in ex for st in comb):
                        COMB += [combTreat]
                        TF = True
                if TF is False:
                    COMB += ['CON']
            pca_df['Treatments'] = COMB
            pcaPlot(
                pca, pca_df, pca_df['Treatments'], 'by Treatments',
                nColorTreat, nShadesTreat, xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
            )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        precLY = 2
        unitLY = '$\mu$m'
        LayersY, LYIntervals, labelsY = Layers(
            dataLabel['y_Pos'], LY, range(-4, -4 + LY), unitLY, precLY
        )
        pca_df['Layers'] = list(LayersY.values.copy())
        pcaPlotLabel(
            pca, pca_df, pca_df['Layers'], 'All Layers by Intervals',
            ['Layers'] + labelsY, nColorLay=nColorLay, nShadesLay=nShadesLay,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        dt = float(dataLabel['dt'].unique())
        unitTI = 'min.'
        prec = 0
        LayersTimeT, LTIntervals, labelsT = Layers(
            dataLabel['TimeIndex'], TI, range(TI), unitTI, prec, mul=dt
        )
        pca_df['TimeLayers'] = list(LayersTimeT.values.copy())
        pcaPlotLabel(
            pca, pca_df, pca_df['TimeLayers'], 'All Time by Intervals',
            ['TimeLayers'] + labelsT, xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        uTimeLayers = len(pca_df['TimeLayers'].unique())
        uLayers = len(pca_df['Layers'].unique())
        uTreatment = len(pca_df['Treatments'].unique())
        uGroup = len(pca_df['Groups'].unique())
        uCellLine = len(pca_df['CellLine'].unique())

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        Par1 = 'Layers'
        Par2 = 'Experiment'
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=figsizeEXP,
            labelsPar1=[Par1] + labelsY, nrows=nrows, nColor=nColor, nShades=nShades,
            nColorLay=nColorLay, nShadesLay=nShadesLay,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=figsizeEXP[::-1],
            labelsPar1=[Par1] + labelsY, nrows=ncols, nColor=nColor, nShades=nShades,
            nColorLay=nColorLay, nShadesLay=nShadesLay,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        Par1 = 'Experiment'
        Par2 = 'Layers'
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=(15, 15),
            labelsPar2=labelsY, nColor=nColor, nShades=nShades,
            nColorLay=nColorLay, nShadesLay=nShadesLay,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        Par1 = 'Experiment'
        Par2 = 'Groups'
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=(15, 5),
            nColor=nColor, nShades=nShades,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        Par1 = 'Groups'
        Par2 = 'Experiment'
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=figsizeEXP,
            nrows=nrows, nColor=nColor, nShades=nShades,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=figsizeEXP[::-1],
            nrows=ncols, nColor=nColor, nShades=nShades,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        Par1 = 'Groups'
        Par2 = 'TimeLayers'
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=(15, 5), labelsPar2=labelsT,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        Par1 = 'TimeLayers'
        Par2 = 'Groups'
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=(15, 5), labelsPar1=[Par1] + labelsT,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )

        if multipleCL:
            st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
            FigureNumber += 1
            Par1 = 'CellLine'
            Par2 = 'Groups'
            PCA_colorPar1_titlePar2(
                pca, pca_df, Par1, Par2, figsize=(15, 5),
                xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
            )

            st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
            FigureNumber += 1
            Par1 = 'Groups'
            Par2 = 'CellLine'
            PCA_colorPar1_titlePar2(
                pca, pca_df, Par1, Par2, figsize=figsizeCL,
                xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
            )

        st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
        FigureNumber += 1
        Par1 = 'Treatments'
        Par2 = 'Groups'
        PCA_colorPar1_titlePar2(
            pca, pca_df, Par1, Par2, figsize=(15, 5),
            nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,
            xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
        )
        if not singleTREAT:
            st.latex(f"\\color{{blue}}{{\\Large Figure\\ {FigureNumber}}}")
            FigureNumber += 1
            Par1 = 'Groups'
            Par2 = 'Treatments'
            PCA_colorPar1_titlePar2(
                pca, pca_df, Par1, Par2, figsize=figsizeTREATS,
                nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,
                xlim_kmeans=xlim_kmeans, ylim_kmeans=ylim_kmeans
            )

        dataAE['Treatments'] = pca_df['Treatments'].copy()
        dataAE['Layers'] = pca_df['Layers'].copy()
        dataAE['TimeLayers'] = pca_df['TimeLayers'].copy()

        # AE_df = AEdata(...)  # Uncomment and adapt if you want to run AE analysis and show results
        AE_df = []

        return pca_df, FigureNumber, kmeans_pca, labelsT, k_cluster, AE_df, pca


# ════════════════════════════════════════════════════════════════════════
#  Per-Dose Visualization Helpers
# ════════════════════════════════════════════════════════════════════════

import re as _re   # used by merge helpers below

DOSE_CATEGORY_COLS = ["Cha1_Category", "Cha2_Category", "Cha3_Category"]


def merge_dose_csv_into_df(df, dose_csv_path):
    """
    Load a dose CSV (Experiment, Parent, Cha*_Category, ...) and left-merge
    its category columns into *df* (matched on Experiment + Parent).

    Matching strategy:
      1. Normalise both sides (strip NNN0, whitespace, .xls extension).
      2. Try exact Experiment match first.
      3. If no overlap, try substring matching — e.g. the dose CSV may have
         short names like ``F2`` that appear inside full well names such as
         ``AM001100425CHR2F02293TNNIRNOCOWH00``.

    Returns True if at least one Cha*_Category column was successfully added.
    """
    try:
        dose = pd.read_csv(dose_csv_path)
    except Exception as e:
        st.warning(f"Could not read dose CSV: {e}")
        return False

    if "Experiment" not in dose.columns or "Parent" not in dose.columns:
        st.warning("Dose CSV must contain 'Experiment' and 'Parent' columns.")
        return False

    # --- Normalise experiment names on BOTH sides ---
    def _norm_exp(s):
        return (s.astype(str)
                .str.replace("NNN0", "", regex=False)
                .str.replace(".xls", "", regex=False)
                .str.replace(".xlsx", "", regex=False)
                .str.strip())

    dose["Experiment"] = _norm_exp(dose["Experiment"])

    # Normalise df experiments for matching, but keep the originals so the
    # caller's dataframe is not permanently altered (prevents mangled names
    # in downstream pickle saves, groupby operations, etc.).
    _orig_df_experiment = df["Experiment"].copy()
    df["Experiment"] = _norm_exp(df["Experiment"])

    cat_cols = [c for c in DOSE_CATEGORY_COLS if c in dose.columns]
    if not cat_cols:
        st.warning("Dose CSV contains no Cha*_Category columns.")
        return False

    # Keep only the columns we need
    dose_slim = dose[["Experiment", "Parent"] + cat_cols].copy()

    # Aggregate per (Experiment, Parent): majority-vote for each category
    from collections import Counter
    def _majority(series):
        s = series.dropna()
        if len(s) == 0:
            return None
        return Counter(s).most_common(1)[0][0]

    dose_agg = (
        dose_slim
        .groupby(["Experiment", "Parent"])
        .agg({c: _majority for c in cat_cols})
        .reset_index()
    )

    # Extract WellLetter from the SHORT dose experiment names BEFORE remapping.
    # Dose names are plate positions like "B2", "D4"; the letter prefix is the row.
    def _get_well_letter(short):
        m = _re.match(r'^([A-Za-z]+)', str(short).strip())
        return m.group(1).upper() if m else "?"
    dose_agg["WellLetter"] = dose_agg["Experiment"].apply(_get_well_letter)

    # --- Check overlap & build experiment mapping ---
    df_exps = set(df["Experiment"].unique())
    dose_exps = set(dose_agg["Experiment"].unique())
    overlap = df_exps & dose_exps

    with st.expander("Dose merge — debug info", expanded=False):
        st.write(f"**data Experiments (sample):** {list(df_exps)[:5]}")
        st.write(f"**dose Experiments (sample):** {list(dose_exps)[:5]}")
        st.write(f"**exact overlapping Experiments:** {len(overlap)}")

    # If no exact overlap, try plate-position matching.
    # Dose short names are plate positions like "B2", "D4".
    # Full experiment names encode the well as zero-padded form, e.g. "B02", "D04".
    # We convert "B2" → "B02" (pad the numeric part) and search inside the full name.
    if len(overlap) == 0 and len(dose_exps) > 0 and len(df_exps) > 0:
        st.info("No exact Experiment match. Trying well-position matching (e.g. B2 → B02)…")

        def _to_padded(short):
            """Convert 'B2' → 'B02', 'D12' → 'D12' (already 2-digit)."""
            m = _re.match(r'^([A-Za-z]+)(\d+)$', str(short).strip())
            if m:
                return m.group(1).upper() + m.group(2).zfill(2)
            return str(short).strip()

        dose_to_full = {}
        for d_exp in dose_exps:
            padded = _to_padded(d_exp)
            for f_exp in df_exps:
                if padded in f_exp:
                    dose_to_full[d_exp] = f_exp
                    break

        if dose_to_full:
            with st.expander("Dose merge — well-position matches", expanded=False):
                st.write(f"**matches found:** {len(dose_to_full)}")
                st.write(f"**mapping sample:** { {k: v for k, v in list(dose_to_full.items())[:5]} }")
            dose_agg["Experiment"] = dose_agg["Experiment"].map(dose_to_full)
            dose_agg.dropna(subset=["Experiment"], inplace=True)
            overlap = df_exps & set(dose_agg["Experiment"].unique())
        else:
            # Final fallback: plain substring match
            for d_exp in dose_exps:
                for f_exp in df_exps:
                    if d_exp in f_exp or f_exp in d_exp:
                        dose_to_full[d_exp] = f_exp
                        break
            if dose_to_full:
                dose_agg["Experiment"] = dose_agg["Experiment"].map(dose_to_full)
                dose_agg.dropna(subset=["Experiment"], inplace=True)
                overlap = df_exps & set(dose_agg["Experiment"].unique())
            else:
                st.warning("Could not match any Experiment names between data and dose CSV.")

    if len(overlap) == 0:
        st.warning("No overlapping Experiment names between data and dose CSV.")
        return False

    # Debug Parent overlap for first matching experiment
    ex0 = list(overlap)[0]
    df_parents = sorted(df.loc[df['Experiment'] == ex0, 'Parent'].unique()[:5])
    dose_parents = sorted(dose_agg.loc[dose_agg['Experiment'] == ex0, 'Parent'].unique()[:5])
    with st.expander("Dose merge — Parent debug", expanded=False):
        st.write(f"**data Parents ('{ex0}'):** {df_parents}  dtype={df['Parent'].dtype}")
        st.write(f"**dose Parents ('{ex0}'):** {dose_parents}  dtype={dose_agg['Parent'].dtype}")

    # Ensure Parent types match
    try:
        dose_agg["Parent"] = dose_agg["Parent"].astype(df["Parent"].dtype)
    except (ValueError, TypeError):
        df["Parent"] = df["Parent"].astype(str)
        dose_agg["Parent"] = dose_agg["Parent"].astype(str)

    # Drop any pre-existing category columns to avoid _x/_y duplicates
    drop_cols = [c for c in cat_cols + ["WellLetter"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    merge_cols = ["Experiment", "Parent"] + cat_cols + ["WellLetter"]
    merged = df.merge(dose_agg[merge_cols], on=["Experiment", "Parent"], how="left")

    # Write the new columns back into the original df (in place)
    for c in cat_cols + ["WellLetter"]:
        df[c] = merged[c].values

    # Restore the original (un-normalised) Experiment names so the caller's
    # dataframe is not permanently mangled.
    df["Experiment"] = _orig_df_experiment

    n_matched = df[cat_cols[0]].notna().sum()
    st.success(f"✅ Merged dose data for **{n_matched}** / {len(df)} rows "
               f"({len(cat_cols)} category columns: {', '.join(cat_cols)}).")
    return n_matched > 0


def mask_control_channel(df, control_channel):
    """
    For each Cha*_Category column, determine whether its channel position
    matches *control_channel* by inspecting the Experiment name, then set
    that column to 'NA'.

    Channel positions in the experiment name:
      positions 22-25 → Cha1, 26-29 → Cha2, 30-33 → Cha3.

    If the experiment names have already been NNN0-cleaned, positions shift.
    As a fallback, we just check whether *control_channel* appears in the name
    at any channel slot and mask the corresponding column.
    """
    cat_cols = detect_dose_columns(df)
    if not cat_cols:
        return

    # Try to figure out *which* Cha column maps to the control channel.
    # Use the first Experiment name as a representative.
    sample_exp = str(df["Experiment"].iloc[0])

    # After NNN0 removal the channel block starts at position 22 or later.
    # We search for the 4-letter control_channel string
    # at each 4-char slot starting from position 18 (conservative).
    masked_col = None
    for ch_idx in range(3):
        col_name = f"Cha{ch_idx + 1}_Category"
        if col_name not in cat_cols:
            continue
        # Check the channel slot in the experiment name
        start = 22 + ch_idx * 4
        if start + 4 <= len(sample_exp):
            if sample_exp[start:start + 4].upper() == control_channel.upper():
                masked_col = col_name
                break

    # Fallback: search anywhere in the experiment string
    if masked_col is None:
        for ch_idx in range(3):
            col_name = f"Cha{ch_idx + 1}_Category"
            if col_name in cat_cols and control_channel.upper() in sample_exp.upper():
                masked_col = col_name
                break

    if masked_col:
        df[masked_col] = "NA"
        st.write(f"Masked `{masked_col}` → NA  (control channel **{control_channel}**)")
    else:
        st.warning(f"Could not determine which channel column corresponds to "
                   f"'{control_channel}'. No masking applied.")


def detect_dose_columns(df):
    """Return the list of Cha*_Category columns that exist in *df*."""
    return [c for c in DOSE_CATEGORY_COLS if c in df.columns]


def build_dose_combo_column(df, dose_cols):
    """
    Create a 'DoseCombo' column by joining the per-channel category values.
    Missing / NaN values are filled with 'NA'.
    Returns the modified DataFrame (in place).
    """
    tmp = df[dose_cols].fillna("NA").astype(str)
    df["DoseCombo"] = tmp.apply(lambda row: "_".join(row), axis=1)
    return df


def histByKmeansDoseCombo(pca_df, k_cluster=3, bar_width=0.25,
                          figsize=(15, 5), rotate=45):
    """
    Grouped bar chart: cluster proportion per DoseCombo (each combo sums to 100 %).
    """
    with st.spinner("Generating cluster distribution per Dose Combo..."):
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.set_title("Cluster Distribution per Dose Combo", fontsize=15)

        combos = sorted(pca_df["DoseCombo"].unique())
        n_combos = len(combos)
        combo_to_idx = {c: i for i, c in enumerate(combos)}
        color = sns.color_palette("hls", k_cluster)

        for g in range(k_cluster):
            heights = []
            for combo in combos:
                subset = pca_df[pca_df["DoseCombo"] == combo]
                total = len(subset)
                count = (subset["Groups"] == g).sum()
                heights.append(count / total if total else 0)
            x_pos = [combo_to_idx[c] + g * bar_width for c in combos]
            ax.bar(x_pos, heights, bar_width, label=f"Group {g}", color=color[g])

        tick_pos = [combo_to_idx[c] + (k_cluster - 1) * bar_width / 2 for c in combos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(combos, rotation=rotate, ha="right", fontsize=10)
        ax.set_xlabel("Dose Combo", fontsize=13)
        ax.set_ylabel("Proportion", fontsize=13)
        ax.legend(bbox_to_anchor=(1.15, 1), framealpha=1, edgecolor="black")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def dose_cluster_heatmap(pca_df, k_cluster=3, figsize=(10, 6)):
    """
    Heatmap of cluster-proportion rows vs DoseCombo columns.
    Each column (combo) sums to 1.
    """
    with st.spinner("Generating Dose Combo × Cluster heatmap..."):
        ct = pd.crosstab(pca_df["Groups"], pca_df["DoseCombo"])
        # normalise so each combo column sums to 1
        ct_norm = ct.div(ct.sum(axis=0), axis=1)
        ct_norm.index = [f"Group {i}" for i in ct_norm.index]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title("Cluster Proportion per Dose Combo", fontsize=14)
        ax.set_xlabel("Dose Combo")
        ax.set_ylabel("Cluster")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # also show raw counts
        ct.index = [f"Group {i}" for i in ct.index]
        st.markdown("**Raw cell counts  (Cluster × Dose Combo)**")
        st.dataframe(ct)


def dose_cluster_chi_square(pca_df, f_html):
    """
    Chi-squared test of independence between DoseCombo and Groups.
    Displays the contingency table and test results.
    """
    import researchpy as rp
    with st.spinner("Running chi-square test: Dose Combo vs Cluster ..."):
        try:
            table, results = rp.crosstab(
                pca_df["DoseCombo"], pca_df["Groups"],
                prop="col", test="chi-square"
            )
            st.markdown("**Dose Combo × Cluster — Chi-Square Test**")
            st.dataframe(table)
            st.dataframe(results)
            f_html.write(table.to_html(index=True))
            f_html.write(results.to_html(index=True))
        except Exception as e:
            st.warning(f"Could not run chi-square test for Dose Combo: {e}")


def dose_feature_kde(pca_df, Features, k_cluster=3):
    """
    For each DoseCombo, plot KDE of every feature coloured by cluster group.
    """
    combos = sorted(pca_df["DoseCombo"].unique())
    Groups = range(k_cluster)
    for combo in combos:
        subset = pca_df[pca_df["DoseCombo"] == combo]
        if len(subset) < 5:
            continue
        st.latex(r"\color{blue}{\Large Dose\ Combo:\ %s}" % combo.replace("_", r"\_"))
        histogramDataKDELabels(
            Groups, subset, Features, 0,
            Par="Groups", nColor=0, nShades=0
        )


def dose_pca_scatter(pca_df, k_cluster=3, figsize=(8, 6)):
    """
    Scatter plot in PCA space coloured by DoseCombo.
    """
    with st.spinner("Generating PCA scatter plot coloured by Dose Combo..."):
        combos = sorted(pca_df["DoseCombo"].unique())
        palette = sns.color_palette("hls", len(combos))
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        sns.scatterplot(x="PC1", y="PC2", hue="DoseCombo", data=pca_df,
                        palette=palette, ax=ax, alpha=0.6)
        ax.set_title("PCA coloured by Dose Combo", fontsize=14)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  framealpha=1, edgecolor="black", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def dose_kmeans_pca(dsn_sub, feature_cols, k_cluster=3, well_label="",
                    dose_combo_labels=None):
    """
    Run independent PCA + K-means clustering on a per-well-letter subset.

    Steps:
      1. Extract numeric feature columns from *dsn_sub*.
      2. Z-score normalise (StandardScaler).
      3. PCA (all components).
      4. Elbow plot for reference.
      5. K-means on significant PCA components (explained-variance ratio ≥ 0.1,
         minimum 2).
      6. Scatter plot of PC1 vs PC2 coloured by cluster.
      7. Scatter plot of PC1 vs PC2 coloured by DoseCombo (if provided).

    Parameters
    ----------
    dsn_sub : pd.DataFrame
        Subset of dataSpecGraphN for one well letter.  Must contain the
        columns listed in *feature_cols*.
    feature_cols : list[str]
        Column names to use as features (e.g. Features[:-1]).
    k_cluster : int
        Number of clusters for K-means.
    well_label : str
        Label used in plot titles (e.g. the well letter).
    dose_combo_labels : array-like or None
        DoseCombo label for every row of *dsn_sub*.  When provided an
        additional PCA scatter coloured by DoseCombo is plotted.

    Returns
    -------
    groups : np.ndarray
        Cluster assignment for every row of *dsn_sub* (sorted so that
        cluster 0 has the smallest centre-sum, matching the main TASC
        convention).
    pc1 : np.ndarray
        First principal-component scores.
    pc2 : np.ndarray
        Second principal-component scores.
    """
    with st.spinner(f"Running per-well PCA + K-means for row {well_label}..."):
        # --- 1. Extract numeric features ----------------------------------
        available_cols = [c for c in feature_cols if c in dsn_sub.columns]
        data = dsn_sub[available_cols].apply(pd.to_numeric, errors="coerce")
        data = data.fillna(0.0)

        if data.shape[0] < k_cluster:
            st.warning(
                f"Too few rows ({data.shape[0]}) for {k_cluster} clusters "
                f"in well row {well_label} — skipping clustering."
            )
            return None, None, None

        # --- 2. Z-score normalisation -------------------------------------
        scaler = StandardScaler(with_std=True)
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index,
        )

        # --- 3. PCA -------------------------------------------------------
        pca = decomposition.PCA(random_state=42)
        pca.fit(data_scaled)
        pca_transformed = pca.transform(data_scaled)

        n_sig = max(2, int(np.sum(pca.explained_variance_ratio_ >= 0.1)))
        pca_exp_var = pca.explained_variance_ratio_ * 100

        # --- 4. Elbow plot ------------------------------------------------
        pca_sig = pca_transformed[:, :n_sig]
        n_clusters_range = range(2, min(16, data.shape[0]))
        distortions = []
        for n_k in n_clusters_range:
            km_tmp = KMeans(n_clusters=n_k, random_state=0).fit(pca_sig)
            distortions.append(
                np.sum(np.min(
                    cdist(pca_sig, km_tmp.cluster_centers_, "euclidean"),
                    axis=1,
                )) / pca_sig.shape[0]
            )
        fig_elbow, ax_elbow = plt.subplots(figsize=(6.5, 6.5), dpi=200)
        ax_elbow.plot(list(n_clusters_range), distortions, "bx-")
        ax_elbow.set_xlabel("K Clusters", fontsize=15)
        ax_elbow.set_ylabel("Distortion", fontsize=15)
        ax_elbow.set_title(
            f"Elbow Method — Well Row {well_label}",
            fontweight="bold", fontsize=18,
        )
        st.pyplot(fig_elbow)
        plt.close(fig_elbow)

        # --- 5. K-means ---------------------------------------------------
        km_model = KMeans(n_clusters=k_cluster, random_state=0).fit(pca_sig)
        # Sort clusters so that group 0 = smallest centre sum (same as main TASC)
        idx = np.argsort(km_model.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(k_cluster)
        groups = lut[km_model.predict(pca_sig)]

        # --- 6. Scatter plot ----------------------------------------------
        PC_cols = [f"PC{i+1}" for i in range(n_sig)]
        km_df = pd.DataFrame(pca_sig, columns=PC_cols, index=data.index)
        km_df["Groups"] = groups
        n_groups = len(np.unique(groups))
        palette = sns.color_palette("hls", n_groups)

        fig_km, ax_km = plt.subplots(figsize=(6.5, 6.5), dpi=200, facecolor="w")
        sns.scatterplot(
            x="PC1", y="PC2", hue="Groups", data=km_df,
            ax=ax_km, legend="full", palette=palette,
        )
        ax_km.set_title(
            f"K-means k={k_cluster} — Well Row {well_label}",
            fontweight="bold", fontsize=18,
        )
        ax_km.set_xlabel(f"PC1 ({pca_exp_var[0]:.2f}%)", fontsize=15)
        ax_km.set_ylabel(f"PC2 ({pca_exp_var[1]:.2f}%)", fontsize=15)
        legend_texts = ax_km.get_legend().texts if ax_km.get_legend() else []
        ncol = 1
        if len(legend_texts) > 25:
            ncol = 3
        elif len(legend_texts) > 17:
            ncol = 2
        ax_km.legend(
            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0,
            ncol=ncol, framealpha=1, edgecolor="black",
        )
        # Plot cluster centres
        centres = km_model.cluster_centers_
        ax_km.scatter(centres[:, 0], centres[:, 1], c="black", s=25)
        for spine in ax_km.spines.values():
            spine.set_visible(True)
        st.pyplot(fig_km)
        plt.close(fig_km)

        # --- 7. Scatter coloured by DoseCombo -----------------------------
        if dose_combo_labels is not None:
            combo_arr = np.asarray(dose_combo_labels)
            if len(combo_arr) == len(km_df):
                km_df["DoseCombo"] = combo_arr
                combos = sorted(km_df["DoseCombo"].unique())
                pal_combo = sns.color_palette("hls", len(combos))

                fig_dc, ax_dc = plt.subplots(
                    figsize=(6.5, 6.5), dpi=200, facecolor="w",
                )
                sns.scatterplot(
                    x="PC1", y="PC2", hue="DoseCombo", data=km_df,
                    ax=ax_dc, legend="full", palette=pal_combo, alpha=0.6,
                )
                ax_dc.set_title(
                    f"K-means PCA by Dose Combo — Well Row {well_label}",
                    fontweight="bold", fontsize=16,
                )
                ax_dc.set_xlabel(f"PC1 ({pca_exp_var[0]:.2f}%)", fontsize=15)
                ax_dc.set_ylabel(f"PC2 ({pca_exp_var[1]:.2f}%)", fontsize=15)
                leg_texts = ax_dc.get_legend().texts if ax_dc.get_legend() else []
                ncol_dc = 1
                if len(leg_texts) > 25:
                    ncol_dc = 3
                elif len(leg_texts) > 17:
                    ncol_dc = 2
                ax_dc.legend(
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0,
                    ncol=ncol_dc, framealpha=1, edgecolor="black", fontsize=8,
                )
                for spine in ax_dc.spines.values():
                    spine.set_visible(True)
                st.pyplot(fig_dc)
                plt.close(fig_dc)

    return groups, pca_transformed[:, 0], pca_transformed[:, 1]
