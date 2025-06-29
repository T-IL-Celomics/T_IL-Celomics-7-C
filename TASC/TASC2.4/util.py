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
import keras.backend.tensorflow_backend as KTF
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from datetime import datetime

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.compat.v1.Session(config=config)
KTF.set_session(sess)


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
        dataSpec = dataSpec.append(dataOne, sort=False)
        dataSpecGraph = dataSpecGraph.append(dataOneGraph, sort=False)

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


def createHierarchicalCluster(data,title,xlabel,ylabel,cmap="RdBu_r", vmin=-2, vmax=2):
    linkaged_pca = linkage(data, 'ward')
    # xlabel='Parameters', ylabel='# Cell'
    s = sns.clustermap(data=data, row_linkage=linkaged_pca, cmap=cmap, vmin=vmin, vmax=vmax, figsize=(30, 15),
                   cbar_kws = dict(use_gridspec=False))
    plt.suptitle(title, fontweight='bold', fontsize=30)

    s.ax_heatmap.set_xlabel(xlabel, fontsize=25, fontweight='bold')
    s.ax_heatmap.set_ylabel(ylabel, fontsize=25, fontweight='bold')

    s.cax.set_yticklabels(s.cax.get_yticklabels(), fontsize=25);
    pos = s.ax_heatmap.get_position();
    cbar = s.cax
    cbar.set_position([0.02, pos.bounds[1], 0.02, pos.bounds[3]]);
    s.ax_heatmap.set_xticklabels(s.ax_heatmap.get_xticklabels(), rotation=40, horizontalalignment='right',
                                     fontsize=20)
    plt.show()
    return None


def pcaVarianceExplained(pca,NSC):
    f, ax = plt.subplots(figsize=(10,10),dpi=100) 
    features = range(pca.n_components_);
    plt.bar(features, pca.explained_variance_ratio_*100);
    plt.xlabel('PCA features')
    plt.ylabel('Variance explained %')
    plt.xticks(features);
    print('There are ' + '{0:d}'.format(NSC) + ' signficant components')
    plt.show()
    return None
	

def pcaPlot(pca, pca_df, hue, title,nColor=0, nShades=0, nColorTreat=0, nShadesTreat=0,
            nColorLay=0,nShadesLay=0,xlim_kmeans=[0,0],ylim_kmeans=[0,0]):
    sns.axes_style({'axes.spines.left': True, 'axes.spines.bottom': True, 
               'axes.spines.right': True,'axes.spines.top': True})
    f, ax = plt.subplots(figsize=(6.5, 6.5),dpi=100, facecolor='w', edgecolor='k')
    num_of_dep = len(hue.unique())
    sns.despine(f, left=True, bottom=True)
    if hue.name=='Experiment' and nColor!=0:
        palette = ChoosePalette(nColor,nShades)
    elif hue.name=='Treatments' and nColorTreat!=0:
        palette = ChoosePalette(nColorTreat,nShadesTreat)
    elif hue.name=='Layers' and nColorLay!=0:
        palette = ChoosePalette(nColorLay,nShadesLay)
    else:
        palette = sns.color_palette("hls", num_of_dep)  # Choose color

    pca_expln_var_r = pca.explained_variance_ratio_*100
    
    s = sns.scatterplot(x="PC1", y="PC2", hue=hue.name, data=pca_df, ax=ax, 
                        palette=palette);

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,framealpha=1,edgecolor='black')

    plt.suptitle(title, fontweight='bold', fontsize=15);
    if xlim_kmeans!=[0,0]:
        plt.xlim(xlim_kmeans)
        plt.ylim(ylim_kmeans)
    plt.xlabel('PC1 ' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%');
    plt.ylabel('PC2 ' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%');
    if len(ax.get_legend().texts)>25:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=3,framealpha=1,edgecolor='black')
    elif len(ax.get_legend().texts)>17:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2,framealpha=1,edgecolor='black')
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1,framealpha=1,edgecolor='black')
        
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.show()
    return None
    
	
def pcaCalcOneExpMod(data,exp,title,whiten,n_components,nColor=0, nShades=0):
    pca = decomposition.PCA(n_components=n_components,random_state=42,whiten=whiten)
    pca.fit(data)
    pca_transformed = pca.transform(data)
    number_of_significant_components = sum(pca.explained_variance_ratio_>0.1)
    pca_df = pd.DataFrame(pca_transformed[:,0:2],index=exp.index)

    pca_df.rename(columns={0:'PC1', 1:'PC2'}, inplace=True)
    
    pca_df['Experiment'] = [expNames.replace('NNN0','') for expNames in exp]
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    pcaPlot(pca, pca_df, pca_df['Experiment'], title, nColor, nShades)  
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    pcaVarianceExplained(pca,number_of_significant_components)
    
    return pca_df, pca, pca_transformed
	

def pcaCalcOneExp(data,exp,title='',FigureNumber=None,nColor=0,nShades=0,show=True):
    pca = decomposition.PCA(random_state=42)
    pca.fit(data)
    pca_transformed = pca.transform(data)
    number_of_significant_components = sum(pca.explained_variance_ratio_>0.1)
    pca_df = pd.DataFrame(pca_transformed[:,0:2],index=exp.index)

    pca_df.rename(columns={0:'PC1', 1:'PC2'}, inplace=True)
    
    pca_df['Experiment'] = [expNames.replace('NNN0','') for expNames in exp]
    if show==True:
        display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))

        FigureNumber+=1    
        pcaPlot(pca, pca_df, pca_df['Experiment'], title, nColor, nShades)  
        display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
        FigureNumber+=1
        pcaVarianceExplained(pca,number_of_significant_components)
    
    return pca_df, pca, pca_transformed


def pcaCalc(data,expNamesInOrder,title,nColor=1,nShades=2):
    pca = decomposition.PCA(random_state=42)
    pca.fit(data)
    pca_transformed = pca.transform(data)
    number_of_significant_components = sum(pca.explained_variance_ratio_>0.1)
    pca_df = pd.DataFrame(pca_transformed[:,0:2],index=expNamesInOrder.index.values)

    pca_df.rename(columns={0:'PC1', 1:'PC2'}, inplace=True)

    pca_df['Experiment'] = [expNames.replace('NNN0','') for expNames in expNamesInOrder]
    
    pcaPlot(pca, pca_df, pca_df['Experiment'], title,nColor,nShades)
    return None
	

def ElbowGraph(pca,pca_transformed):
    f, ax = plt.subplots(figsize=(6.5, 6.5),dpi=100)
    number_of_significant_components = sum(pca.explained_variance_ratio_>0.1)
    # kmeans = KMeans(random_state=0).fit(pca_transformed)
    pca_transformed_n = pca_transformed[:,0:number_of_significant_components]
    n_clusters_i = range(2,16)
    # kmeans_pca = pca_df.copy().drop(pca_df.columns[2:], axis=1)
    PC_col = ['PC'+str(x) for x in range(1,number_of_significant_components+1)]
    kmeans_pca = pd.DataFrame(pca_transformed_n, columns=PC_col)

    distortions = []
    for i in n_clusters_i: 
        kmeanModel = KMeans(n_clusters=i, n_jobs=-1, random_state=0, verbose=0).fit(pca_transformed_n)
        kmeans_pca['Groups'] = kmeanModel.predict(pca_transformed_n)
        distortions.append(sum(np.min(cdist(pca_transformed_n, kmeanModel.cluster_centers_, 'euclidean'),
                                      axis=1)) / pca_transformed_n.shape[0])

    # Plot the elbow
    plt.plot(n_clusters_i, distortions, 'bx-')
    plt.xlabel('K Clusters',fontdict={'fontsize':15})
    plt.ylabel('Distortion',fontdict={'fontsize':15})
    plt.title('Elbow Method to find the Optimal K', fontdict={'fontweight':'bold', 'fontsize':25})

    plt.show()
    return None


def kmeansPlot(k_cluster,pca_transformed,pca,dataLabel):
    number_of_significant_components = sum(pca.explained_variance_ratio_>=0.1)
    if number_of_significant_components<2:
        number_of_significant_components = 2
	
    pca_transformed_n = pca_transformed[:,0:number_of_significant_components]
    f, ax = plt.subplots(figsize=(6.5, 6.5),dpi=100, facecolor='w', edgecolor='k')
    pca_expln_var_r = pca.explained_variance_ratio_*100
    PC_col = ['PC'+str(x) for x in range(1,number_of_significant_components+1)]
    kmeans_pca = pd.DataFrame(pca_transformed_n, columns=PC_col, index=dataLabel.index)
    kmeanModel = KMeans(n_clusters=k_cluster, n_jobs=-1, random_state=0).fit(pca_transformed_n)
    idx = np.argsort(kmeanModel.cluster_centers_.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(k_cluster)
    kmeans_pca['Groups'] = lut[kmeanModel.predict(pca_transformed_n)]
    num_of_dep = len(kmeans_pca['Groups'].unique())
    sns.despine(f, left=True, bottom=True)
    palette = sns.color_palette("hls", num_of_dep)  # Choose color  
    s = sns.scatterplot(x="PC1", y="PC2", hue='Groups', data=kmeans_pca, ax=ax,
                        legend='full', palette=palette);
    plt.suptitle('K-means clustering k=' + '{0:.0f}'.format(k_cluster), fontdict={'fontweight':'bold', 'fontsize':25})
    plt.xlabel('PC1 (' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%)', fontdict={'fontsize':15});
    plt.ylabel('PC2 (' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%)', fontdict={'fontsize':15});

    ## splitting the legend list into few columns
    if len(ax.get_legend().texts)>25:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=3,framealpha=1,edgecolor='black')
    elif len(ax.get_legend().texts)>17:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2,framealpha=1,edgecolor='black')
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1,framealpha=1,edgecolor='black')
    xlim_kmeans_l,xlim_kmeans_r = plt.xlim()
    ylim_kmeans_l,ylim_kmeans_r = plt.ylim()
    xlim_kmeans = [xlim_kmeans_l,xlim_kmeans_r]
    ylim_kmeans = [ylim_kmeans_l,ylim_kmeans_r]
    centers = kmeanModel.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=25, );
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.show()
    return kmeans_pca,xlim_kmeans,ylim_kmeans


def histogramData(exp,data,Features,FigureNumber):
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    nrows, ncols = get_rows_cols(len(Features))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(40,30),dpi=100)
    colors = sns.color_palette("hls", len(exp))
    for par, ax in zip(Features,axes.flat):
        for label, color in zip(range(len(exp)), colors):
            vals = np.float64(data[par].loc[data['Experiment'] == exp[label]])
            ax.hist(vals,
                    label=exp[label], color=color, density=True, stacked=True,
                    )
            ax.set_xlabel(par,) 
    fig.set_tight_layout(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.set_tight_layout(False)

    fig.tight_layout(pad=1.05)
    fig.legend(handles, labels, loc='upper right',fontsize='xx-large',framealpha=1,edgecolor='black')
    plt.subplots_adjust(right=0.8,top=0.9)

    plt.show()
    return None


def histogramDataKDE(exp,data,Features,FigureNumber,nColor=0,nShades=0):
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    nrows, ncols = get_rows_cols(len(Features))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(30,30),dpi=100)
    fig2, ax2 = plt.subplots(figsize=(10,6))

    if nColor==0:
        colors = sns.color_palette("hls", len(exp))
    else:
        colors = ChoosePalette(nColor,nShades)
    for par, ax in zip(Features,axes.flat):
        for label, color in zip(range(len(exp)), colors):
            vals = np.float64(data[par].loc[data['Experiment']==exp[label]])
            try:
                sns.kdeplot(vals, ax=ax,
                        label=exp[label], color=color, #density=True, stacked=True,
                        )
            except: 
                sns.kdeplot(vals, ax=ax,
                        label=exp[label], color=color,bw=50
                            )
            ax.set_xlabel(par,fontdict={'fontsize':15}) 
    fig.set_tight_layout(True)
    fig.tight_layout(pad=1.03)
    labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}

    fig.set_tight_layout(False)
    for a in axes.flat:
        try:
            a.get_legend().remove()
        except:
            display('No axis a (ignore this message)')
    fig2.legend(labels_handles.values(),
               labels_handles.keys(),
               loc='center',fontsize='xx-large',
               framealpha=1,edgecolor='black'
              )
    # fig.subplots_adjust(top=0.9)

    plt.show()
    return None


def AEdata(data, labelsCol, LE, shape, nrows=0, nColor=0, nShades=0,
            nColorTreat=0, nShadesTreat=0, nColorLay=0, nShadesLay=0,
            labelsLay='',figsize=(15,10),  labelsTime='', AE_model=False, model_name=''):
    # fix random seed for reproducibility
    np.random.seed(42)

    dataNumeric = data.drop(columns=['Experiment','Treatments','Layers','TimeLayers']).copy()
    Features = dataNumeric.columns
    for par in Features:
        dataNumeric[par] = dataNumeric[par]/(dataNumeric[par].abs().max())

    X = dataNumeric.values.copy()  
    
    if AE_model:
        input_layer = Input(shape=(shape,))
        
        encoding_layer1 = Dense(16,activation='elu')(input_layer)
        encoding_layer2 = Dense(4,activation='elu')(encoding_layer1)

        encoding_layer = Dense(2,activation='elu')(encoding_layer2)
        
        decoding_layer1 = Dense(4,activation='elu')(encoding_layer)
        decoding_layer2 = Dense(16,activation='elu')(decoding_layer1)
        
        decoding_layer = Dense(shape,activation='elu')(decoding_layer2)
        autoencoder = Model(input_layer, decoding_layer)
    #     adam = Adam(lr=0.001,)
        # compile model
        autoencoder.compile(optimizer='adadelta', loss='mse',)

        # fit the model
        autoencoder.fit(x = X, y = X, epochs=10) 

        encoder = Model(input_layer, encoding_layer)

        # encodings = encoder.predict(X)

        now = datetime.now()
        date_time = now.strftime("d%d%m%yh%H%M%S")
        # serialize model to JSON
        model_json_autoencoder = autoencoder.to_json()
        model_json_encoder = encoder.to_json()
        with open("model/model ae "+date_time+".json", "w") as file_json:
            file_json.write(model_json_autoencoder)
        with open("model/model e "+date_time+".json", "w") as file_json:
            file_json.write(model_json_encoder)
        # serialize weights to HDF5
        autoencoder.save_weights("model/model ae "+date_time+".h5")
        encoder.save_weights("model/model e "+date_time+".h5")
        print("Saved model to disk")
    else:
        # load json and create model
        ae_json_file = open("model/model ae "+model_name+'.json', 'r')
        loaded_model_ae_json = ae_json_file.read()
        ae_json_file.close()
        # encoder loading
        e_json_file = open("model/model e "+model_name+'.json', 'r')
        loaded_model_e_json = e_json_file.read()
        e_json_file.close()
        ae_loaded_model = model_from_json(loaded_model_ae_json)
        e_loaded_model = model_from_json(loaded_model_e_json)

        # load weights into new model
        ae_loaded_model.load_weights("model/model ae "+model_name+'.h5')
        e_loaded_model.load_weights("model/model e "+model_name+'.h5')
        autoencoder = ae_loaded_model
        encoder = e_loaded_model
        print("Loaded model from disk")
    encodings = encoder.predict(X)
    AE_df = pd.DataFrame(columns=['Encoder 0','Encoder 1'])
    for par,le in zip(labelsCol,LE):
        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
        if par=='Experiment' and nColor!=0:
            palette = ChoosePalette(nColor,nShades)
        elif par=='Treatments' and nColorTreat!=0:
            palette = ChoosePalette(nColorTreat,nShadesTreat)
        elif par=='Layers' and nColorLay!=0:
            palette = ChoosePalette(nColorLay,nShadesLay)
        else:
            num_of_dep = len(data[par].unique())
            palette = sns.color_palette("hls", num_of_dep)  # Choose color  
        
        if le:
            lb_make = LabelEncoder()
            labels = lb_make.fit_transform(data[par])
            uLabel = [u.replace('NNN0','') for u in data[par].unique()]
        else:
            if par=='Layers':
                labels = data[par].values.copy()
                uLabel = labelsLay            
            if par=='TimeLayers':
                labels = data[par].values.copy()
                uLabel = labelsTime     
            else:
                labels = data[par].values.copy()
                uLabel = data[par].unique().tolist()


        # fig,ax = plt.subplots(figsize=(6,6))
        # palette = sns.color_palette("hls", len(data[labelsCol].unique()))  # Choose color  
        
        AE_df['Encoder 0'] = encodings[:, 0]
        AE_df['Encoder 1'] = encodings[:, 1]
        AE_df[par] = labels
        s = sns.scatterplot(x='Encoder 0', y='Encoder 1', hue=par, data=AE_df , 
                            palette=palette)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,framealpha=1,edgecolor='black')
        for t, l in zip(s.get_legend().texts, [par]+uLabel): t.set_text(l)
        plt.show()
    # print(AE_df.columns)
    AE_colorPar1_titlePar2(AE_df, Par1='', Par2='Treatments', figsize=figsize, 
                            nrows=nrows, nColor=nColor, nShades=nShades, 
                            nColorTreat=nColorTreat,nShadesTreat=nShadesTreat,
                            nColorLay=nColorLay, nShadesLay=nShadesLay)
    AE_colorPar1_titlePar2(AE_df, Par1='', Par2='TimeLayers', figsize=(15,5), 
                            nrows=1, nColor=nColor, nShades=nShades, 
                            nColorTreat=nColorTreat,nShadesTreat=nShadesTreat,
                            nColorLay=nColorLay, nShadesLay=nShadesLay, labelsPar2=labelsTime)
    AE_colorPar1_titlePar2(AE_df, Par1='', Par2='Layers', figsize=(15,15), 
                            nrows=0, nColor=nColor, nShades=nShades, 
                            nColorTreat=nColorTreat,nShadesTreat=nShadesTreat,
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
    gmm = GaussianMixture(n_components=3,random_state=42)

    gmm.fit(pca_transformed[:,0:2])

    #predictions from gmm
    labels = gmm.predict(pca_transformed[:,0:2])
    frame = pd.DataFrame(pca_transformed[:,0:2])
    frame['GMM'] = labels
    frame.columns = ['PC1','PC2','GMM']

    num_of_dep = len(frame['GMM'].unique())
    palette = sns.color_palette("hls", num_of_dep)  # Choose color  
    sns.scatterplot(x='PC1', y='PC2',hue='GMM', palette=palette)
    plt.title(title)
    plt.show()
    return None


def histByKmeansTreatsLabel(pca_df,Par='TreatmentsLabels',k_cluster=3,bar_width=0.2,figsize=(10,5),labels='',rotate=0):
    fig, ax = plt.subplots(figsize=(15,5),dpi=100)
    Treat = pca_df.groupby(Par)
    rangeLabel = [float(i) for i in list(Treat.describe().index.values)]
    rangeLabelX = [float(i)+bar_width for i in list(Treat.describe().index.values)]
    for j in list(Treat.describe().index.values):
        T = Treat.get_group(j)['Groups']
        xlabels = T.unique()
        xlabels.sort()
        N = len(xlabels)
        color = sns.color_palette('hls',k_cluster)
        xrange = range(N)
        SUM = T.value_counts().sort_index().sum()
        for i in range(k_cluster):
            Group = T.loc[T==i].value_counts().sort_index()
            if len(xlabels)==k_cluster:
                plt.bar(rangeLabel[j] + i*bar_width, Group/SUM, bar_width,
                        label='Group '+str(i),
                        color=color[i])
            else:
                for e in xlabels:
                    if e not in Group:
                        Group[e] = 0
                Group.sort_index()
                plt.bar(rangeLabel[j] + i*bar_width, Group/SUM, bar_width, 
                        label='Group '+str(i),
                        color=color[i])
    if labels=='':
        plt.xticks(xrange + bar_width, (xlabels), rotation=rotate, fontsize=12)
    else:
        labels.sort()
        plt.xticks(rangeLabelX , (labels), rotation=rotate, fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),bbox_to_anchor=(1.15,1),framealpha=1,edgecolor='black')
    plt.xlabel(Par,fontdict={'fontsize':15})
    plt.suptitle('each '+Par+'is 100%',fontdict={'fontsize':15}) 
    
    plt.show()
    return None
   
    
def PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize, labelsPar1='', labelsPar2='',
                            nrows=0, nColor=0, nShades=0, nColorTreat=0,nShadesTreat=0,
                            nColorLay=0, nShadesLay=0,xlim_kmeans=[-10,10],ylim_kmeans=[-10,10]):

    fig2,ax2 = plt.subplots(figsize=(10,6), facecolor='w', edgecolor='k')
    title = 'Each graph is a seperate ' + Par2 + ' and colored by ' + Par1 
    num_of_dep = len(pca_df[Par1].unique())
    if Par1=='Experiment' and nColor!=0:
        palette = ChoosePalette(nColor,nShades)
    elif Par1=='Treatments' and nColorTreat!=0:
        palette = ChoosePalette(nColorTreat,nShadesTreat)
    elif Par1=='Layers' and nColorLay!=0:
        palette = ChoosePalette(nColorLay,nShadesLay)
    else:
        palette = sns.color_palette("hls", num_of_dep)  # Choose color
    
    pca_expln_var_r = pca.explained_variance_ratio_*100
    uPar2 = pca_df[Par2].unique()
    uPar2.sort()
    uPar1 = list(pca_df[Par1].unique())
    uPar1.sort()
    if nrows==0:
        nrows = int(np.floor(np.sqrt(len(uPar2))))
        if nrows<=1:
            nrows = 1
            ncols = len(uPar2)
        elif nrows**2==len(uPar2):
            nrows = nrows
            ncols = nrows
        else:
            if len(uPar2)%nrows==0:
                # nrows += 1
                ncols = int(len(uPar2)/nrows)
            else:
                ncols = nrows + 1
                nrows += 1
    else:
        ncols = int(len(uPar2)/nrows)
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, 
                           ncols=ncols, constrained_layout=True,dpi=100)
    if type(ax) != np.ndarray:
        ax = np.array([ax])
    for i, a, t in zip(uPar2,ax.reshape(-1), range(len(uPar2))):
        df_PCA = pca_df.loc[pca_df[Par2]==i].copy()
        uPar1tmp = df_PCA[Par1].unique()
        uPar1tmp.sort()
        pal = sns.color_palette([palette[p] for p in [uPar1.index(value) for value in uPar1tmp]])
        sns.despine(fig, left=True, bottom=True)
        s = sns.scatterplot(x="PC1", y="PC2", hue=Par1, data=df_PCA, ax=a, 
                            palette=pal);
        if labelsPar2=='':
            a.set_title(i, fontweight='bold', fontsize=15);
        else:
            a.set_title(labelsPar2[t], fontweight='bold', fontsize=15);
        a.set_xlabel('PC1 ' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%');
        a.set_ylabel('PC2 ' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%');
        a.set_xlim(xlim_kmeans)
        a.set_ylim(ylim_kmeans)
        # a.set_xlim([pca_df['PC1'].min()+0.1*pca_df['PC1'].min(), pca_df['PC1'].max()+0.1*pca_df['PC1'].max()])
        # a.set_ylim([pca_df['PC2'].min()+0.1*pca_df['PC2'].min(), pca_df['PC2'].max()+0.1*pca_df['PC2'].max()])
        
    if labelsPar1!='':
        labels_handles = {
          lPar1: handle for ax in fig.axes for handle, label, lPar1 in zip(*ax.get_legend_handles_labels(),labelsPar1)
        }
    else:
        labels_handles = {
          label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
        }
    # fig.set_tight_layout(True)

    # fig.tight_layout(pad=1.02)

    # fig.legend(
                # labels_handles.values(),
                # labels_handles.keys(),
                # loc="upper center",
                # bbox_to_anchor=(1.2, 1),
                # bbox_transform=plt.gcf().transFigure,
                # )
    for a in ax.flat:
        try:
            a.get_legend().remove()
            for spine in a.spines.values():
                spine.set_visible(True)
        except:
            print('')
        
    # for item in [fig, ax]:
    # fig.patch.set_visible(True)
    fig2.legend(labels_handles.values(),
           labels_handles.keys(),
           loc='center',fontsize='xx-large',
           framealpha=1,edgecolor='black',
          )
    fig2.subplots_adjust(right=0.5,top=0.9)
    # fig.set_tight_layout(False)
    plt.suptitle(title, fontweight='bold', fontsize=25)

    plt.show()
    return None


def pcaPlotLabel(pca, pca_df, hue, title, labels, nColorLay=0, nShadesLay=0,xlim_kmeans=[-10,10],ylim_kmeans=[-10,10]):
    f, ax = plt.subplots(figsize=(6.5, 6.5),dpi=100, facecolor='w', edgecolor='k',)
    num_of_dep = len(hue.unique())
    sns.despine(f, left=True, bottom=True)
    if hue.name=='Layers' and nColorLay!=0:
        palette = ChoosePalette(nColorLay,nShadesLay)
    else:
        palette = sns.color_palette("hls", num_of_dep)  # Choose color  
    pca_expln_var_r = pca.explained_variance_ratio_*100

    s = sns.scatterplot(x="PC1", y="PC2", hue=hue.name, data=pca_df,
                        ax=ax, 
                        palette=palette,);
    handles,_ = ax.get_legend_handles_labels()

    plt.legend(handles,labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,framealpha=1,edgecolor='black')

    plt.suptitle(title, fontweight='bold', fontsize=15);
    plt.xlim(xlim_kmeans)
    plt.ylim(ylim_kmeans)
    plt.xlabel('PC1 ' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%');
    plt.ylabel('PC2 ' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%');
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.show()
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
    

def chi_square_test_tables(pca_df,labels,expectation='CON',Par='Treatments'):
    try:
        chi_table = pd.DataFrame()
        for la in labels:
            if la!=expectation:
                chi_df = pd.DataFrame(pca_df[Par].loc[pca_df[Par]==expectation].values.tolist()+
                                   pca_df[Par].loc[pca_df[Par]==la].values.tolist(), 
                                   columns=[Par])
                chi_df['Groups'] = pca_df['Groups'].loc[pca_df[Par]==expectation].values.tolist()+\
                                pca_df['Groups'].loc[pca_df[Par]==la].values.tolist()
                table, results = rp.crosstab(chi_df[Par],
                                             chi_df['Groups'],
                                             prop='row',
                                             test='chi-square')

                pvalue = results.at[1,'results']

                for i in table.columns:
                    chi_table.at[la,i[1]] = table[i].loc[la]

                chi_table.at[la,'N'] = pca_df[Par].loc[pca_df[Par]==la].size

                for i in results['Chi-square test'].values:
                    chi_table.at[la,i] = results.at[results['Chi-square test'].\
                                                               loc[results['Chi-square test']==i].index.values.tolist()[0],
                                                               'results']
        for i in table.columns:
            chi_table.at[expectation,i[1]] = table[i].loc[expectation]
        chi_table.at[expectation,'N'] = pca_df[Par].loc[pca_df[Par]==expectation].size
        chi_table.at['All','N'] = chi_table['N'].sum()
        display(chi_table)
        return None    
    except:
      print('ERROR')

def fix_old_no_pca_var(dataDrop, dataLabel, title='',FigureNumber=None, nColor=0, nShades=0,show=False):
    standardScaler = StandardScaler(with_std=True,)
    dataDrop = pd.DataFrame(standardScaler.fit_transform(dataDrop),columns=dataDrop.columns)
    _, pca, _ = pcaCalcOneExp(dataDrop, dataLabel['Experiment'], 'PCA of '+\
                                               title,FigureNumber, nColor=nColor, nShades=nShades,show=False)
    return pca
    

def histogramDataKDELabels(Labels,data,Features,FigureNumber,Par='Experiment',nColor=0,nShades=0):
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    nrows, ncols = get_rows_cols(len(Features))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(30,30),dpi=100)
    fig2,ax2 = plt.subplots(figsize=(6,6))
    if nColor==0:
        colors = sns.color_palette("hls", len(Labels))
    else:
        colors = ChoosePalette(nColor,nShades)
    # 
    for par, ax in zip(Features,axes.flat):
        for label, color in zip(range(len(Labels)), colors):
            try:
                vals = np.float64(data[par].loc[data[Par]==Labels[label]])
                sns.kdeplot(vals, ax=ax,
                            label=Labels[label], color=color,
                            )
            except:
                vals = np.float64(data[par].loc[data[Par]==Labels[label]])
                sns.kdeplot(vals, ax=ax,
                            label=Labels[label], color=color, bw=50
                            )
            ax.set_xlabel(par,) 
    fig.set_tight_layout(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.set_tight_layout(False)
    for a in axes.flat:
        try:
            a.get_legend().remove()
        except:
            continue
    fig.tight_layout(pad=1.01)
    fig2.legend(handles, labels, loc='upper right',fontsize='xx-large',framealpha=1,edgecolor='black')
    # plt.subplots_adjust(right=0.8)
    plt.show()
    return None


def Layers(data, q, labeling, unit, prec=0, mul=1.):
    LayX = pd.qcut(x=data,q=q,labels=labeling)
    LayXIntervals = pd.qcut(x=data,q=q)
    labels = roundInterval(LayXIntervals, unit, prec, mul)
    return LayX, LayXIntervals, labels


def histByKmeansTreats(pca_df,Par='TimeIndex',k_cluster=3,bar_width=0.2,figsize=(10,5),labels='',rotate=0):
    fig, ax = plt.subplots(figsize=figsize,dpi=100)
    plt.title(Par+' Histogram by group',fontdict={'fontsize':20})
    xlabels = pca_df[Par].unique()
    xlabels.sort()
    N = len(xlabels)
    xrange = np.arange(N)
    color = sns.color_palette('hls',k_cluster)
    for i in range(k_cluster):
        Group = pca_df[Par].loc[pca_df['Groups']==i].value_counts().sort_index()
        if len(Group)==N:
            plt.bar(xlabels + i*bar_width, Group/Group.sum(), bar_width, label='Group '+str(i), color=color[i])
        else:
            for e in xlabels:
                if e not in Group:
                    Group[e] = 0
            Group.sort_index()
            plt.bar(xlabels + i*bar_width, Group/Group.sum(), bar_width, label='Group '+str(i), color=color[i])
        
    plt.legend(bbox_to_anchor=(1.2,1),framealpha=1,edgecolor='black')
    if labels=='':
        plt.xticks(xrange + bar_width, (xlabels),rotation=rotate, fontsize=12)
    else:
        labels.sort()
        plt.xticks(xlabels + bar_width, (labels),rotation=rotate, fontsize=12)
    plt.xlabel(Par + ' range',fontdict={'fontsize':15}) 
    plt.show()
    return None


def histByKmeans(pca_df,Par='TimeIndex',k_cluster=3,bar_width=0.2,figsize=(10,5),labels='',rotate=0):
    fig, ax = plt.subplots(figsize=figsize,dpi=100)
    plt.title(Par+' Histogram by group',fontdict={'fontsize':20})
    xlabels = pca_df[Par].unique()
    xlabels.sort()
    xrange = np.arange(0,len(xlabels),1)
    for i in range(k_cluster):
        Group = pca_df[Par].loc[pca_df['Groups']==i].value_counts().sort_index()
        plt.bar(Group.index.values + i*bar_width, Group/Group.sum(), bar_width, label='Group '+str(i))

    plt.legend(bbox_to_anchor=(1.2,1),framealpha=1,edgecolor='black')
    if labels=='':
        plt.xticks(xlabels + bar_width, (xlabels),rotation=rotate, fontsize=12)
    else:
        plt.xticks(xlabels + bar_width, (labels),rotation=rotate, fontsize=12)
    plt.xlabel(Par + ' range',fontdict={'fontsize':15}) 
    plt.show()
    return None

  
def analysisExp(dataAll, dataAllGraph, expList, expNamesInOrderU):

    #Test for valid input in expList
    try:
        exp = expNamesInOrderU[expList];
    except IndexError as e:
        bad_arg = str(e).split(" ")[1]
        e.args = tuple(["You entered %s in expList, but that was out of range. Please go back to \"Sample cells\" and make sure to replace/delete it and run again." % bad_arg]) + e.args[1:]
        raise

    dataSpec, dataSpecGraph = extractSpecificExp(dataAll,dataAllGraph,exp)

    dataSpec['Experiment'] = dataSpec['Experiment'].replace(to_replace='NNN0', value='', regex=True)
    dataSpecGraph['Experiment'] = dataSpecGraph['Experiment'].replace(to_replace='NNN0', value='', regex=True)
    Features = dataSpec.columns.copy()

    print('The experiment you choose is:')
    [print(e.replace('NNN0','')) for e in exp]
    print('The cell line is:')
    [print((e.replace('NNN0',''))[18:22]) for e in exp]
    bol_s = '\033[1m' 
    bol_e = '\033[0m'
    print(bol_s+'The treatments are:'+bol_e)
    # [print((exp.replace('NNN0',''))[i:i+4]) for i in range(22, len(exp.replace('NNN0','')), 4)];
    print("Number of features: " , len(Features)-1 ,'\n')
    print("\033[1mThe Features are:\033[0m")
    [print(str(i)+'.',Feature) for i,Feature in zip(range(1,len(Features)),Features)];

    exp = [e.replace('NNN0','') for e in exp]
    return exp, Features, dataSpec, dataSpecGraph


def analysisExpH(dataAll, dataAllGraph, expList, expNamesInOrderU):
    exp = expNamesInOrderU[expList];

    dataSpec, dataSpecGraph = extractSpecificExp(dataAll,dataAllGraph,exp)

    dataSpec['Experiment'] = dataSpec['Experiment'].replace(to_replace='NNN0', value='', regex=True)
    dataSpecGraph['Experiment'] = dataSpecGraph['Experiment'].replace(to_replace='NNN0', value='', regex=True)
    Features = dataSpec.drop(columns=['Parent','TimeIndex','x_Pos','y_Pos','ID','dt']).columns.copy()

    print('The experiment you choose is:')
    [print(e.replace('NNN0','')) for e in exp]
    print('The cell line is:')
    [print((e.replace('NNN0',''))[18:22]) for e in exp]
    bol_s = '\033[1m' 
    bol_e = '\033[0m'
    print(bol_s+'The treatments are:'+bol_e)
    # [print((exp.replace('NNN0',''))[i:i+4]) for i in range(22, len(exp.replace('NNN0','')), 4)];
    print("Number of features: " , len(Features)-1 ,'\n')
    print("\033[1mThe Features are:\033[0m")
    [print(str(i)+'.',Feature) for i,Feature in zip(range(1,len(Features)),Features)];

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
                            nrows=0, nColor=0, nShades=0, nColorTreat=0,nShadesTreat=0,
                            nColorLay=0, nShadesLay=0):
    title = 'Each graph is a seperate ' + Par2 + ' and colored by ' + Par1 
    if Par1!='':
        uPar1 = list(AE_df[Par1].unique())
        uPar1.sort()
        num_of_dep = len(AE_df[Par1].unique())
        if Par1=='Experiment' and nColor!=0:
            palette = ChoosePalette(nColor,nShades)
        elif Par1=='Treatments' and nColorTreat!=0:
            palette = ChoosePalette(nColorTreat,nShadesTreat)
        elif Par1=='Layers' and nColorLay!=0:
            palette = ChoosePalette(nColorLay,nShadesLay)
        else:
            palette = sns.color_palette("hls", num_of_dep)  # Choose color  
    else:
        uPar1 = AE_df[Par2].unique()
        uPar1.sort()
        pal = sns.color_palette("hls", 1)  # Choose color  
    uPar2 = AE_df[Par2].unique()
    uPar2.sort()

    if nrows==0:
        nrows = int(np.floor(np.sqrt(len(uPar2))))
        if nrows<=1:
            nrows = 1
            ncols = len(uPar2)
        elif nrows**2==len(uPar2):
            nrows = nrows
            ncols = nrows
        else:
            if len(uPar2)%nrows==0:
                # nrows += 1
                ncols = int(len(uPar2)/nrows)
            else:
                ncols = nrows + 1
                nrows += 1
    else:
        ncols = int(len(uPar2)/nrows)
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, 
                           ncols=ncols, constrained_layout=True,dpi=100)
    for i, a, t in zip(uPar2,ax.reshape(-1), range(len(uPar2))):
        df_AE = AE_df.loc[AE_df[Par2]==i].copy()
        if Par1!='':
            uPar1tmp = df_AE[Par1].unique()
            uPar1tmp.sort()
            pal = sns.color_palette([palette[p] for p in [uPar1.index(value) for value in uPar1tmp]])
            sns.despine(fig, left=True, bottom=True)
            s = sns.scatterplot(x="Encoder 0", y="Encoder 1", hue=Par1, data=df_AE, ax=a, 
                                palette=pal);
        else:
            sns.despine(fig, left=True, bottom=True)
            s = sns.scatterplot(x="Encoder 0", y="Encoder 1", data=df_AE, ax=a, 
                                );
        if labelsPar2=='':
            a.set_title(i, fontweight='bold', fontsize=15);
        else:
            a.set_title(labelsPar2[t], fontweight='bold', fontsize=15);
        a.set_xlabel('Encoder 0');
        a.set_ylabel('Encoder 1');
        a.set_xlim([AE_df['Encoder 0'].min()+0.1*AE_df['Encoder 0'].min(), AE_df['Encoder 0'].max()+0.1*AE_df['Encoder 0'].max()])
        a.set_ylim([AE_df['Encoder 1'].min()+0.1*AE_df['Encoder 1'].min(), AE_df['Encoder 1'].max()+0.1*AE_df['Encoder 1'].max()])

    if labelsPar1!='':
        labels_handles = {
          lPar1: handle for ax in fig.axes for handle, label, lPar1 in zip(*ax.get_legend_handles_labels(),labelsPar1)
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
    framealpha=1,edgecolor='black'
    )
    if Par1!='':
        for a in ax.flat:
            a.get_legend().remove()
    plt.suptitle(title, fontweight='bold', fontsize=25)
    plt.show()
    return None


def DescriptiveTable(dataSpecGraphGroups, title):
    gb_Groups = dataSpecGraphGroups.groupby(['Groups']).agg(['count', 'mean', 'std', ('sem',sem),
                                                                             ('ci95_hi',lambda x:
                                                                                      (np.mean(x) + 1.96*
                                                                                       np.std(x)/math.sqrt(np.size(x)))),
                                                                             ('ci95_lo',lambda x:
                                                                                      (np.mean(x) - 1.96*
                                                                                       np.std(x)/math.sqrt(np.size(x))))])

    gb_Groups.rename(columns = {"std": "std. Deviation", 'count':'N',
                                'sem':'std. Error', 'mean':'Mean',
                                'ci95_hi':'95 confidence Interval for Mean Upper Bound',
                                'ci95_lo':'95 confidence Interval for Mean Lower Bound'},inplace=True)

    gb_Groups.to_csv(title+' Descriptive Table - Groups.csv')
    display(gb_Groups)
    return None

def ANOVE_DESC_TABLE(dataSpecGraphGroups, Features, title, dep='Groups',groupList=[0,1,2]):
    display(Latex('$\color{blue}{\Large ANOVA\ Table\ feature\ per\ Group}$'))
    ANOVA_MI = pd.MultiIndex.from_product([['Between '+dep,'Within '+dep,'Total'], 
                                           ['Sum of Squares','df','Mean Sqaure','F','Sig.']])
    ANOVA_df = pd.DataFrame(columns=ANOVA_MI,index=Features[:-1])
    index = pd.MultiIndex.from_product([Features[:-1],[0]],
                                       names=['Feature','Sig.'])
    columns = pd.MultiIndex.from_product([[dep],groupList, 
                                           ['N','Mean','Standard Deviation','Standard Deviation Error','95% Upper Bound Mean','95% Lower Bound Mean']])
    ANOVA_Desc_df = pd.DataFrame(columns=columns,index=index)
    ANOVA_Desc_df = ANOVA_Desc_df.reset_index(level='Sig.')
    for par in Features[:-1]:
        model_name = ols(par+' ~ C('+dep+')', data=dataSpecGraphGroups).fit()
        ano = sm.stats.anova_lm(model_name,typ=1)
        ano = ano.append(pd.DataFrame({"df":[ano.df.sum()],"sum_sq":[ano.sum_sq.sum()]},index={"Total"}))
        ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
        ano.rename(columns = {"sum_sq": "Sum of Squares", 'mean_sq':'Mean Sqaure', 'PR(>F)':'Sig.'},inplace=True)
        ano.rename(index = {'C('+dep+')': "Between "+dep, 'Residual':'Within '+dep},inplace=True)
        ANOVA_df.at[par] = ano.values.ravel()
        ANOVA_Desc_df.loc[par,'Sig.'] = ANOVA_df.loc[par,'Between '+dep]['Sig.']
        gb_Groups = dataSpecGraphGroups.groupby([dep])[par].agg(['count', 'mean', 'std', ('sem',sem),
                                                                         ('ci95_hi',lambda x:
                                                                                  (np.mean(x) + 1.96*
                                                                                   np.std(x)/math.sqrt(np.size(x)))),
                                                                         ('ci95_lo',lambda x:
                                                                                  (np.mean(x) - 1.96*
                                                                                   np.std(x)/math.sqrt(np.size(x))))])

        ANOVA_Desc_df.loc[par,'Groups'] = gb_Groups.values.ravel()
    display(ANOVA_Desc_df)
    ANOVA_Desc_df.to_csv(title+' ANOVA + Descriptive Table - ' +dep +'.csv')
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

def ANOVA_TABLE(dataSpecGraphGroups, Features, title='', dep='Groups'):
    display(Latex('$\color{blue}{\Large ANOVA\ Table\ feature\ per\ Group}$'))
    ANOVA_MI = pd.MultiIndex.from_product([['Between Groups','Within Groups','Total'], 
                                           ['Sum of Squares','df','Mean Square','F','Sig.']])
    ANOVA_df = pd.DataFrame(columns=ANOVA_MI,index=Features[:-1])
    for par in Features[:-1]:
        model_name = ols(par+' ~ C('+dep+')', data=dataSpecGraphGroups).fit()
        ano = sm.stats.anova_lm(model_name,typ=1)
        ano = ano.append(pd.DataFrame({"df":[ano.df.sum()],"sum_sq":[ano.sum_sq.sum()]},index={"Total"}))
        ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
        ano.rename(columns = {"sum_sq": "Sum of Squares", 'mean_sq':'Mean Square', 'PR(>F)':'Sig.'},inplace=True)
        ano.rename(index = {'C('+dep+')': "Between "+dep, 'Residual':'Within '+dep},inplace=True)
        ANOVA_df.at[par] = ano.values.ravel()
    ANOVA_df.dropna(axis=1,inplace=True)
    ANOVA_df.to_csv(title+' ANOVA Table - ' +dep +'.csv')
    display(ANOVA_df)
    return 

def TASC(dataDrop, dataLabel, labelsCol='Experiment', LE=True,
        title='All Cell Lines', HC=False, treats=['HGF'], combTreats=[['HGF'],['PHA']],
        LY = 9, TI = 3, multipleCL=True, singleTREAT=False, FigureNumber=2, nrows=0, ncols=1, 
        nColor=0, nShades=0, k_cluster=3,
        nColorTreat=0, nShadesTreat=0, nColorLay=0, nShadesLay=0,
        figsizeEXP=(15,40), figsizeTREATS=(15, 25),figsizeCL=(15, 25), Features='',
        AE_model=True, model_name=''): 
    '''
    Total Analysis - Single Cell:
    Input:
        dataDrop      - data to do analysis on 
        columnsToDrop - which parameters aren't suppose to be as numeric variables
        labelsCol     - 
        LE            - LabelEncoder if the label column isn't a numbers
        title         - title for the main PCA
        HC            - Hirarchical clustering analysis: True/False
    Output:
        
    '''
    dataAE = dataDrop.copy()
    dataAE['Experiment'] = dataLabel['Experiment'].copy()
    # z-score data
    standardScaler = StandardScaler(with_std=True,)
    dataDrop = pd.DataFrame(standardScaler.fit_transform(dataDrop),columns=dataDrop.columns)
    if HC:
        display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
        FigureNumber+=1
        createHierarchicalCluster(dataDrop,title,'Parameters','# Cell',
                                  sns.diverging_palette(150, 10, n=100),vmin=-2,vmax=2)
    listCells = dataDrop.index.values.copy()
    pca_df,pca,pca_transformed = pcaCalcOneExp(dataDrop, 
                                               dataLabel['Experiment'], 'PCA of '+\
                                               title,FigureNumber, nColor=nColor, nShades=nShades,show=True)
    FigureNumber+=2
    ## Find the Best K for k-means
    # display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    # FigureNumber+=1
    # ElbowGraph(pca,pca_transformed)
    ## K-means
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    kmeans_pca,xlim_kmeans,ylim_kmeans = kmeansPlot(k_cluster,pca_transformed,pca,dataLabel['Experiment'])   
    ## Times (Time point)
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    pca_df['Time'] = dataLabel['TimeIndex'].copy()
    pcaPlot(pca, pca_df, pca_df['Time'], 'by Time points', nColor, nShades)


    pca_df['Experiment'] = dataLabel['Experiment']
    pca_df['Groups'] = kmeans_pca['Groups'].values.copy()


    ## Color by Cell Line
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    Par = 'CellLine'
    pca_df[Par] = [ex[18:18+4] for ex in pca_df['Experiment']]
    pcaPlot(pca, pca_df, pca_df[Par], 'by ' + Par, nColor, nShades,xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)

    '''figureN = 0
    for treat in treats:
        ## sign by treatment
        display(Latex('$\color{blue}{\Large Figure\ %i.%i}$'%(FigureNumber,figureN)))
        figureN+=1
        pca_df[treat] = [1 if treat in ex else 0 for ex in pca_df['Experiment']]
        pcaPlot(pca, pca_df, pca_df[treat], 'by ' + treat, nColor, nShades)
    FigureNumber+=1'''

    ## sign by groups
    combPossible = createPossibleCOMB(combTreats)
    COMB = []
    combPossible = [var if len(np.asarray(elem).shape)>1 else elem for elem in combPossible for var in elem]

    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1

    
    if combTreats!=None:
        for ex in pca_df['Experiment']:
            TF = False
            for comb in combPossible:
                combTreat = ''.join(comb)
                sumtreats = sum([st in ex for st in comb])
                if len(comb)>1 and TF==False:
                    if all(st in ex for st in comb):
                        COMB += [combTreat]
                        TF = True
                elif TF==False and all(st in ex for st in comb):
                    COMB += [combTreat]
                    TF = True
            if TF==False:
                COMB += ['CON']
        pca_df['Treatments'] = COMB
        pcaPlot(pca, pca_df, pca_df['Treatments'], 'by Treatments', nColorTreat, nShadesTreat,
                xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)

    
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    precLY = 2
    unitLY = '$\mu$m'
    LayersY, LYIntervals, labelsY = Layers(dataLabel['y_Pos'],
                                          LY, range(-4,-4+LY), unitLY, precLY)
    pca_df['Layers'] = list(LayersY.values.copy())
    pcaPlotLabel(pca, pca_df, pca_df['Layers'],
                 'All Layers by Intervals', ['Layers']+labelsY, nColorLay=nColorLay,
                 nShadesLay=nShadesLay,xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)

    
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    dt = float(dataLabel['dt'].unique())
    unitTI = 'min.'
    prec = 0
    LayersTimeT, LTIntervals, labelsT = Layers(dataLabel['TimeIndex'], TI,
                                              range(TI), unitTI, prec, mul=dt)
    pca_df['TimeLayers'] = list(LayersTimeT.values.copy())
    pcaPlotLabel(pca, pca_df, pca_df['TimeLayers'], 'All Time by Intervals', ['TimeLayers']+labelsT,
                 xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)

    uTimeLayers = len(pca_df['TimeLayers'].unique())
    uLayers = len(pca_df['Layers'].unique())
    uTreatment = len(pca_df['Treatments'].unique())
    uGroup = len(pca_df['Groups'].unique())
    uCellLine = len(pca_df['CellLine'].unique())
    
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    Par1 = 'Layers'
    Par2 = 'Experiment'
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=figsizeEXP,
                            labelsPar1=[Par1]+labelsY, nrows=nrows, nColor=nColor, nShades=nShades,
                            nColorLay=nColorLay, nShadesLay=nShadesLay,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=figsizeEXP[::-1],
                            labelsPar1=[Par1]+labelsY, nrows=ncols, nColor=nColor, nShades=nShades,
                            nColorLay=nColorLay, nShadesLay=nShadesLay,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)


    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    Par1 = 'Experiment'
    Par2 = 'Layers'
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=(15, 15),
                            labelsPar2=labelsY, nColor=nColor, nShades=nShades,
                            nColorLay=nColorLay, nShadesLay=nShadesLay,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)


    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    Par1 = 'Experiment'
    Par2 = 'Groups'
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=(15, 5),
                            nColor=nColor, nShades=nShades,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)


    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    Par1 = 'Groups'
    Par2 = 'Experiment'
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=figsizeEXP,
                            nrows=nrows, nColor=nColor, nShades=nShades,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=figsizeEXP[::-1],
                            nrows=ncols, nColor=nColor, nShades=nShades,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)
    
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    Par1 = 'Groups'
    Par2 = 'TimeLayers'
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=(15, 5), labelsPar2=labelsT,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)


    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    Par1 = 'TimeLayers'
    Par2 = 'Groups'
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=(15, 5), labelsPar1=[Par1]+labelsT,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)

    if multipleCL:
        display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
        FigureNumber+=1
        Par1 = 'CellLine'
        Par2 = 'Groups'
        PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=(15, 5),
                                xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)

        display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
        FigureNumber+=1
        Par1 = 'Groups'
        Par2 = 'CellLine'
        PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=figsizeCL,
                                xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)

    
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    Par1 = 'Treatments'
    Par2 = 'Groups'
    PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=(15, 5),
                            nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,
                            xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)
    if not singleTREAT:
        display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
        FigureNumber+=1
        Par1 = 'Groups'
        Par2 = 'Treatments'
        PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize=figsizeTREATS,
                                nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,
                                xlim_kmeans=xlim_kmeans,ylim_kmeans=ylim_kmeans)

    dataAE['Treatments'] = pca_df['Treatments'].copy()
    dataAE['Layers'] = pca_df['Layers'].copy()
    dataAE['TimeLayers'] = pca_df['TimeLayers'].copy()
    
    # display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    # FigureNumber+=1
    # AE_df = AEdata(dataAE,labelsCol,LE,len(Features[:-1]), nrows=nrows, nColor=nColor, nShades=nShades,
                     # nColorLay=nColorLay, nShadesLay=nShadesLay,
                     # nColorTreat=nColorTreat, nShadesTreat=nShadesTreat,labelsLay=labelsY, labelsTime=labelsT,
                     # figsize=figsizeTREATS, AE_model=True, model_name='')
    AE_df = []
    #in output, AE_df
    
    return pca_df, FigureNumber, kmeans_pca, labelsT, k_cluster, AE_df, pca 
