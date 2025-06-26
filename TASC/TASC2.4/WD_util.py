''' 
WD_util.py
Is a script that include all the generic function from the Wasserstein distance calculations
'''
from scipy.stats import wasserstein_distance
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
import numpy as np
from itertools import product


def create_comb_lists(celllines_,treatment_,experiment):
    '''
    create_comb_lists:
    creates combinations between a list of cell lines, and a list of treatments
    
    input: 
    celllines_ - a list of strings of the different cell lines
    treatment_ - a list of strings of the different treatments
    experiment - a list of all the experiments available in this use case
    
    output:
    experiment_ - a list of experiments that belong only to the combination the user needed by his selection
    
    '''
    comb_ = list(product(celllines_,treatment_))
    comb_ = [''.join(comb) for comb in comb_] 
    experiment_ = []
    for comb in comb_:
        experiment_ += [i for i in experiment if comb in i]

    experiment_ = list(set(experiment_))
    experiment_.sort()
    return experiment_

def wasserstein_comparing(data,data_sim=[],col='Experiment',col_sim=[],experiment_list=[],experiment_list_sim=[]):
    '''
    
    Wasserstein distance comparing between two experiments
    input:
        data - the raw data 
        data_sim - the data which we compare to
        col - column name for comparison of cases
        col_sim - column name for comparison of cases which we compare to
        experiment_list - the list of the combination by experiment name
        experiment_list_sim - the list of the combination by experiment name of cases which we compare to
    
    output:
        wasser_comparing_ - a dataframe with the comparison in each section independently
    
    '''
    if not experiment_list_sim:
        experiment_list_sim = experiment_list
        data_sim = data
        col_sim = col
    wasser_comparing_ = pd.DataFrame(columns=experiment_list_sim,index=experiment_list,dtype=float)
    for exp_c in experiment_list_sim:
        df = pd.DataFrame(columns=data_sim.columns,index=experiment_list_sim,dtype=float)
        for exp_r in experiment_list:
            if data_sim.loc[data_sim[col_sim]==exp_c].shape[0]>10 and \
               data.loc[data[col]==exp_r].shape[0]>10:
                ecdf_c = ECDF(data_sim['PC1'].loc[data_sim[col_sim]==exp_c])
                ecdf_r = ECDF(data['PC1'].loc[data[col]==exp_r])
                df['PC1'][exp_r] = wasserstein_distance(ecdf_c.x[1:],ecdf_r.x[1:],
                                                        ecdf_c.y[1:],ecdf_r.y[1:])
            else:
                df['PC1'][exp_r] = np.nan
        wasser_comparing_.loc[experiment_list,exp_c] = df['PC1']
    return wasser_comparing_
