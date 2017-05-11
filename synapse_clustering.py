# -*- coding: utf-8 -*-
"""
This script performs hierachical clustering using Seaborn "clustermap" function
on synaptic features extracted from PRISM data. 
The script loads the synaptic features from the MAT files, concatenates the features from 
the same well or from the same replicate together (depending on the value of "split_wells"), 
subsamples the synapses if the sample size is too large, standardizes the features, and 
performs hierachical clustering. The clustering result was represented using dendrograms 
and heatmaps.  
Created on Sat Nov 19 20:04:01 2016
@author: Syuan-Ming Guo
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import scipy.io
from time import time
import pandas as pd
import seaborn as sns
from sklearn import manifold
import matplotlib as mpl
import math
import scipy
#import pylab
import scipy.cluster.hierarchy as sch

mpl.rcParams['pdf.fonttype'] = 42 # change the default settings of matplotlib
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams.update({'figure.autolayout': True})
sns.set_context("poster")
#%%
def read_features(dir_path): # load synaptic features from MAT files
    subdir_path = glob.glob(os.path.join(dir_path, '*/'))    
    subdir_name = [os.path.split(subdir[:-1])[1] for subdir in subdir_path]    
    nDir = len(subdir_name)
    file_path = os.path.join(dir_path, subdir_name[1],"mask.mat")
    mat = scipy.io.loadmat(file_path)
    IabInMarkerCell = mat['IabInMarkerCell']    
    IabInMarker = np.zeros(3,dtype=object) # define empty array for collecting features from individual MAT files
    IabWithMarker = np.zeros(3,dtype=object)
    AabWithMarker = np.zeros(3,dtype=object)
    for j in range(0, nDir): # going through each sub-folder
        file_path = os.path.join(dir_path, subdir_name[j],"mask.mat")
        mat = scipy.io.loadmat(file_path)
        IabInMarkerCell = mat['IabInMarkerCell']                 
        IabInMarkerS = np.concatenate(IabInMarkerCell[:]) # convert nested arrays to arrays 
        IabInMarkerS = np.concatenate(IabInMarkerS[:])
    
        IabWithMarkerCell = mat['IabWithMarkerCell']                
        IabWithMarkerS = np.concatenate(IabWithMarkerCell[:])
        IabWithMarkerS = np.concatenate(IabWithMarkerS[:])  
    
        AabWithMarkerCell = mat['AabWithMarkerCell']                  
        AabWithMarkerS = np.concatenate(AabWithMarkerCell[:])
        AabWithMarkerS = np.concatenate(AabWithMarkerS[:])
        
        IabInMarker[j] = IabInMarkerS # each array in "IabInMarker" contains all the features from a single well (total 3 wells)
        IabWithMarker[j] = IabWithMarkerS
        AabWithMarker[j] = AabWithMarkerS
        
    imgRoundNamesNoWO = mat['imgRoundNamesNoWO']    
    imgRoundNamesNoWO = np.concatenate(imgRoundNamesNoWO[:])
    imgRoundNamesNoWO = np.concatenate(imgRoundNamesNoWO[:])              
    return IabInMarker, IabWithMarker, AabWithMarker, imgRoundNamesNoWO

def concat_data(IabInMarkerS, IabWithMarkerS, AabWithMarkerS):
    n_features = IabInMarkerS[0].shape[1]  
    nDir = IabInMarkerS.shape[0]
    IabInMarker = np.array([]).reshape(0,n_features) # define empty array for collecting features from individual MAT files
    IabWithMarker = np.array([]).reshape(0,n_features)
    AabWithMarker = np.array([]).reshape(0,n_features)
    for j in range(0, nDir): # going through each sub-folder
        IabInMarker = np.vstack([IabInMarker, IabInMarkerS[j]])
        IabWithMarker = np.vstack([IabWithMarker, IabWithMarkerS[j]])
        AabWithMarker = np.vstack([AabWithMarker, AabWithMarkerS[j]])    
    return IabInMarker, IabWithMarker, AabWithMarker
    
def stand_feature(X):
    # standardize the feature matrix so all the features have std = 1 and min = 0, 0 is assigned to missing data  
    for j in range(0,X.shape[1]):
        Xcol = X[:,j]
        Xcol_nonzero =  Xcol[Xcol>0] ;
        out_ind = left_outlier_1d(Xcol_nonzero, -2.5) ;
        Xcol_nonzero[out_ind] = 0 ;        
        Xcol[Xcol>0] = Xcol_nonzero ;        
        Xcol = Xcol/np.std(Xcol[Xcol>0])
        Xcol[Xcol>0] = Xcol[Xcol>0] - np.amin(Xcol[Xcol>0])
        X[:,j] = Xcol
    return X
    
def is_outlier(points, thresh=3.5): # find outlier indices at both sides of the distribution
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
    
def left_outlier_1d(points, thresh=-3.5): # find outlier indices at only the left side of the distribution
    # find the outliers that are on the left side of the input 1D distribution only
    if len(points.shape) > 1:
        raise ValueError('left outlier is not defined for multidimentional data')    
#    points = points[:,None]
    median = np.median(points, axis=0)
    diff = points - median
    diff_abs = abs(diff)
    med_abs_deviation = np.median(diff_abs)
    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score < thresh
    
def build_features(IabInMarker, IabWithMarker, AabWithMarker, log_transform=False): # subsample the synapses if needed, stadardize the features and return the feature matrix
    IabWithMarker[:,8] = IabInMarker[:,8]
    IabWithMarker[:,12] = IabInMarker[:,12]
#    X = np.hstack((IabWithMarker[:,0:8], IabWithMarker[:,9:12], AabWithMarker[:,0:8], AabWithMarker[:,9:12])) # leave out Tuj-1 and MAP2  
    X = np.hstack((IabWithMarker, AabWithMarker[:,0:8], AabWithMarker[:,9:12])) # leave out Tuj-1 and MAP2
    if log_transform:
        X = np.log(X+1)        
    X = stand_feature(X)    
    n_sub = 58000 #subsample size. No subsampling for clustering
    
    if X.shape[0]>n_sub:
        rand_arr = np.random.rand(X.shape[0])
        subsample_ind = rand_arr<(n_sub/X.shape[0])
        X = X[subsample_ind,:] # subsample the data to reduce memory usage  
#        subsample_f = math.ceil(X.shape[0]/n_sub) # (subsample fraction)^-1
#        X = X[0::subsample_f,:] # subsample the data to reduce memory usage     
    return X
    
def run_clustering(IabInMarker, IabWithMarker, AabWithMarker,labelIA,log_transform=False):        
    X = build_features(IabInMarker, IabWithMarker, AabWithMarker, log_transform=log_transform)         
    X_df = pd.DataFrame(X.T, index=labelIA)        
    g = sns.clustermap(X_df, method='ward', metric='euclidean', 
       figsize=(4, 12),xticklabels=False, cmap = 'bwr',vmin=-6, vmax=6)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)                                                   
    #            plot_clustergram(X, Y1, Y2, imgRoundNamesNoWO)
    return g
    
def plot_clustergram(X, Y1, Y2, imgRoundNamesNoWO): # Alternative to sns.clustermap. Currently unused 
    label = np.hstack((imgRoundNamesNoWO[0:8], imgRoundNamesNoWO[9:12]))
    labelI = ["I(" + s + ")" for s in label]
    labelA = ["A(" + s + ")" for s in label]
    labelIA = np.hstack((labelI, labelA))
    
    fig = pylab.figure(figsize=(8,8))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])    
    Z1 = sch.dendrogram(Y1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])    
    Z2 = sch.dendrogram(Y2,orientation='right')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    X = X[idx1,:]
    X = X[:,idx2]
    im = axmatrix.matshow(X, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    fig.show()
    fig.savefig('dendrogram.png')
     
#%%
fig_path = 'E:/Google Drive/Python/figures/'   # change 'fig_path' to the desired figure output path. 
for rep_num in range(1, 2): # specify replicate number    
    split_wells = False # if "True", run t-SNE for each well in the replicate, otherwise pool data from all the wells together
    
    if rep_num == 1:
        dir_path = "E:/data/Neuron/cortical/Broad_HCS/14days/20161021-crotalk in cells-t9-Rep2/post_processing_multicolor/"     # Change `dir_path' in the script to the folder where the MAT files are located    
    elif rep_num == 2:        
        dir_path = "E:/data/Neuron/cortical/Broad_HCS/14days/20161027-Rep3 and Rep4/Rep3/post_processing_multicolor/"         
    elif rep_num == 3:        
        dir_path = "E:/data/Neuron/cortical/Broad_HCS/14days/20161027-Rep3 and Rep4/Rep4/post_processing_multicolor/"       
    else:
        raise ValueError('replicate number is out of range')
        
    IabInMarkerAll, IabWithMarkerAll, AabWithMarkerAll, imgRoundNamesNoWO = read_features(dir_path)
    nWell = IabInMarkerAll.shape[0]
    #%%    
    label = imgRoundNamesNoWO # labels of targets
    labelI = ["I(" + s + ")" for s in label]
    label = np.hstack((imgRoundNamesNoWO[0:8], imgRoundNamesNoWO[9:12]))
    labelA = ["A(" + s + ")" for s in label]
    labelIA = np.hstack((labelI, labelA))
    if split_wells:    
        for j in range(0, nWell): # going through each sub-folder
            IabInMarker = IabInMarkerAll[j] 
            IabWithMarker = IabWithMarkerAll[j] 
            AabWithMarker = AabWithMarkerAll[j]
            g = run_clustering(IabInMarker, IabWithMarker, AabWithMarker,labelIA)
            title = 'rep%i-%i'% (rep_num, j+1)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_path+'cluster_Iabwith+Aab_'+title+'.png',dpi=300)
            plt.savefig(fig_path+'cluster_Iabwith+Aab_'+title+'.pdf',dpi=300)                            
    #        plt.savefig('E:/Google Drive/Python/figures/t-SNE_Iabwith+Aab_no_Tuj1_MAP2_p= %i_rep%i_well%i.eps'% (perplexity, rep_num, j),dpi=300)
            plt.close("all")
    else:        
        IabInMarker, IabWithMarker, AabWithMarker = concat_data(IabInMarkerAll, IabWithMarkerAll, AabWithMarkerAll)
        g = run_clustering(IabInMarker, IabWithMarker, AabWithMarker,labelIA,log_transform=False)
        title = 'rep%i_narrow'% (rep_num)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_path+'cluster_Iabwith+Aab_'+title+'.png',dpi=300)
        plt.savefig(fig_path+'cluster_Iabwith+Aab_'+title+'.pdf',dpi=300)       
#        plt.savefig('E:/Google Drive/Python/figures/t-SNE_Iabwith+Aab_no_Tuj1_MAP2_p= %i_rep%i_well%i.eps'% (perplexity, rep_num, j),dpi=300)        
        plt.close("all")
        

    
