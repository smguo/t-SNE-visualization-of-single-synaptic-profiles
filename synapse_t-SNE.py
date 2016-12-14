# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 20:04:01 2016

@author: smguo_000
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import scipy.io
from time import time
from matplotlib.ticker import NullFormatter
from sklearn import manifold
#%%
def read_features(dir_path): # load synaptic features from MAT files
    subdir_path = glob.glob(os.path.join(dir_path, '*/'))    
    subdir_name = [os.path.split(subdir[:-1])[1] for subdir in subdir_path]
    seq_name =  subdir_name
    nSeq = len(seq_name)
    file_path = os.path.join(dir_path, seq_name[1],"mask.mat")
    mat = scipy.io.loadmat(file_path)
    IabInMarkerCell = mat['IabInMarkerCell']
    n_features = IabInMarkerCell[0,0].shape[1]  
    IabInMarker = np.array([]).reshape(0,n_features) # define empty array for collecting features from individual MAT files
    IabWithMarker = np.array([]).reshape(0,n_features)
    AabWithMarker = np.array([]).reshape(0,n_features)
    for j in range(0, nSeq): # going through each sub-folder
        file_path = os.path.join(dir_path, seq_name[j],"mask.mat")
        mat = scipy.io.loadmat(file_path)
        IabInMarkerCell = mat['IabInMarkerCell']                 
        IabInMarkerS = np.concatenate(IabInMarkerCell[:])
        IabInMarkerS = np.concatenate(IabInMarkerS[:])
    
        IabWithMarkerCell = mat['IabWithMarkerCell']                
        IabWithMarkerS = np.concatenate(IabWithMarkerCell[:])
        IabWithMarkerS = np.concatenate(IabWithMarkerS[:])  
    
        AabWithMarkerCell = mat['AabWithMarkerCell']                  
        AabWithMarkerS = np.concatenate(AabWithMarkerCell[:])
        AabWithMarkerS = np.concatenate(AabWithMarkerS[:])
        
        IabInMarker = np.vstack([IabInMarker, IabInMarkerS])
        IabWithMarker = np.vstack([IabWithMarker, IabWithMarkerS])
        AabWithMarker = np.vstack([AabWithMarker, AabWithMarkerS])
        
    imgRoundNamesNoWO = mat['imgRoundNamesNoWO']    
    imgRoundNamesNoWO = np.concatenate(imgRoundNamesNoWO[:])
    imgRoundNamesNoWO = np.concatenate(imgRoundNamesNoWO[:])              
    return IabInMarker, IabWithMarker, AabWithMarker, imgRoundNamesNoWO
    
def stand_feature(X):
    # standardize the feature matrix so all the features have std = 1 and min = 0, 0 is assigned to missing data  
    for j in range(0,X.shape[1]):
        Xcol = X[:,j]
        Xcol = Xcol/np.std(Xcol[Xcol>0])
        Xcol[Xcol>0] = Xcol[Xcol>0] - np.amin(Xcol[Xcol>0])
        X[:,j] = Xcol
    return X
    
def is_outlier(points, thresh=3.5):
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
#%%
dir_path = "E:/data/Neuron/cortical/Broad_HCS/14days/20161021-crotalk in cells-t9-Rep2/post_processing_multicolor/"         
#dir_path = "E:/data/Neuron/cortical/Broad_HCS/14days/20161027-Rep3 and Rep4/Rep3/post_processing_multicolor/"         
IabInMarker, IabWithMarker, AabWithMarker, imgRoundNamesNoWO = read_features(dir_path)
#%%
#X = IabInMarker[:, :]
#X = np.hstack((IabWithMarker[:,0:8], IabWithMarker[:,9:], AabWithMarker[:,0:8], AabWithMarker[:,9:]))
X = np.hstack((IabInMarker[:,0:8], IabInMarker[:,9:], AabWithMarker[:,0:8], AabWithMarker[:,9:])) # leave out Tuj-1
label = np.hstack((imgRoundNamesNoWO[0:8], imgRoundNamesNoWO[9:]))
X = stand_feature(X)
X = X[0::3,:] # subsample the data to reduce memory usage 
n_neighbors = 10
n_components = 2
#perplexity = [10, 40, 100, 400, 1000, 4000]
perplexity = [10, 40, 100]
color = X[:,0]
#%
fig = plt.figure(figsize=(20, 12))
plt.suptitle("t-SNE_perplexity_scan", fontsize=14)
#% scan perplexity values

for k in range(0, len(perplexity)): 
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, perplexity=perplexity[k], 
                         n_iter= 5000, init='pca', random_state=1)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    ax = fig.add_subplot(2, 3, k+1)
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=color, s=5,cmap=plt.cm.nipy_spectral, edgecolors='none')
    plt.title("p= %i (%.2g sec)" % (perplexity[k],t1 - t0))
    plt.colorbar(sc)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')    
    plt.show()
    plt.savefig('E:/Google Drive/Python/figures/t-SNE_perplexity_scan.png',dpi=300)
#%% scan colormaps
    Ind_outlier = is_outlier(Y, thresh=3.5)
    fig = plt.figure(figsize=(20, 12))
    plt.suptitle("t-SNE with p= %i color-coded by each feature"
                 % (perplexity[k]), fontsize=14)
    for j in range(0, X.shape[1]):        
        color = X[~Ind_outlier,j]
        ax = fig.add_subplot(4, 6, j+1)
        sc = plt.scatter(Y[~Ind_outlier, 0], Y[~Ind_outlier, 1], c=color, s=1,cmap=plt.cm.nipy_spectral,
                         vmin=0, vmax=7, edgecolors='none')
        plt.title("%s" % (label[j%len(label)])) 
        plt.colorbar(sc)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')    
        plt.show()
    plt.savefig('E:/Google Drive/Python/figures/t-SNE_color_coded_feature_Iabin+Aab_no_Tuj1_p= %i_rep1.png'% (perplexity[k]),dpi=300)
    plt.savefig('E:/Google Drive/Python/figures/t-SNE_color_coded_feature_Iabin+Aab_no_Tuj1_p= %i_rep1.eps'% (perplexity[k]),dpi=300)
