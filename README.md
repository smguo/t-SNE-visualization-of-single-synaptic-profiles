# t-SNE-visualization-of-single-synaptic-profiles
* Visualizing high-dimensional single synaptic profiles using t-SNE

This script loads the synaptic features from the MAT files, concatenates the 
features from the same well or from the same replicate together (depending on 
the value of "split_wells"), subsample the synapses if the sample size is too 
large, standardize the features, and gereate t-SNE maps for the features. The 
output t-SNE maps are color-coded based on the feature values. 
