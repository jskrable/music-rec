#!/usr/bin/env python3
# coding: utf-8
"""
kmeans.py
04-08-19
jack skrable
"""

import numpy as np
from sklearn.cluster import KMeans

def kmeans(X, clusters):

    # Perform kmeans classifier
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    # Get classes
    classes = kmeans.labels_.reshape(-1,1)
    # Normalize classes
    classes = classes / np.linalg.norm(classes)
    # Append to input matrix
    X = np.hstack((X, kmeans.labels_.reshape(-1,1)))
    return X



def find_optimal_k(X):

    for k in range (1, 21):
 
        # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
        kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
        
        # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
        labels = kmeans_model.labels_
     
        # Sum of distances of samples to their closest cluster center
        interia = kmeans_model.inertia_
        print("k:",k, "cost:", interia)