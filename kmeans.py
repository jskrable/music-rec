#!/usr/bin/env python3
# coding: utf-8
"""
neural_net.py
04-03-19
jack skrable
"""

import numpy as np
from sklearn.cluster import KMeans

def kmeans(X, clusters):

    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    X = np.hstack((X, kmeans.labels_.reshape(-1,1)))
    return X