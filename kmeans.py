#!/usr/bin/env python3
# coding: utf-8
"""
kmeans_clustering.py
02-26-19
jack skrable
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.factorization.python.ops import clustering_ops


# Template from https://www.tensorflow.org/api_docs/python/tf
# /contrib/factorization/KMeansClustering
#############################################################

def kmeans_v0(DATA, k, epochs):
    
    def input_fn():
        return tf.train.limit_epochs(
            tf.convert_to_tensor(DATA, dtype=tf.float32), num_epochs=1)


    kmeans = tf.contrib.factorization.KMeansClustering(
        num_clusters=k, use_mini_batch=False)

    # Train
    iterations = epochs
    previous_centers = None
    for i in range(iterations):
        print('Training iteration',str(i+1))
        kmeans.train(input_fn)
        cluster_centers = kmeans.cluster_centers()
        # if previous_centers is not None:
        #   print('delta:', cluster_centers - previous_centers)
        previous_centers = cluster_centers
        # print('score:', kmeans.score(input_fn))
    # print('cluster centers:', cluster_centers)

    # map the input points to their clusters
    cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    for i, point in enumerate(DATA):
        cluster_index = cluster_indices[i]
        center = cluster_centers[cluster_index]
        print('point:', i, 'is in cluster', cluster_index)

    return cluster_indices


# Template from https://github.com/serengil/tensorflow-101/
# blob/master/python/KMeansClustering.py
###########################################################

def kmeans_v1(DATA, k, epochs):

    row = len(DATA)
    col = len(DATA[0])

    print("[", row, "x", col, "] sized input")

    model = tf.contrib.learn.KMeansClustering(
        # SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE
        k, distance_metric=clustering_ops.SQUARED_EUCLIDEAN_DISTANCE,
        initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT
    )


    def train_input_fn():
        data = tf.constant(DATA, tf.float32)
        return (data, None)


    model.fit(input_fn=train_input_fn, steps=epochs)

    print("--------------------")
    print("kmeans model: ", model)


    def predict_input_fn():
        return np.array(DATA, np.float32)


    predictions = model.predict(input_fn=predict_input_fn, as_iterable=True)

    results = {}
    for i, p in enumerate(predictions):
        results.update({i: p['cluster_idx']})

    return results

# SPOT CHECKING
###############
# Promising with 'Error_Message_Extended'
# Need to join cluster classification back to input dataset
# train['Stacktrace'].values[index]
