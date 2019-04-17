#!/usr/bin/env python3
# coding: utf-8
"""
main.py
03-29-19
jack skrable
"""

# Library imports
import time
import pandas as pd
import numpy as np

# Custom imports
import utils
import read_h5 as read
import preprocessing as pp
import neural_net as nn
import kmeans as km


# Read data from h5 files into dataframe
###############################################################################
t_start = time.time()
df = read.h5_to_df('./data/MillionSongSubset/data', 10000)
t_extract = time.time()
print('\nGot', len(df.index), 'songs in',
      round((t_extract-t_start), 2), 'seconds.')

# Transform data into vectors for processing by neural network
###############################################################################
print('Pre-processing extracted song data...')
df = pp.convert_byte_data(df)
df = pp.create_target_classes(df)
X, y, y_map = pp.vectorize(df, 'target')
t_preproc = time.time()
print('Cleaned and processed', len(df.index), 'rows in',
      round((t_preproc - t_extract), 2), 'seconds.')

# Train neural network
###############################################################################
print('Training neural network...')
print('[', X.shape[1], '] x [', y.shape[0], ']')
model_simple = nn.deep_nn(X, y, 'std')
# nn.deep_nn(X, y)
t_nn = time.time()
print('Neural network trained in', round((t_nn - t_preproc), 2), 'seconds.')
# Check model
# utils.model_check(X, y_map, 10, df, model_simple)

# Perform k-Means clustering and send classified data through neural network
###############################################################################
clusters = 10
print('Applying k-Means classifier with', clusters, 'clusters...')
kmX = km.kmeans(X, clusters)
print('Complete.')
print('Training neural network...')
print('[', kmX.shape[1], '] x [', y.shape[0], ']')
model_classified = nn.deep_nn(kmX, y, 'hyb')
t_km = time.time()
print('Hybrid k-Means neural network trained in', round((t_km - t_nn), 2), 'seconds.')
# Check model
# utils.model_check(kmX, y_map, 10, df, model_classified)