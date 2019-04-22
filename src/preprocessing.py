#!/usr/bin/env python3
# coding: utf-8
"""
preprocessing.py
02-22-19
jack skrable
"""

import json
import pandas as pd
import numpy as np
import genre_splitter as gs
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.externals import joblib


# Globals
max_list = {}
maps = {}


def max_length(col):
    measurer = np.vectorize(len)
    return measurer(col).max(axis=0) 


def min_length(col):
    measurer = np.vectorize(len)
    return measurer(col).min(axis=0) 


def sample_ndarray(row):
    SAMPLE_SIZE = 30
    sample = np.ceil(row.flatten().shape[0]/SAMPLE_SIZE).astype(int)
    output = np.concatenate([r for i, r in enumerate(row) if i % sample == 0]).astype(np.float)
    if output.size != 36:
        output = np.pad(output, (36 - output.size)//2, 'constant')
    return output


def sample_flat_array(row):
    SAMPLE_SIZE = 28
    if row.shape[0] <= SAMPLE_SIZE:
        s = 1
    else:
        s = row.shape[0] // SAMPLE_SIZE

    x = [r for i, r in enumerate(row) if i % s == 0]

    if len(x) > SAMPLE_SIZE:
        mid = len(x) // 2
        x = x[int(mid-(SAMPLE_SIZE/2)):int(mid+(SAMPLE_SIZE/2))]
    else:
        x = np.pad(x, (0, SAMPLE_SIZE - len(x)), 'constant')
    
    return np.array(x).astype(np.float)


def process_audio(col):
    dim = len(col.iloc[0].shape)
    # size = max_length(col)

    if dim > 1:
        col = col.apply(sample_ndarray)
    else:
        col = col.apply(sample_flat_array)

    xx = np.stack(col.values)
    return xx


# Function to vectorize a column made up of numpy arrays containing strings
def process_metadata_list(col, archive=None):
    x_map, _ = np.unique(np.concatenate(col.values, axis=0), return_inverse=True)
    col = col.apply(lambda x: lookup_discrete_id(x, x_map))
    max_len = max_length(col)
    if archive is not None:
        max_list.update({col.name : int(max_len)})
    col = col.apply(lambda x: np.pad(x, (0, max_len - x.shape[0]), 'constant'))
    xx = np.stack(col.values)

    return xx, x_map


def lookup_discrete_id(row, m):
    _, row, _ = np.intersect1d(m, row, assume_unique=True, return_indices=True)
    return row


# Need to save this scaler
def scaler(X, kind='robust', archive=None):
    if kind == 'mms':
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    scaler = scaler.fit(X)
    if archive is not None:
        joblib.dump(scaler, archive+'/preprocessing/'+kind+'.scaler')

    return scaler.transform(X)


# Takes a dataframe full of encoded strings and cleans it
def convert_byte_data(df):

    print('Cleaning byte data...')
    obj_df = df.select_dtypes([np.object])

    str_cols = []
    np_str_cols = []
    for col in set(obj_df):
        if isinstance(obj_df[col][0], bytes):
            str_cols.append(col)
        elif str(obj_df.metadata_artist_terms[0].dtype)[1] == 'S':
            np_str_cols.append(col)

    str_df = obj_df[str_cols]
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]

    np_str_df = obj_df[np_str_cols]
    for col in np_str_cols:
        try:
            print('Cleaning ',col)
            df[col] = np_str_df[col].apply(lambda x: x.astype('U'))
        except UnicodeDecodeError as e:
            print(e)

    print('Cleaning complete.')

    return df


# Function to vectorize full dataframe 
def vectorize(data, label, archive=None, predict=False):

    print('Vectorizing dataframe...')
    # output = np.zeros(shape=(len(data),1))

    for col in data:
        print('Vectorizing ',col) 
        if col == label:
            y_map, y = np.unique(data[col].values, return_inverse=True)
            maps.update({col: y_map.tolist()})
        else:        
            if data[col].dtype == 'O':
                if type(data[col].iloc[0]) is str:
                    x_map , xx = np.unique(data[col].values, return_inverse=True)
                    xx = xx.reshape(-1,1)
                    maps.update({col: x_map.tolist()})
                elif col.split('_')[0] == 'metadata':
                    if archive is None:
                        xx, x_map = process_metadata_list(data[col])
                    else:
                        xx, x_map = process_metadata_list(data[col], archive)
                        maps.update({col: x_map.tolist()})
                else:
                    xx = process_audio(data[col])
            else:
                xx = data[col].values[...,None]

            # if xx.shape[0] == len(data):
            # Normalize each column    
            xx = xx / (np.linalg.norm(xx) + 0.00000000000001)
            try:
                output = np.hstack((output, xx))
            except NameError:
                output = xx

    if archive is not None:
        with open(archive + '/preprocessing/max_list.json', 'w') as file:
            json.dump(max_list, file)
        with open(archive + '/preprocessing/maps.json', 'w') as file:
            json.dump(maps, file)

    print('Vectorization complete.')
    return output, y, y_map


def create_target_classes(df):

    print('Calculating overarching genre for labels...')
    df['target'] = df.metadata_artist_terms.apply(gs.target_genre)
    return df

