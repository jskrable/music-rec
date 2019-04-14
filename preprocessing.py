#!/usr/bin/env python3
# coding: utf-8
"""
preprocessing.py
02-22-19
jack skrable
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler


# def normalize_array(arr, scale):
#     arr = arr.astype(np.float32)
#     arr *= (scale / np.abs(arr).max())
#     return arr


def max_length(col):
    measurer = np.vectorize(len)
    return measurer(col).max(axis=0) 

def min_length(col):
    measurer = np.vectorize(len)
    return measurer(col).min(axis=0) 


# # Function to transform string fields into numerical data
# def bag_of_words(corpus):
#     vectorizer = CountVectorizer()
#     x = vectorizer.fit_transform(corpus)

#     # NEED TO RETURN MAPPING FOR Y ALSO
#     # vectorizer.inverse_transform(x)
#     return x.toarray()


def sample_ndarray(row):
    SAMPLE_SIZE = 30
    sample = np.ceil(row.flatten().shape[0]/SAMPLE_SIZE).astype(int)
    z = []
    for i,r in enumerate(row):
        if (i % sample) == 0:
            z.append(r)
    return np.concatenate(z).astype(np.float)


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


def lookup_discrete_id(row, m):
    _, row, _ = np.intersect1d(m, row, assume_unique=True, return_indices=True)
    return row


# Function to vectorize a column made up of numpy arrays containing strings
def process_metadata_list(col):
    x_map, _ = np.unique(np.concatenate(col.values, axis=0), return_inverse=True)
    # col = col.apply(lambda x: re.sub("['\n]", '', np.array2string(x))[1:-1])
    col = col.apply(lambda x: lookup_discrete_id(x, x_map))
    max_len = max_length(col)
    col = col.apply(lambda x: np.pad(x, (0, max_len - x.shape[0]), 'constant'))
    xx = np.stack(col.values)
    return xx


# Function to translate target artist list into discrete integer ids
def categorical(col):
    # Simplify to one artist
    col = col.apply(lambda x: x[0])
    # Get all unique values, unpack into a map
    # TODO reassign these to the right row
    y_map, y = np.unique(col.values, return_inverse=True)
    return y_map, y


def scaler(X, range):
    mms = MinMaxScaler()
    return mms.fit_transform(X)


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

    return df


# Function to vectorize full dataframe 
def vectorize(data, label):

    print('Vectorizing dataframe...')
    output = np.zeros(shape=(len(data),0))

    for col in data:
        try:
            print('Vectorizing ',col)
            if col == label:
                y_map, y = categorical(data[col])

            elif data[col].dtype == 'O':
                if str(type(data[col].iloc[0])) == "<class 'str'>":
                    # print('case 1',col)
                    xx = pd.get_dummies(data[col]).values
                elif col.split('_')[0] == 'metadata':
                    # print('case 2',col)
                    xx = process_metadata_list(data[col])
                else:
                    # print('case 3',col)
                    xx = process_audio(data[col])

            else:
                # print('case 4', col)
                xx = data[col].values[...,None]

            output = np.hstack((output, xx))
        except Exception as e:
            print(col)
            print(e)

    # CHANGE THIS TO -1, 1??????????
    # SCALE ONLY ON TRAIN SET?????
    # THEN SPLIT TO TEST/VALID??
    output = scaler(output, 1)

    return output, y, y_map



# Function to compare model input and output
# MOVE TO A NEW MODULE????
def target_vocab(data, col, y):

    # Init count vectorizer
    vec = CountVectorizer()
    vec.fit_transform(data[col].values)

    # Create the lookup list ordered correctly by index
    terms = np.array(list(vec.vocabulary_.keys()))
    indices = np.array(list(vec.vocabulary_.values()))
    inverse_vocabulary = terms[np.argsort(indices)]

    # Get input data and output data
    # TODO vectorize this, too slow
    for i, v in data[col].iteritems():
        source = np.sort(np.char.lower(v))
        pred = np.sort(np.array([inverse_vocabulary[i] for i, v in enumerate(y[i]) if v == 1]))
        # Get intersection of arrays
        matching = np.intersect1d(source, pred)

        print(matching.shape[0])



def set_target(df):
    # Compare output of NN artist to this list of artists
    rel_cols = ['metadata_songs_song_id','metadata_songs_artist_id','metadata_songs_title','metadata_similar_artists']
    relatedDF = df[['metadata_songs_song_id','metadata_songs_title','metadata_similar_artists']]
    artist_ids = np.unique(np.concatenate(relatedDF.metadata_similar_artists.to_numpy(), axis=0))
    relatedDF['dummies'] = relatedDF.metadata_similar_artists.apply(lambda x: pd.get_dummies(x).values)

