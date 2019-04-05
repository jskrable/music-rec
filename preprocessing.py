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
from sklearn.preprocessing import normalize


def normalize_array(arr, scale):
    arr = arr.astype(np.float32)
    arr *= (scale / np.abs(arr).max())
    return arr


def max_length(col):
    measurer = np.vectorize(len)
    return measurer(col).max(axis=0) 


# Function to transform string fields into numerical data
def bag_of_words(corpus):
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(corpus)

    # NEED TO RETURN MAPPING FOR Y ALSO
    # vectorizer.inverse_transform(x)
    return x.toarray()


def sample_ndarray(row):
    SAMPLE_SIZE = 30
    sample = np.ceil(row.flatten().shape[0]/SAMPLE_SIZE).astype(int)
    z = []
    for i,r in enumerate(row):
        if (i % sample) == 0:
            z.append(normalize_array(r,1))
    return np.concatenate(z)


# TODO work on this, need to standardize size better
def sample_flat_array(row, size):
    row = row.astype(np.float)
    # Pad shorter arrays with zeros to the max length
    # Maybe change to pad to mean length???
    z = np.asarray(np.pad(row, (0, size - row.shape[0]), 'constant'))
    return normalize_array(z, 1)


def process_audio(col):
    dim = len(col.iloc[0].shape)
    size = max_length(col)

    if dim > 1:
        col = col.apply(sample_ndarray)
    else:
        col = col.apply(lambda x: sample_flat_array(x, size))

    xx = np.stack(col.values)
    return xx


# Function to vectorize a column made up of numpy arrays containing strings
def process_metadata_list(col):
    col = col.apply(lambda x: re.sub("['\n]", '', np.array2string(x))[1:-1])
    xx = bag_of_words(col.values)
    return xx


# Simplify target to one artist
def categorical(col):
    col = col.apply(lambda x: x[0])
    y_map, y = np.unique(col.values, return_inverse=True)
    return y_map, y


# Takes a dataframe full of encoded strings and cleans it
def convert_byte_data(df):

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
            df[col] = np_str_df[col].apply(lambda x: x.astype('U'))
        except UnicodeDecodeError as e:
            print(e)

    return df


# Function to vectorize full dataframe 
def vectorize(data, label):

    output = np.zeros(shape=(len(data),0))

    # TODO Solve for all caught exceptions
    # DONE, can probably drop the try-catch
    for col in data:
        try:
            if col == label:
                y_map, y = categorical(data[col])

            elif data[col].dtype == 'O':
                if str(type(data[col].iloc[0])) == "<class 'str'>":
                    xx = pd.get_dummies(data[col]).values
                elif col.split('_')[0] == 'metadata':
                    xx = process_metadata_list(data[col])
                else: 
                    # MORE CONDITIONS HERE 
                    xx = process_audio(data[col])

            output = np.hstack((output, xx))
        except Exception as e:
            print(col)
            print(e)

    return output, y



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

