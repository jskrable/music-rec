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


def set_target(df):

    # Compare output of NN artist to this list of artists
    rel_cols = ['metadata_songs_song_id','metadata_songs_artist_id','metadata_songs_title','metadata_similar_artists']
    relatedDF = df[['metadata_songs_song_id','metadata_songs_title','metadata_similar_artists']]
    artist_ids = np.unique(np.concatenate(relatedDF.metadata_similar_artists.to_numpy(), axis=0))
    relatedDF['dummies'] = relatedDF.metadata_similar_artists.apply(lambda x: pd.get_dummies(x).values)


def proc_number_array_col(row, max_col):


    sample = np.ceil(row.flatten().shape[0]/30).astype(int)
    return np.pad(row.flatten(), max_col, 'constant')

    # np.ceil(row.flatten().shape[0])
    # # Slice array to get a constant length by sampling every n values based on length
    # songsDF.col.apply(lambda x: x.flatten()[1::int(x.flatten().shape[0]/30)])


def max_val_in_col(col):

    measurer = np.vectorize(len)
    measurer(col).max(axis=0) 


# Function to transform string fields into numerical data
def bag_of_words(corpus):
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(corpus)

    return x.toarray()


# Function to vectorize a column made up of numpy arrays containing strings
def proc_str_array_col(col):
    col = col.apply(lambda x: re.sub("['\n]", '', np.array2string(x))[1:-1])
    col = bag_of_words(col.values)
    return col


def vectorize(data):

    for col in data:

        if str(type(data[col].iloc[0])) == "<class 'str'>":
            pd.get_dummies(data[col]).values
        elif str(type(data[col].iloc[0])) == "<class 'numpy.ndarray'>":
            proc_str_array_col(data[col])
        else:
            proc_number_array_col()