#!/usr/bin/env python3
# coding: utf-8
"""
utils.py
04-10-19
jack skrable
"""

import numpy as np
import pandas as pd


# CHECK MODEL
def model_check(X, y_map, n, df, model):
    for i in range(n):
        chk = np.random.randint(df.shape[0])
        assert df.metadata_similar_artists.iloc[chk][0] == y_map[np.argmax(
            model.predict(X[chk].reshape(1, -1)))]


def save_lookup_file(df):

    lookupDF = df[['metadata_songs_song_id','metadata_songs_artist_id','metadata_songs_title','metadata_songs_artist_name','musicbrainz_songs_year','metadata_songs_release']]

    convert_to_byte_data(lookupDF)

    lookupDF.to_hdf('./frontend/data/lookup.h5', key='df', mode='w')

    pd.read_hdf('./frontend/data/lookup.h5', 'df')
         
