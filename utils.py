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
def model_check(X, n):
    for i in range(n):
        chk = np.random.randint(songsDF.shape[0])
        assert songsDF.metadata_similar_artists.iloc[chk][0] == y_map[np.argmax(
            model_simple.predict(X[chk].reshape(1, -1)))]


lookupDF = df[['metadata_songs_song_id','metadata_songs_artist_id','metadata_songs_title','metadata_songs_artist_name','musicbrainz_songs_year','metadata_songs_release']]

convert_to_byte_data(lookupDF)

lookupDF.to_hdf('./frontend/data/lookup.h5', key='df', mode='w')

pd.read_hdf('./frontend/data/lookup.h5', 'df')
         
