#!/usr/bin/env python3
# coding: utf-8
"""
utils.py
04-10-19
jack skrable
"""

import os
import argparse
import time
import datetime
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
         

def setup_model_dir():
	t = time.time()
	dt = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
	path = './model/' + dt
	os.mkdir(path)
	os.mkdir(path + '/preprocessing')

	return path


def arg_parser():
    # function to parse arguments sent to CLI
    # setup argument parsing with description and -h method
    parser = argparse.ArgumentParser(
        description='Music recommendation engine using a neural network')
    # add size int
    parser.add_argument('-s', '--size', default=10000, type=int, nargs='?',
                        help='the number of files to use for training')
    # add iterations int
    parser.add_argument('-i', '--initialize', default=False, type=bool, nargs='?',
                        help='flag to run initial setup for web app')
    # parse args and return
    args = parser.parse_args()
    return args