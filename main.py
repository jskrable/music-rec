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
import read_h5 as read
import preprocessing as pp


# MAIN
###############################################################################

t_start = time.time()
songsDF = read.h5_to_df('./MillionSongSubset/data', 200)
t_extract = time.time()
print('\nGot', len(songsDF.index), 'songs in', round((t_extract-t_start), 2), 'seconds.')

print('Pre-processing extracted song data...')
songsDF = pp.convert_byte_data(songsDF)
t_preproc = time.time()
print('Cleaned and processed',len(songsDF.index),'rows in',round((t_preproc - t_extract), 2), 'seconds.')

x, y = pp.vectorize(songsDF, 'metadata_similar_artists')
# print('Storing compiled song dataframe in HDF5...')
# songsDF.to_hdf('preprocessing/songs.h5', 'songs')
# print(songsDF)