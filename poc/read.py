#!/usr/bin/env python3
# coding: utf-8
"""
read.py
02-22-19
jack skrable
"""

import os
import sys
import tables
import glob
import time
import pandas as pd
import numpy as np
# from pandas import DataFrame, HDFStore


# Progress bar for cli
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


# Get list of all h5 files in basedir
def get_all_files(basedir, ext='.h5'):
    print('Getting list of all h5 files in',basedir)
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        for f in files:
            allfiles.append(os.path.abspath(f))
    return allfiles


# From a list of h5 files, extracts song metadata and creates a dataframe
def extract_song_data(files):
    # Init empty df
    allsongs = pd.DataFrame()
    # Get total h5 file count
    size = len(files)
    print(size, 'files found.')
    # Iter thru files
    for i, f in enumerate(files):
        # Update progress bar
        progress(i, size, 'of files processed')
        # Read file into store
        s_hdf = pd.HDFStore(f)
        # Create dfs from store tables
        # TODO get numpy arrays here too
        # set(s_hdf)
        # for row in s_hdf.root.analysis.segments_timbre.iterrows():
        #     print(row)
        meta = pd.DataFrame.from_records(s_hdf.root.metadata.songs[:])
        meta['artist_terms'] = [np.array(temp.root.metadata.artist_terms)]

        data = pd.DataFrame()
        for item in temp.root._f_walknodes():
            name = item._v_pathname[1:].replace('/','_')
            if type(item) is tables.earray.EArray:
                data[name] = [np.array(item)]
            elif type(item) is tables.table.Table:
                cols =  item.coldescrs.keys()
                for row in item:
                    for col in cols:
                        col_name = '_'.join([name,col])
                        try:
                            data[col_name] = row[col]
                        except Exception as e:
                            print(e)




        # combine into song df
        song = pd.concat([meta, analysis], axis=1, sort=False)
        # Append to main df
        allsongs = allsongs.append(song, ignore_index=True)
        # Close store for reading
        s_hdf.close()

    return allsongs


# Takes a dataframe full of encoded strings and cleans it
def convert_byte_data(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]

    return df


# MAIN
###############################################################################

t1 = time.time()
files = get_all_files('./MillionSongSubset/data', '.h5')
t2 = time.time()
songsDF = extract_song_data(files)
t3 = time.time()

print('\nGot', len(songsDF.index), 'songs in', round((t3-t1), 2), 'seconds.')

print('Storing compiled song dataframe in HDF5...')
songsDF = convert_byte_data(songsDF)
songsDF.to_hdf('preprocessing/songs.h5', 'songs')
# print(songsDF)
