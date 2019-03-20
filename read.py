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
        # DF to hold single file info
        data = pd.DataFrame()
        # Walk nodes under root
        for item in s_hdf.root._f_walknodes():
            # Get name for column
            name = item._v_pathname[1:].replace('/','_')
            # Store arrays
            if type(item) is tables.earray.EArray:
                data[name] = [np.array(item)]
            # Store tables
            elif type(item) is tables.table.Table:
                # Get all columns
                cols =  item.coldescrs.keys()
                for row in item:
                    for col in cols:
                        col_name = '_'.join([name,col])
                        try:
                            data[col_name] = row[col]
                        except Exception as e:
                            print(e)

        # Append to main df
        allsongs = allsongs.append(data, ignore_index=True)
        # Close store for reading
        s_hdf.close()

    return allsongs


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


# def categorize_string_data(df):
#     None

def get_user_taste_data(filename):
    tasteDF = pd.read_csv('./TasteProfile/train_triplets_SAMPLE.txt', sep='\t', header=None, names={'user,song,count'})


def set_target(df):

    # Compare output of NN artist to this list of artists
    rel_cols = ['metadata_songs_song_id','metadata_songs_artist_id','metadata_songs_title','metadata_similar_artists']
    relatedDF = df[['metadata_songs_song_id','metadata_songs_title','metadata_similar_artists']]
    artist_ids = np.unique(np.concatenate(relatedDF.metadata_similar_artists.to_numpy(), axis=0))
    relatedDF['dummies'] = relatedDF.metadata_similar_artists.apply(lambda x: pd.get_dummies(x).values)


def proc_array_col(row, max_col):


    sample = np.ceil(row.flatten().shape[0]/30).astype(int)
    return np.pad(row.flatten(), max_col, 'constant')

    # np.ceil(row.flatten().shape[0])
    # # Slice array to get a constant length by sampling every n values based on length
    # songsDF.col.apply(lambda x: x.flatten()[1::int(x.flatten().shape[0]/30)])


def max_val_in_col(col):

    measurer = np.vectorize(len)
    measurer(col).max(axis=0) 

# MAIN
###############################################################################

t_start = time.time()
files = get_all_files('./MillionSongSubset/data', '.h5')

dev_set = files[:200]
songsDF = extract_song_data(dev_set)
# songsDF = extract_song_data(files)
t_extract = time.time()
print('\nGot', len(songsDF.index), 'songs in', round((t_extract-t_start), 2), 'seconds.')

print('Pre-processing extracted song data...')
songsDF = convert_byte_data(songsDF)
t_preproc = time.time()
print('Cleaned and processed',len(songsDF.index),'rows in',round((t_preproc - t_extract), 2), 'seconds.')




# print('Storing compiled song dataframe in HDF5...')
# songsDF.to_hdf('preprocessing/songs.h5', 'songs')
# print(songsDF)
