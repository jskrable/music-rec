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
import pandas as pd
import numpy as np


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
    df = pd.DataFrame()
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
        df = df.append(data, ignore_index=True)
        # Close store for reading
        s_hdf.close()

    # Dev set of columns
    # df = df[['metadata_songs_artist_id','metadata_songs_title','musicbrainz_songs_year','metadata_artist_terms','analysis_songs_analysis_sample_rate','metadata_songs_artist_location','analysis_sections_confidence','analysis_sections_start','analysis_segments_start','analysis_segments_timbre','analysis_segments_pitches','analysis_songs_tempo','analysis_bars_confidence','analysis_bars_start','analysis_beats_confidence','analysis_beats_start','analysis_songs_duration','analysis_songs_energy','analysis_songs_key','analysis_songs_key_confidence','analysis_songs_time_signature','analysis_songs_time_signature_confidence','metadata_similar_artists']]

    # Drop bad columns
    df.drop(['musicbrainz_artist_mbtags_count','musicbrainz_artist_mbtags',
             'musicbrainz_songs_idx_artist_mbtags'], inplace=True, axis=1)

    return df


def get_song_file_map(files):

    # Init empty df
    songmap = {}
    # Get total h5 file count
    size = len(files)
    print(size, 'files found.')
    # Iter thru files
    for i, f in enumerate(files):
        # Update progress bar
        progress(i, size, 'of files processed')
        # Read file into store
        s_hdf = pd.HDFStore(f)
        song_id = s_hdf.root.metadata.songs[0]['song_id'].astype('U')
        filepath = s_hdf.filename
        songmap.update({song_id: s_hdf.filename})
        # Close store for reading
        s_hdf.close()

    with open('./data/song-file-map.json', 'w') as file:
        json.dump(songmap, file, sort_keys=True, indent=2)

    return songmap


def get_user_taste_data(filename):
    tasteDF = pd.read_csv('./TasteProfile/train_triplets_SAMPLE.txt', sep='\t', header=None, names={'user,song,count'})

    return tasteDF


# Function to read all h5 files in a directory into a dataframe
def h5_to_df(basedir, limit=None, init=False):
    files = get_all_files(basedir, '.h5')
    files = files if limit is None else files[:limit]
    df = extract_song_data(files)

    if init:
        get_song_file_map(files)
    return df