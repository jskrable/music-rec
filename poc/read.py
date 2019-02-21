import os
import sys
import tables
import glob
import time
import pandas as pd
import numpy as np
from pandas import DataFrame, HDFStore

# temp = HDFStore('./MillionSongSubset/data/A/A/A/TRAAAAW128F429D538.h5')
# song = pd.DataFrame.from_records(temp.root.metadata.songs[:])
# artist = song.artist_name[0].decode()
# temp.close()

# store = HDFStore('test.h5')

# Progress bar for cli
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def get_all_files(basedir, ext='.h5'):
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Return all absolute paths in a list.
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        for f in files:
            allfiles.append(os.path.abspath(f))
    return allfiles


# From a list of h5 files, extracts song metadata and creates a dataframe
def extract_song_data(files):
    allsongs = pd.DataFrame()
    size = len(files)
    print(size,'files found.')
    for i, f in enumerate(files):
        progress(i, size, 'of files processed')
        # print(complete, 'of files processed', end='\r', flush=True)
        s_hdf = pd.HDFStore(f)
        s_df = pd.DataFrame.from_records(s_hdf.root.metadata.songs[:])
        allsongs = allsongs.append(s_df, ignore_index=True)
        s_hdf.close()

    return allsongs


# Takes a dataframe full of encoded strings and cleans it
def convert_byte_data(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]

    return df


# Main
t1 = time.time()
print('Getting list of all h5 files...')
files = get_all_files('./MillionSongSubset/data', '.h5')
t2 = time.time()
songsDF = extract_song_data(files)
t3 = time.time()

print('\n','Got', len(songsDF.index), 'songs in', round((t3-t1), 2), 'seconds.')

print('Storing in HDF5...')
songsDF = convert_byte_data(songsDF)
songsDF.to_hdf('preprocessing/songs.h5', 'songs')
# print(songsDF)
