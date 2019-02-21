import os
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

<<<<<<< HEAD

=======
>>>>>>> 3de93ddae4ad8209e2b55faa2c029f2b89b9c07e
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
    limit = 1000
    size = len(files)
<<<<<<< HEAD
    # limit = size
    for i in range(0, limit):
        complete = '[ '+str(round((i/limit)*100, 2))+'% ]'
        print(complete, 'of files processed', end='\r', flush=True)
=======
    limit = size
    for i in range(0, limit):
        complete = str(round((i/limit)*100,2))
        print(complete+'% of files processed', end='\r', flush=True)
>>>>>>> 3de93ddae4ad8209e2b55faa2c029f2b89b9c07e
        s_hdf = pd.HDFStore(files[i])
        s_df = pd.DataFrame.from_records(s_hdf.root.metadata.songs[:])
        allsongs = allsongs.append(s_df, ignore_index=True)
        s_hdf.close()
<<<<<<< HEAD

    return allsongs


# Takes a dataframe full of encoded strings and cleans it
def convert_byte_data(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]

    return df
=======

    return allsongs
>>>>>>> 3de93ddae4ad8209e2b55faa2c029f2b89b9c07e


# Main
t1 = time.time()
files = get_all_files('./MillionSongSubset/data', '.h5')
t2 = time.time()
songsDF = extract_song_data(files)
t3 = time.time()

print('Got', len(songsDF.index), 'songs in', round((t3-t1), 2), 'seconds.')
<<<<<<< HEAD

print('Storing in HDF5...')
songsDF = convert_byte_data(songsDF)
songsDF.to_hdf('preprocessing/songs.h5', 'songs')
=======
>>>>>>> 3de93ddae4ad8209e2b55faa2c029f2b89b9c07e
# print(songsDF)
