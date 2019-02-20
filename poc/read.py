import os
import tables
import pandas as pd
import numpy as np
from pandas import DataFrame, HDFStore

temp = HDFStore('./MillionSongSubset/data/A/A/A/TRAAAAW128F429D538.h5')
song = pd.DataFrame.from_records(temp.root.metadata.songs[:])
artist song.artist_name[0].decode()

store = HDFStore('test.h5')

def get_all_files(basedir,ext='.h5') :
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Return all absolute paths in a list.
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files :
            allfiles.append( os.path.abspath(f) )
    return allfiles


# From a list of h5 files, extracts song metadata and creates a dataframe
def extract_song_data(files):
	allsongs = pd.DataFrame()
    limit = 1000
    for i in range(0,limit):
    	print('Getting',f)
        s_hdf = pd.HDFStore(files[i])
        # print(s_hdf)
        s_df = pd.DataFrame.from_records(s_hdf.root.metadata.songs[:])
        # print(s_df)
        allsongs = allsongs.append(s_df, ignore_index=True)

    return allsongs


# Main
files = get_all_files('./MillionSongSubset/data','.h5')
songsDF = extract_song_data(files)
