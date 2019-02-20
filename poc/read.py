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