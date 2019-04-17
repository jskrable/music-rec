#!/usr/bin/env python3
# coding: utf-8
"""
taste.py
04-16-19
jack skrable
"""

import pandas as pd
import numpy as np


def read_user_taste():

    global tasteDF
    print('Reading user taste csv...')
    tasteDF = pd.read_csv('./data/TasteProfile/train_triplets.txt', sep='\t', header=None, names=['user','song','listens'])

    print('Getting top listens...')
    tasteDF = tasteDF.loc[tasteDF.groupby(['song'])["listens"].idxmax()] 
    # tastedf.sort_values(by=['song','listens'], ascending=False, inplace=True)    
    # tastedf.set_index('song', inplace=True)

    return tasteDF


def get_tastemaker(col):

    global tasteDF
    user = tasteDF.loc[tasteDF.song == col].user.values
    if user.size == 0:
        user = '0'
    else:
        user = user[0]
    return user