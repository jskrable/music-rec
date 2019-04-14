#!/usr/bin/env python3
# coding: utf-8
"""
utils.py
04-10-19
jack skrable
"""

import numpy as np
# CHECK MODEL
def model_check(X, n):
    for i in range(n):
        chk = np.random.randint(songsDF.shape[0])
        assert songsDF.metadata_similar_artists.iloc[chk][0] == y_map[np.argmax(
            model_simple.predict(X[chk].reshape(1, -1)))]