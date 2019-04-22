#!/usr/bin/env python3
# coding: utf-8
"""
plot.py
04-22-19
jack skrable
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE

def pca(X, archive=None):

    pca = decomposition.PCA(n_components=50).fit(X)
    X = pca.transform(X)
    return X


def tsne(X, archive=None):

    X = TSNE(n_components=2, verbose=1,
                learning_rate=500.0,
                perplexity=40).fit_transform(X)

    return X



def plot(X, clusters):

    print('Condensing datapoint dimensions...')
    # clusters = X[:,-1]
    # X = np.delete(X,-1,1)
    X = tsne(pca(X))
    df = pd.DataFrame({'x':X[:,0], 'y':X[:,1], 'c':clusters})
    print('Plotting...')
    sns.scatterplot(df.x, df.y, hue=df.c)
    plt.show()


