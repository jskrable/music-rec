#!/usr/bin/env python3
# coding: utf-8
"""
plot.py
04-22-19
jack skrable
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

def pca(X, archive=None):

    pca = decomposition.PCA(n_components=20).fit(X)
    X = pca.transform(X)
    return X


def lda(X, y, archive=None):

    lda = LDA(n_components=19)
    X = lda.fit_transform(X, y)
    return X


def tsne(X, archive=None):

    X = TSNE(n_components=2, verbose=1).fit_transform(X)
                # learning_rate=500.0,
                # perplexity=35).fit_transform(X)

    return X



def plot_tsne(X, y):

    print('Condensing datapoint dimensions...')
    # clusters = X[:,-1]
    # X = np.delete(X,-1,1)
    X = tsne(pca(X))
    df = pd.DataFrame({'x':X[:,0], 'y':X[:,1], 'c':y})
    print('Plotting...')
    sns.scatterplot(df.x, df.y, hue=df.c)
    plt.show()


def plot_nn_training(path, col):

    std = pd.read_csv(path + '/std/logs.csv')
    hyb = pd.read_csv(path + '/hyb/logs.csv')

    sns.lineplot(std.epoch, std[col])
    sns.lineplot(std.epoch, std['val_'+ col])
    sns.lineplot(hyb.epoch, hyb[col])
    sns.lineplot(hyb.epoch, hyb['val_'+ col])
    plt.legend(labels=['standard train ' + col,
                       'standard test ' + col,
                       'classified train ' + col,
                       'classified val ' + col])

    plt.grid()
    plt.tight_layout()
    plt.show()
