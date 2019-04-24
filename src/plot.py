#!/usr/bin/env python3
# coding: utf-8
"""
plot.py
04-22-19
jack skrable
"""

import sys
import os
import logging as log
import imageio as im
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



def animate_training(path):

    std = pd.read_csv(path + '/std/logs.csv')
    hyb = pd.read_csv(path + '/hyb/logs.csv')

    df = pd.DataFrame.from_records({'epoch': std.Step, 'nn':std.Value, 'hyb':hyb.Value})

    for i in range(len(df)):
        sns.lineplot(df[:i].epoch, df[:i].nn, c='b')
        sns.lineplot(df[:i].epoch, df[:i].hyb, c='r')
        plt.ylim(0,0.7)
        plt.xlim(0,200)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(labels=['standard nn','k-means augmented nn'])
        plt.savefig('./animate/' + (str(i)+'.png').zfill(7), format='png', dpi=300)
        plt.close()


    image_dir = os.fsencode(path)
    files = os.listdir(image_dir)

    # Get list of filenames including path
    fnames = [path+os.fsdecode(f) for f in files if f.endswith(b'.png')]
    fnames.sort()

    # Read images
    images = [im.imread(f) for f in fnames]

    # Write GIF
    # Duration per frame
    im.mimsave(os.path.join(path+'animate.gif'), images, duration=0.10)