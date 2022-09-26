# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE


def get_data(path):
    data = np.load(path)

    x_vector = data['vector']
    label_vector = data['utt']
    x_vector = np.array(x_vector)
    label_vector = np.array(label_vector)

    return x_vector, label_vector


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    if np.min(label, 0) != np.max(label, 0):
        label_min, label_max = np.min(label, 0), np.max(label, 0)
        label = (label - label_min) / (label_max - label_min)
    else:
        label = label

    fig = plt.figure()
    ax = plt.subplot(111)

    plt.scatter(data[:, 0], data[:, 1], 10, c=label, cmap=plt.cm.Spectral, alpha=0.5)
    plt.title(title)
    return fig


def main(path0, epoch,root,backboon_name):
    data, labels_color = get_data(path0)
    print('Computing t-SNE embedding epoch')
    n_labels = len(set(labels_color))
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    result = tsne.fit_transform(data)

    fig = plot_embedding(result, labels_color, '')

    # path0 = root + backboon_name + '/train_data/z0_epoch{}.npz'.format(epoch)
    path  = root + backboon_name + '/tnes/z_vector_epoch{}_n_class={}.png'.format(epoch, n_labels)

    plt.savefig(path)
    plt.close()


# main('/home/daip/PycharmProjects/wxf/FeatureEx/DNF/marine/densennet/densenet121_pre/test_data/z0_epoch199.npz',-1)



