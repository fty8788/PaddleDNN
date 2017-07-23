#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os

__all__ = ['train', 'test']

TRAIN_DATA = None
TEST_DATA = None


def feature_range(maximums, minimums):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    feature_num = len(maximums)
    ax.bar(range(feature_num), maximums - minimums, color='r', align='center')
    ax.set_title('feature scale')
    plt.xticks(range(feature_num), feature_names)
    plt.xlim([-1, feature_num])
    fig.set_figheight(6)
    fig.set_figwidth(10)
    if not os.path.exists('./image'):
        os.makedirs('./image')
    fig.savefig('image/ranges.png', dpi=48)
    plt.close(fig)


def load_data(filename, feature_num=14, ratio=0.8):
    global TRAIN_DATA, TEST_DATA
    if TRAIN_DATA is not None and TEST_DATA is not None:
        return
    
    data = np.fromfile(filename, sep=' ')
    data = data.reshape(data.shape[0] / feature_num, feature_num)
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]
    #feature_range(maximums[:-1], minimums[:-1])
    for i in xrange(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    for i in xrange(len(data)):
        x = int(data[i, -1])
        data[i, -1] = x
    offset = int(data.shape[0] * ratio)

    print data.shape[0]
    TRAIN_DATA = data[:offset]
    TEST_DATA = data[offset:]


def train(data_path, split_num):
    """
    UCI_HOUSING training set creator.

    It returns a reader creator, each sample in the reader is features after
    normalization and price number.

    :return: Training reader creator
    :rtype: callable
    """
    global TRAIN_DATA
    load_data(data_path, split_num)
    #load_data("housing.data")

    def reader():
        for d in TRAIN_DATA:
            yield d[:-1], int(d[-1])

    return reader


def test(data_path, split_num):
    """
    UCI_HOUSING test set creator.

    It returns a reader creator, each sample in the reader is features after
    normalization and price number.

    :return: Test reader creator
    :rtype: callable
    """
    global TEST_DATA
    load_data(data_path, split_num)
    #load_data("housing.data")

    def reader():
        for d in TEST_DATA:
            yield d[:-1], int(d[-1:])

    return reader


