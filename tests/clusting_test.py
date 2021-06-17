#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" clusting_test.py
Description: 
"""
__author__ = "Anthony Fong"
__copyright__ = "Copyright 2021, Anthony Fong"
__credits__ = ["Anthony Fong"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Anthony Fong"
__email__ = ""
__status__ = "Prototype"

# Default Libraries #
import datetime

# Downloaded Libraries #
import numpy as np
from sklearn.cluster import MeanShift

# Local Libraries #
from src.hdf5xltek import *


# Definitions #
# Functions #
def find_spikes(x, c, top=5000, bottom=-5000, b=None):
    high = np.asarray(x[:, c] > top).nonzero()
    low = np.asarray(x[:, c] < bottom).nonzero()
    bounds = np.append(high[0], low[0], 0)
    bounds = np.sort(bounds)
    X = bounds.reshape(-1, 1)

    ms = MeanShift(bandwidth=b)
    ms.fit(X)
    labels = ms.labels_
    c_centers = ms.cluster_centers_
    sep = separate_clusters(X, labels)
    return labels, c_centers, sep


def separate_clusters(x, labels):
    n_clusters = len(np.unique(labels))
    sep = [[] for i in range(n_clusters)]
    for v, c in zip(x, labels):
        sep[int(c)].append(int(v))
    return sep


# Main #
if __name__ == "__main__":
    v_c = list(range(72, 80))
    # v_c = [30, 59, 51, 43, 35]
    first = datetime.datetime(2018, 10, 17, 9, 34, 00)
    second = datetime.datetime(2018, 10, 17, 12, 5, 30)

    study = HDF5XLTEKstudy('EC188')

    d, f, g = study.data_range_time(first, second, frame=True)
    print(study.find_time(datetime.datetime(2019, 1, 24, 9, 40, 4)))

    fs = d[0].frames[0].sample_rate
    task = d[0][5576000:7700000, :]

    all_viewer = EEGScanner(d[0][8934000:11750000], v_c, ylim=2000, show=True)
    #all_viewer = eegscanner.eegscanner(task, v_c, show=True)
    task2 = [d[0][8934000:9269000], d[0][9450000:9686000], d[0][9750000:10060000], d[0][10060000:10380000],
             d[0][10380000:11000000], d[0][11000000:11500000], d[0][11500000:11750000]]
    tests = [task[:420000, :], task[420000:715000, :], task[715000:1020000, :],
             task[1020000:1300000, :], task[1300000:1560000, :], task[1560000:1830000, :], task[1830000:, :]]

    all_t = tests+task2
    #all_t = task2

    pre = 128
    length = 128
    total = length+pre
    meme = []
    s_chans = [79, 79, 79, 79, 79, 79, 72, 72, 72, 72, 72, 72, 72, 72]
    # s_chans = [51, 51, 3, 3, 79, 51, 51]
    # valid = ((0, -3), (4, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1))
    inx = np.transpose(np.arange(32 - 1, 0 - 1, -1).reshape(4, 8)).flatten()
    inx2 = np.transpose(np.arange(64 - 1, 32 - 1, -1).reshape(4, 8)).flatten()
    cn1 = [['32: parstriangularis', '24: Parsopercularis',  '16: Parsopercularis',  '8: Precentral'],
           ['31: parstriangularis', '23: Parsopercularis',  '15: Precentral',       '7: Precentral'],
           ['30: parstriangularis', '22: Parsopercularis',  '14: Precentral',       '6: Precentral'],
           ['29: parstriangularis', '21: Precentral',       '13: Precentral',       '5: Postcentral'],
           ['28: Superiortemporal', '20: Superiortemporal', '12: Superiortemporal', '4: Superiortemporal'],
           ['27: Middletemporal',   '19: Middletemporal',   '11: Middletemporal',   '3:Superiortemporal'],
           ['26: Middletemporal',   '18: Middletemporal',   '10: Middletemporal',   '2: Middletemporal'],
           ['25: Middletemporal',   '17: Middletemporal',   '9: Middletemporal',    '1: Middletemporal']]
    cn1 = np.array(cn1, str)

    #cn2 = np.transpose(np.arange(64 - 1, 32 - 1, -1).reshape(4, 8)) + 1
    cn2 = [['64: Postcentral',      '56: Supramarginal',    '48: Postcentral',      '40: Supramarginal'],
           ['63: Postcentral',      '55: Supramarginal',    '47: Supramarginal',    '39: Supramarginal'],
           ['62: Postcentral',      '54: Supramarginal',    '46: Supramarginal',    '38: Supramarginal'],
           ['61: Superiortemporal', '53: Supramarginal',    '45: Supramarginal',    '37: Superiortemporal'],
           ['60: Superiortemporal', '52: Superiortemporal', '44: Superiortemporal', '36: Superiortemporal'],
           ['59: Middletemporal',   '51: Middletemporal',   '43: Middletemporal',   '35: Middletemporal'],
           ['58: Middletemporal',   '50: Middletemporal',   '42: Middletemporal',   '34: Middletemporal'],
           ['57: Middletemporal',   '49: Inferiortemporal', '41: Middletemporal',   '33: Middletemporal']]
    cn2 = np.array(cn2, str)


    t_name = ['PMGC1 and PMGC2', 'PMGC2 and PMGC3', 'PMGC3 and PMGC4', 'PMGC4 and PMGC5', 'PMGC5 and PMGC6', 'PMGC6 and PMGC7',
              'PMGC7 and PMGC8', 'TGA4 and TGA12', 'TGB28 and TGA4', 'TGB20 and TGB28', 'TGB12 and TGB20', 'TGA4 and TGA12',
              'TGB4 and TGB12', 'TGB28 and TGB29']

    # t_name = ['TGA4 and TGA12', 'TGB28 and TGA4', 'TGB20 and TGB28', 'TGB12 and TGB20', 'TGA4 and TGA12',
    #          'TGB4 and TGB12', 'TGB28 and TGB29']

    for t, c, n in zip(all_t[8:], s_chans[8:], t_name[8:]):
        me = np.ndarray((total, task.shape[1], 0))
        one = np.ndarray((total, 0))
        bins = []
        lab, cent, sep = find_spikes(t, c, top=4000, bottom=-4000, b=1000)
        indices = [int(x) for x in cent]
        indices.sort()
        mx = [x-50+np.argmax(t[x-50:x+50,c],0) for x in indices]
        # ind = indices[v[0]:v[1]]
        for i in mx:
            start = i-pre
            finish = i+length
            bins.append(t[start:finish, :])
            me = np.append(me, np.expand_dims(bins[-1], 2), axis=2)
            one = np.append(one, np.expand_dims(bins[-1][:, 72], 1), axis=1)
            #eegscanner.eegscanner(bins[-1], list(range(72, 80)), vs=1000, tick=100, ylim=10000, show=True)
        meme.append(np.mean(me, 2))
        EEGScanner(one, list(range(0, len(bins))), vs=total, tick=100, ylim=1000, show=True)
        #eegscanner.eegscanner(bins[-1], list(range(72, 80)), vs=total, tick=100, ylim=1000, show=True)
        #eegscanner.eegscanner(meme[-1], list(range(72, 80)), vs=total, tick=100, ylim=1000, show=True)

        ccep = EEGViewer(meme[-1][:, inx], 4, 8, vs=total, tick=int(total / 10), fs=fs, scale=1000, center=pre, c_num=cn1, show=True)
        ccep.fig.suptitle(n+': Anterior Grid', fontsize=16)
        ccep.fig.text(0.5, 0.0015, 'Time [ms]')
        ccep.fig.text(0.0015, 0.5, 'Microvolts', rotation='vertical')

        ccep2 = EEGViewer(meme[-1][:, inx2], 4, 8, vs=total, tick=int(total / 10), fs=fs, scale=1000, center=pre, c_num=cn2, show=True)
        ccep2.fig.suptitle(n+': Posterior Grid', fontsize=16)
        ccep2.fig.text(0.5, 0.0015, 'Time [ms]')
        ccep2.fig.text(0.0015, 0.5, 'Microvolts', rotation='vertical')

        dep1 = EEGVeiwSingle(meme[-1], list(range(72, 82)), vs=total, tick=int(total / 10), fs=fs, scale=1000, center=pre, show=True)
        dep1.fig.suptitle(n+': Heschl Gyrus Depth', fontsize=16)
        dep1.fig.text(0.5, 0.0015, 'Time [ms]')
        dep1.fig.text(0.0015, 0.5, 'Microvolts', rotation='vertical')

        pmg1 = EEGVeiwSingle(meme[-1], list(range(64, 72)), vs=total, tick=int(total / 10), fs=fs, scale=1000, center=pre, show=True)
        pmg1.fig.suptitle(n + ': Polymicrogyria Depth', fontsize=16)
        pmg1.fig.text(0.5, 0.0015, 'Time [ms]')
        pmg1.fig.text(0.0015, 0.5, 'Microvolts', rotation='vertical')

        amg1 = EEGVeiwSingle(meme[-1], list(range(82, 92)), vs=total, tick=int(total / 10), fs=fs, scale=1000, center=pre, show=True)
        amg1.fig.suptitle(n + ': Amygdala Depth', fontsize=16)
        amg1.fig.text(0.5, 0.0015, 'Time [ms]')
        amg1.fig.text(0.0015, 0.5, 'Microvolts', rotation='vertical')

        hip1 = EEGVeiwSingle(meme[-1], list(range(92, 102)), vs=total, tick=int(total / 10), fs=fs, scale=1000, center=pre, show=True)
        hip1.fig.suptitle(n + ': Hippocampus Depth', fontsize=16)
        hip1.fig.text(0.5, 0.0015, 'Time [ms]')
        hip1.fig.text(0.0015, 0.5, 'Microvolts', rotation='vertical')

        tps1 = EEGVeiwSingle(meme[-1], list(range(102, 109)), vs=total, tick=int(total / 10), fs=fs, scale=1000, center=pre, show=True)
        tps1.fig.suptitle(n + ': Temporal Strip', fontsize=16)
        tps1.fig.text(0.5, 0.0015, 'Time [ms]')
        tps1.fig.text(0.0015, 0.5, 'Microvolts', rotation='vertical')

        sts1 = EEGVeiwSingle(meme[-1], list(range(109, 115)), vs=total, tick=int(total / 10), fs=fs, scale=1000, center=pre, show=True)
        sts1.fig.suptitle(n + ': Subtemporal Strip', fontsize=16)
        sts1.fig.text(0.5, 0.0015, 'Time [ms]')
        sts1.fig.text(0.0015, 0.5, 'Microvolts', rotation='vertical')

    print(cent)
    print(f)

