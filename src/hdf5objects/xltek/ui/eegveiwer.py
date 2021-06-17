#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" eegviewer.py
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
import pathlib
import datetime

# Downloaded Libraries #
import numpy as np

import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
import matplotlib.widgets as widgets
from matplotlib import style

# Local Libraries #


# Definitions #
# Classes #
class EEGViewer:
    def __init__(self, data, x, y, vs=1000, ylim=2500, tick=100, fs=1, scale=1, center=0, c_num=None, show=False):
        self.data = data
        self.vis_samples = vs
        self.xs = np.linspace(0-center, self.vis_samples-center, self.vis_samples)*scale/fs
        self.n_samples = len(data)
        self.ratio = self.n_samples * .01
        self.ylim = ylim
        self.tick = tick

        self.x_plots = x
        self.y_plots = y

        if c_num is None:
            c_num = np.array(np.transpose(np.arange(x*y - 1, 0 - 1, -1).reshape(x, y))+1, str)

        style.use('fivethirtyeight')
        self.fig, self.axes = pyplot.subplots(self.y_plots, self.x_plots)
        self.lines = []
        #xticks = np.linspace(0-center, self.vis_samples-center, int(self.vis_samples/self.tick), endpoint=False)*scale/fs
        xticks = range(int(0-center*scale/fs)+1, int((self.vis_samples-center)*scale/fs), 5)
        for i in range(0, self.y_plots):
            for j in range(0, self.x_plots):
                self.axes[i, j].set_title(('Channel ' + c_num[i, j]), fontsize=10)
                self.axes[i, j].set_xlim((0-center)*scale/fs, (self.vis_samples-center)*scale/fs)
                self.axes[i, j].set_xticks(xticks)
                #self.axes[i, j].set_xticklabels(["%d" % (x*scale) for x in xticks])
                self.axes[i, j].xaxis.set_tick_params(labelsize=5)
                self.axes[i, j].set_ylim(-self.ylim, self.ylim)
                self.axes[i, j].yaxis.set_tick_params(labelsize=5)
                self.lines += self.axes[i, j].plot(self.xs, self.data[:, i*self.x_plots+j], '-', linewidth=1)
        for i in range(0, self.y_plots - 1):
            pyplot.setp([a.get_xticklabels() for a in self.axes[i, :]], visible=False)
        for i in range(1, self.x_plots):
            pyplot.setp([a.get_yticklabels() for a in self.axes[:, i]], visible=False)

        pyplot.subplots_adjust(left=0.03, bottom=0.035, right=0.975, top=0.925, wspace=0.1, hspace=0.1)
        if show:
            pyplot.show(block=False)


class EEGVeiwSingle:
    def __init__(self, data, c_range, vs=1000, tick=100, ylim=2500, fs=1, scale=1, center=0, show=False):
        self.data = data
        self.vis_samples = vs
        self.xs = np.linspace(0-center, self.vis_samples-center, self.vis_samples)*scale/fs
        self.n_samples = len(data)
        self.ratio = self.n_samples * .01
        self.tick = tick
        self.ylim = ylim

        style.use('fivethirtyeight')

        self.c_range = c_range
        self.n_channels = len(c_range)
        self.fig, self.axes = pyplot.subplots(self.n_channels, 1)

        self.lines = []
        #xticks = np.linspace(0, self.vis_samples / fs, int(self.vis_samples / self.tick), endpoint=False)
        xticks = range(int(0 - center * scale / fs) + 1, int((self.vis_samples - center) * scale / fs), 5)
        frame = self.data[:self.vis_samples, :]
        for i in range(0, self.n_channels):
            self.axes[i].set_title('Channel %d' % (i + 1), fontsize=10)
            self.axes[i].set_xlim((0-center)*scale/fs, (self.vis_samples-center)*scale/fs)
            self.axes[i].set_xticks(xticks)
            #self.axes[i].set_xticklabels(["%d" % (x * scale) for x in xticks])
            self.axes[i].xaxis.set_tick_params(labelsize=5)
            self.axes[i].set_ylim(-self.ylim, self.ylim)
            self.axes[i].yaxis.set_tick_params(labelsize=5)
            self.lines += self.axes[i].plot(self.xs, frame[:, self.c_range[i]], '-', linewidth=1, animated=False)
        for i in range(0, self.n_channels - 1):
            pyplot.setp(self.axes[i].get_xticklabels(), visible=False)


        pyplot.subplots_adjust(left=0.03, bottom=0.035, right=0.975, top=0.925, wspace=0.1, hspace=0.1)
        if show:
            pyplot.show(block=False)


if __name__ == "__main__":
    pass

