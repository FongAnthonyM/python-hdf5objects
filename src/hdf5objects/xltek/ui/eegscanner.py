#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" eegscanner.py
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
class EEGScanner:
    def __init__(self, data, c_range, vs=320000, tick=4000, ylim=10000, show=False):
        self.data = data
        self.vis_samples = vs
        self.xs = range(0, self.vis_samples)
        self.n_samples = len(data)
        self.ratio = self.n_samples * .01
        self.tick = tick
        self.ylim = ylim

        style.use('fivethirtyeight')

        self.c_range = c_range
        self.n_channels = len(c_range)
        self.fig, self.axes = pyplot.subplots(self.n_channels, 1)

        axcolor = 'lightgoldenrodyellow'
        self.s_ax = pyplot.axes([0.1, 0, 0.8, 0.03], facecolor=axcolor)
        self.s_time = widgets.Slider(self.s_ax, 'Time', 0, 100, valinit=0)
        self.lines = []
        xticks = range(0, self.vis_samples, self.tick)
        frame = self.data[:self.vis_samples, :]
        for i in range(0, self.n_channels):
            #self.axes[i].set_title('Channel %d' % (i + 1), fontsize=5)
            self.axes[i].set_xlim(0, self.vis_samples)
            self.axes[i].set_xticks(xticks)
            self.axes[i].set_xticklabels(["%d" % x for x in xticks])
            self.axes[i].xaxis.set_tick_params(labelsize=5)
            self.axes[i].set_ylim(-self.ylim, self.ylim)
            #yticks = [-1000, -750, -500, -250, 0, 250, 500, 750, 1000]
            #self.axes[i].set_yticks(yticks)
            #self.axes[i].set_yticklabels(["%d" % y for y in yticks])
            self.axes[i].yaxis.set_tick_params(labelsize=5)
            #self.axes[i].grid(False, 'major')
            self.lines += self.axes[i].plot(self.xs, frame[:, self.c_range[i]], '-', linewidth=0.25, animated=False)
        #for i in range(0, self.n_channels - 1):
            #pyplot.setp(self.axes[i].get_xticklabels(), visible=False)

        self.s_time.on_changed(self.update)

        pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        if show:
            pyplot.show(block=False)

    def update(self, val):
        first = int(self.ratio * self.s_time.val)
        last = int(self.ratio * self.s_time.val) + self.vis_samples
        if last > self.n_samples:
            first = -self.vis_samples-1
            last = -1
        frame = self.data[first:last, :]
        xticks = range(0, last-first, self.tick)
        for i in range(0, self.n_channels):
            self.axes[i].set_xticks(xticks)
            self.axes[i].set_xticklabels(["%d" % (x+first) for x in xticks])
            self.lines[i].set_data(self.xs, frame[:, self.c_range[i]])
        return self.lines

    def save(self):
        Writer = animation.writers['ffmpeg']
        self.writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        self.animator = animation.FuncAnimation(self.fig, self.update, interval=50, blit=True)
        self.animator.save('lines.mp4', writer=self.writer)
        pyplot.show()

