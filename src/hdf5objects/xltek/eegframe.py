#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" eegframe.py
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

# Downloaded Libraries #
import numpy as np

# Local Libraries #


# Definitions #
# Functions #
def smart_append(a, b, axis=0):
    if isinstance(a, np.ndarray):
        return np.append(a, b, axis)
    else:
        return a.append(b)


# Classes #
class EEGFrame:
    def __init__(self, frames=[], update=True):
        self.frames = frames
        self._lengths = [len(frame) for frame in frames]
        self._shape = None
        self.is_updating = update

    def __len__(self):
        return int(sum([float(x) for x in self.lengths]))

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start is None:
                start = 0
            else:
                start = item.start
            if item.stop is None:
                stop = -1
            else:
                stop = item.stop
            rein = self.find_indices([start, stop])
            data = self.get_data(rein[0][0], rein[1][0], rein[0][1], rein[1][1], item.step)
        elif isinstance(item, tuple):
            if item[0].start is None:
                start = 0
            else:
                start = item[0].start
            if item[0].stop is None:
                stop = -1
            else:
                stop = item[0].stop
            rein = self.find_indices([start, stop])
            data = self.get_multidata(rein[0][0], rein[1][0], rein[0][1], rein[1][1], item[0].step, item[1:])
        return data

    # Arithmetic
    def __add__(self, other):
        if isinstance(other, EEGFrame):
            return type(self)(self.frames + other.frames, self.is_updating)
        else:
            return type(self)(self.frames + other, self.is_updating)

    @property
    def lengths(self):
        if self.is_updating or self._lengths is None:
            self._lengths = [len(frame) for frame in self.frames]
        return self._lengths

    @property
    def shape(self):
        if self.is_updating or self._shape is None:
            self._shape = [len(frame) for frame in self.frames]
        return self._shape

    def find_index(self, item):
        difference = 0
        for i, f_length in enumerate(self.lengths):
            check = difference + f_length
            if item < check:
                return i, item - difference, difference
            else:
                difference = check
        return None, None, None

    def find_indices(self, items):
        indices = []
        difference = 0
        for i, f_length in enumerate(self.lengths):
            check = difference + f_length
            for item in items:
                if difference <= item < check:
                    indices.append([i, item - difference, difference])
            difference = check
        for item in items:
            if item == -1:
                indices.append([difference-1, self.lengths[-1]-1, difference-self.lengths[-1]])
        return indices

    def data(self, item):
        if isinstance(item, slice):
            rein = self.find_indices([item.start, item.stop])
            data = self.get_data(rein[0][0], rein[1][0], rein[0][1], rein[1][1], item.step)
        elif isinstance(item, tuple):
            rein = self.find_indices([item[0].start, item[0].stop])
            data = self.get_multidata(rein[0][0], rein[1][0], rein[0][1], rein[1][1], item[0].step, item[1:])
        return data

    def append(self, item):
        self.frames.append(item)

    def get_data(self, fstart, fstop, start, stop, step=None):
        if fstart == fstop:
            data = self.frames[fstart][start:stop:step]
        else:
            data = self.frames[fstart][start::step]
            for fi in range(fstart+1, fstop):
                data = smart_append(data, self.frames[fi][::step])
            data = smart_append(data, self.frames[fstop][:stop:step])
        return data

    def get_multidata(self, fstart, fstop, start, stop, step=None, slices=tuple()):
        if fstart == fstop:
            slicer = (slice(start, stop, step),) + slices
            data = self.frames[fstart][slicer]
        else:
            slicer = (slice(start, None, step),) + slices
            data = self.frames[fstart][slicer]
            slicer = (slice(None, None, step),) + slices
            for fi in range(fstart+1, fstop):
                data = smart_append(data, self.frames[fi][slicer])
            slicer = (slice(None, stop, step),) + slices
            data = smart_append(data, self.frames[fstop][slicer])
        return data
