#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" axis.py
An HDF5 Dataset subclass whose prupose is to be an Axis.
"""
# Package Header #
from ...__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache_method
import numpy as np

# Local Packages #
from ...hdf5objects import HDF5Map, HDF5Dataset


# Definitions #
# Classes #
class AxisMap(HDF5Map):
    ...


class Axis(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Magic Methods #
    # Construction/Destruction
    def __init__(self, start=None, stop=None, step=None, rate: float = None, size: int = None,
                 s_name: str = None, build: bool = None, init: bool = True, **kwargs):
        super().__init__(init=False)
        self.default_kwargs = None  # {"dtype": 'i', "maxshape": (None,)}
        # self._scale_name = None  # Set this to the name of the axis

        if init:
            self.construct(start=start, stop=stop, step=step, rate=rate, size=size,
                           s_name=s_name, build=build, **kwargs)

    @property
    def start(self):
        try:
            return self.get_start.caching_call()
        except AttributeError:
            return self.get_start()

    @property
    def end(self):
        try:
            return self.get_start.caching_call()
        except AttributeError:
            return self.get_start()

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, start: int = None, stop: int = None, step: int = None, rate: float = None, size: int = None,
                  s_name: str = None, build: bool = None, init: bool = True, **kwargs):
        if "data" in kwargs:
            kwargs["build"] = build
            build = False

        super().construct(**kwargs)

        if s_name is not None:
            self._scale_name = s_name

        if (build or build is None) and start is not None:
            self.from_range(start, stop, step, rate, size)

    def from_range(self, start: int = None, stop: int = None, step: int = 1, rate: float = None, size: int = None,
                   **kwargs):
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)

        if step is None and rate is not None:
            step = 1 / rate

        if start is None:
            start = stop - step * size

        if stop is None:
            stop = start + step * size

        if size is not None:
            self.set_data(data=np.linspace(start, stop, size), **d_kwargs)
        else:
            self.set_data(data=np.arange(start, stop, step), **d_kwargs)

    # File
    def refresh(self):
        super().refresh()
        self.get_start.clear_cache()
        self.get_end.clear_cache()

    # Getters/Setters
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_start(self):
        with self:
            return self._dataset[0]

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_end(self):
        with self:
            return self._dataset[-1]

    def get_intervals(self, start=None, stop=None, step=None):
        return np.ediff1d(self.all_data[slice(start, stop, step)])

    # Find
    def find_index(self, index, aprox=False, tails=False):
        samples = self.shape[0]
        if index < self.start:
            if tails:
                return 0, self.start
        elif index > self.end:
            if tails:
                return samples, self.end
        else:
            index = int(np.searchsorted(self.all_data, index, side="right") - 1)
            if aprox or index == self.all_data[index]:
                return index, self.all_data[index]

        return index

    def get_range(self, start=None, stop=None, step=None, aprox=False, tails=False):
        start_sample, true_start = self.find_index(index=start, aprox=aprox, tails=tails)
        stop_sample, true_stop = self.find_index(index=stop, aprox=aprox, tails=tails)

        return self.all_data[slice(start=start_sample, stop=stop_sample, step=step)], true_start, true_stop

    # Manipulation
    def shift(self, shift, start=None, stop=None, step=None):
        with self:
            self._dataset[start:stop:step] += shift
        self.refresh()


# Assign Cyclic Definitions
AxisMap.default_type = Axis
Axis.default_map = AxisMap()

