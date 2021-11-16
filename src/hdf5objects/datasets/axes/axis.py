#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" axis.py
An HDF5 Dataset subclass whose prupose is to be an Axis.
"""
# Package Header #
from ..__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #

# Third-Party Packages #
import h5py
import numpy as np

# Local Packages #
from ..hdf5objects import HDF5Map, HDF5Dataset


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
                 data=None, s_name: str = None, build: bool = True, init: bool = True, **kwargs):
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
                  s_name: str = None, build: bool = True, init: bool = True, **kwargs):
        if "data" in kwargs:
            kwargs["build"] = build
            build = False

        super().construct(**kwargs)

        if s_name is not None:
            self.scale_name = s_name

        if build:
            self.from_range(start, stop, step, rate, size)

    # Getters/Setters
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_start(self):
        with self:
            return self._dataset[0]

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_end(self):
        with self:
            return self._dataset[-1]

    # Modification
    def require(self, name=None, **kwargs):
        super().require(name=name, **kwargs)
        self.make_scale()
        return self

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
            self.require(data=np.linspace(start, stop, size), **d_kwargs)
        else:
            self.require(data=np.arange(start, stop, step), **d_kwargs)


# Assign Cyclic Definitions
AxisMap.default_type = SampleAxis
Axis.default_map = SampleAxisMap()

