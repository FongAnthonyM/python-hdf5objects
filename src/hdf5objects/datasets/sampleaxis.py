#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" sampleaxis.py
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
import h5py
import numpy as np

# Local Libraries #
from ..hdf5object import HDF5Map, HDF5Dataset


# Definitions #
# Classes #
class SampleAxisMap(HDF5Map):
    ...


class SampleAxis(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_map = SampleAxisMap()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, obj=None, start=None, stop=None, step=None, rate=None, size=None,
                 create=True, init=True, **kwargs):
        super().__init__(init=False)
        self.default_kwargs = {"dtype": 'i', "maxshape": (None,)}
        self.label = "samples"

        if init:
            self.construct(obj, start, stop, step, rate, size, create, **kwargs)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, obj=None, stop=None, step=None, rate=None, size=None, create=True, start=None, **kwargs):
        super().construct(**kwargs)
        if isinstance(obj, (h5py.Dataset, HDF5Dataset)):
            self.set_dataset(obj)
        elif create:
            if obj is None:
                self.from_range(start, stop, step, rate, size, **kwargs)
            elif isinstance(obj, (int, float)):
                self.from_range(obj, stop, step, rate, size, **kwargs)
            elif isinstance(obj, h5py.Dataset):
                self._dataset = obj
            elif isinstance(obj, HDF5Dataset):
                self._dataset = obj._dataset
            else:
                self.require(data=obj, **kwargs)

    def from_range(self, start=None, stop=None, step=1, rate=None, size=None, **kwargs):
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

        with self:
            self._dataset.make_scale(self.label)


# Assign Cyclic Definitions
SampleAxisMap.default_type = SampleAxis
SampleAxis.default_map = SampleAxisMap()
