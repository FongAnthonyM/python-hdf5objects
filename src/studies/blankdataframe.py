#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" blankdataframe.py
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
from .dataframeinterface import DataFrameInterface


# Definitions #
# Classes #
class BlankDataFrame(DataFrameInterface):
    # Static Methods #
    @staticmethod
    def create_nan_array(shape=None, **kwargs):
        a = np.empty(shape=shape, **kwargs)
        a.fill(np.nan)
        return a

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, shape=None, dtype=None, init=True):
        # Parent Attributes #
        super().__init__()

        # New Attributes #
        # Descriptors #
        # System
        self.is_cache = True
        self.mode = 'a'

        # Shape
        self._shape = None
        self.axis = 0

        # Data Type
        self.dtype = "f4"

        # Assign Methods #
        self.generate_data = self.create_nan_array

        # Construct Object #
        if init:
            self.construct(shape=shape, dtype=dtype)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    # Container Methods
    def __len__(self):
        return self.get_length()

    def __getitem__(self, item):
        return self.get_item(item)

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, shape=None, dtype=None):
        if shape is not None:
            self._shape = shape

        if dtype is not None:
            self.dtype = dtype

    # Editable Copy Methods
    def default_editable_method(self, **kwargs):
        return self.copy

    # Getters
    def get_length(self):
        return self.shape[self.axis]

    def get_item(self, item):
        if isinstance(item, slice):
            return self.create_data_slice(item)
        elif isinstance(item, (tuple, list)):
            return self.create_data_slice(item)
        elif isinstance(item, ...):
            return self.create_data()

    # Setters
    def set_data_generator(self, obj):
        if isinstance(obj, str):
            obj = obj.lower()
            if obj == "nan":
                self.generate_data = self.create_nan_array
            elif obj == "empty":
                self.generate_data = np.empty
            elif obj == "zeros":
                self.generate_data = np.zeros
            elif obj == "ones":
                self.generate_data = np.ones
            elif obj == "full":
                self.generate_data = np.full
        else:
            self.generate_data = obj

    # Data
    def create_data(self, start=None, stop=None, step=None, dtype=None, **kwargs):
        size = self.get_length()
        shape = self.shape

        if dtype is None:
            dtype = self.dtype

        if start is None:
            start = 0

        if stop is None:
            stop = size
        elif stop < 0:
            stop = size + stop

        if start >= size or stop < 0:
            raise IndexError("index is out of range")

        size = stop - start
        shape[self.axis] = size
        if step is not None:
            slices = [slice(None)] * len(shape)
            slices[self.axis] = slice(None, None, step)

            return self.generate_data(shape=self.shape, dtype=dtype, **kwargs)[tuple(slices)]
        else:
            return self.generate_data(shape=self.shape, dtype=dtype, **kwargs)

    def create_data_slice(self, slices, dtype=None, **kwargs):
        size = self.get_length()
        shape = self.shape

        if slices is None:
            start = None
            stop = None
            step = None
            slices = [slice(None)] * len(shape)
        elif isinstance(slices, slice):
            start = slices.start
            stop = slices.stop
            step = slices.step
            slices = [slice(None)] * len(shape)
        else:
            start = slices[self.axis].start
            stop = slices[self.axis].stop
            step = slices[self.axis].step

        if dtype is None:
            dtype = self.dtype

        if start is None:
            start = 0

        if stop is None:
            stop = size
        elif stop < 0:
            stop = size + stop

        if step is None:
            step = 1

        if start >= size or stop < 0:
            raise IndexError("index is out of range")

        size = stop - start
        shape[self.axis] = size
        slices[self.axis] = slice(None, None, step)

        return self.generate_data(shape=self.shape, dtype=dtype, **kwargs)[tuple(slices)]

    def get_range(self, start=None, stop=None, step=None):
        return self.create_data(start=start, stop=stop, step=step)

    # Shape
    def validate_shape(self):
        return True

    def change_size(self, shape=None, **kwargs):
        self.shape = shape

    # Get Index
    def get_index(self, indices, reverse=False, frame=True):
        if isinstance(indices, int):
            start = indices
        elif len(indices) == 1:
            start = indices[0]
        else:
            raise IndexError("index out of range")

        if frame:
            return self
        else:
            return self.create_data(start=start, stop=start + 1)
