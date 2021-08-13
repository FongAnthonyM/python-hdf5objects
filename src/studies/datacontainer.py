#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" datacontainer.py
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
from contextlib import contextmanager

# Downloaded Libraries #
from baseobjects import StaticWrapper
import numpy as np

# Local Libraries #
from .dataframeinterface import DataFrameInterface


# Definitions #
# Classes #
class DataContainer(DataFrameInterface):  # Todo: Make this a StaticWrapper (StaticWrapper needs to be expanded)
    default_editable_type = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, frames=None, data=None, shape=None, init=True, **kwargs):
        # Parent Attributes #
        super().__init__()

        # Descriptors #
        # System
        self._cache = True
        self.is_cache = True
        self.is_updating = True
        self.mode = 'a'

        # Shape
        self.target_shape = None
        self.is_truncate = False
        self.axis = 0

        # Assign Methods #
        self.editable_method = None
        self.editable_type = self.default_editable_type

        # Containers #
        self.data = None

        # Object Construction #
        if init:
            self.construct(frames=frames, data=data, shape=shape, **kwargs)

    @property
    def shape(self):
        return self.data.shape

    # Instance Methods
    # Constructors/Destructors
    def construct(self, frames=None, data=None, shape=None, **kwargs):
        if shape is not None and self.data is None:
            self.data = np.zeros(shape=shape, **kwargs)

        if data is not None:
            self.data = data

        if frames is not None:
            if isinstance(frames, np.ndarray):
                self.data = frames
            else:
                self.add_frames(frames)

    def editable_copy(self, **kwargs):
        copy_ = self.copy()
        copy_.mode = 'a'
        return copy_

    # Editable Copy Methods
    def default_editable_method(self):
        return self.editable_type()

    # Cache and Memory
    @contextmanager
    def cache(self):
        was_cache = self._cache

        if self.is_cache:
            self._cache = True
        else:
            self._cache = False

        yield self.is_cache

        self._cache = was_cache

    # Getters
    def get_length(self):
        return self.data.shape[self.axis]

    def get_item(self, item):
        return self.data[item]

    # Data
    def append(self, data, axis=None):
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if isinstance(data, np.ndarray):
            self.data = np.append(self.data, data, axis)

    def append_frame(self, frame, axis=None, truncate=None):
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if truncate is None:
            truncate = self.is_truncate

        shape = self.shape
        slices = ...
        if not frame.validate_shape or frame.shape != shape:
            if not truncate:
                raise ValueError("the frame's shape does not match this object's.")
            else:
                slices = [None] * len(shape)
                for index, size in enumerate(shape):
                    slices[index] = slice(None, size)
                slices[axis] = slice(None, None)
                slices = tuple(slices)

        self.data = np.append(self.data, frame[slices], axis)

    def add_frames(self, frames, axis=None, truncate=None):
        if self.mode == 'r':
            raise IOError("not writable")

        frames = list(frames)

        if self.data is None:
            self.data = frames.pop(0)[...]

        for frame in frames:
            self.append_frame(frame, axis=axis, truncate=truncate)

    def get_range(self, start=None, stop=None, step=None, axis=None):
        if axis is None:
            axis = self.axis

        slices = [slice(None, None)] * len(self.shape)
        slices[axis] = slice(start=start, stop=stop, step=step)

        return self.data[tuple(slices)]

    def set_range(self, data, start=None, stop=None, step=None, axis=None):
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if start is None:
            start = 0

        if stop is None:
            stop = start + data.shape[axis]

        slices = [0] * len(self.shape)
        for index, ax in enumerate(data.shape):
            slices[index] = slice(None, None)
        slices[axis] = slice(start=start, stop=stop, step=step)

        self.data[tuple(slices)] = data

    # Shape
    def validate_shape(self):
        return True

    def change_size(self, shape=None, dtype=None, **kwargs):
        if self.mode == 'r':
            raise IOError("not writable")

        if shape is None:
            shape = self.target_shape

        if dtype is None:
            dtype = self.data.dtype

        new_slices = [0] * len(shape)
        old_slices = [0] * len(self.shape)
        for index, (n, o) in enumerate(zip(shape, self.shape)):
            slice_ = slice(None, n if n > o else o)
            new_slices[index] = slice_
            old_slices[index] = slice_

        new_ndarray = np.zeros(shape, dtype, **kwargs)
        new_ndarray[tuple(new_slices)] = self.data[tuple(old_slices)]

        self.data = new_ndarray

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
            return self.get_range(start=start, stop=start + 1)


# Assign Cyclic Definitions
DataContainer.default_editable_type = DataContainer
