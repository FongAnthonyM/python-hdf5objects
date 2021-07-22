#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" dataframe.py
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
from abc import abstractmethod
from contextlib import contextmanager

# Downloaded Libraries #
from baseobjects import BaseObject
import numpy as np

# Local Libraries #


# Definitions #
# Classes #
class DataFrame(BaseObject):
    default_return_frame_type = None
    default_combine_type = None

    # Magic Methods
    # Construction/Destruction
    def __init__(self, frames=None, update=True, init=True):
        self._cache = False
        self._shapes = None
        self._shape = None
        self._lengths = None
        self._length = None
        self.target_shape = None

        self.is_cache = True
        self.is_updating = True
        self.returns_frame = False
        self.axis = 0

        self.combine_type = self.default_combine_type
        self.return_frame_type = self.default_return_frame_type
        self.frames = []

        if init:
            self.construct(frames, update)

    @property
    def shapes(self):
        if self._shapes is None or (self.is_updating and not self._cache):
            return self.get_shapes()
        else:
            return self._shapes

    @property
    def shape(self):
        if self._shape is None or (self.is_updating and not self._cache):
            return self.get_shape()
        else:
            return self._shape

    @property
    def lengths(self):
        if self._lengths is None or (self.is_updating and not self._cache):
            return self.get_lengths()
        else:
            return self._lengths

    @property
    def length(self):
        if self._length is None or (self.is_updating and not self._cache):
            return self.get_length()
        else:
            return self._length

    # Container Methods
    def __len__(self):
        return self.get_length()

    def __getitem__(self, item):
        return self.get_item(item)

    # Arithmetic
    def __add__(self, other):
        if isinstance(other, DataFrame):
            return type(self)(self.frames + other.frames, self.is_updating)
        else:
            return type(self)(self.frames + other, self.is_updating)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, frames=None, update=None):
        if frames is not None:
            self.frames = frames

        if update is not None:
            self.is_updating = update

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

    def refresh(self):
        self.get_shapes()
        self.get_shape()
        self.get_lengths()
        self.get_length()

    # Getters/Setters
    def get_shapes(self):
        self._shapes = [frame.shape for frame in self.frames]
        return self._shapes

    def get_shape(self):
        shape_array = np.array(self.get_shapes())
        self._shape = [None] * shape_array.shape[0]
        for ax in range(shape_array.shape[0]):
            if ax == self.axis:
                self._shape[ax] = sum(shape_array[:, ax])
            else:
                self._shape[ax] = min(shape_array[:, ax])

        return self._shape

    def get_lengths(self):
        shape_array = np.array(self.get_shapes())
        self._lengths = list(shape_array[:, self.axis])
        return self._lengths

    def get_length(self):
        shape_array = np.array(self.get_shapes())
        self._length = sum(shape_array[:, self.axis])
        return self._length

    def get_item(self, item):
        if isinstance(item, slice):
            return self.get_range_slice(item)
        elif isinstance(item, (tuple, list)):
            is_slices = True
            for element in item:
                if isinstance(element, int):
                    is_slices = False
                    break
            if is_slices:
                return self.get_ranges(item)
            else:
                return self.get_frame_within(item)
        elif isinstance(item, int):
            return self.get_frame(item)
        elif isinstance(item, ...):
            return self.get_all_data()

    # Shape
    def validate_shape(self):
        shapes = self.get_shapes()
        if shapes:
            shape = shapes.pop()
            for s in shapes:
                if s != shape:
                    return False
        return True

    def reshape(self, shape=None, **kwargs):
        if shape is None:
            shape = self.target_shape

        for frame in self.frames:
            if not frame.validate_shape() or frame.shape != shape:
                frame.reshape(shape, **kwargs)

    # Frames
    def frame_sort_key(self, frame):
        return frame

    def sort_frames(self, key=None, reverse=False):
        if key is None:
            key = self.frame_sort_key
        self.frames.sort(key=key, reverse=reverse)

    # Container
    def append(self, item):
        self.frames.append(item)

    # General
    def smart_append(self, a, b, axis=None):
        if axis is None:
            axis = self.axis

        if isinstance(a, np.ndarray):
            return np.append(a, b, axis)
        else:
            return a + b

    # Find within Frames
    def find_frame_index(self, super_index):
        with self.cache():
            # Check if index is in range.
            if super_index >= self.length or (super_index + self._length) < 0:
                raise IndexError("index is out of range")

            # Change negative indexing into positive.
            if super_index < 0:
                super_index = self.length - super_index

            # Find
            previous = 0
            for frame_index, frame_length in enumerate(self.lengths):
                end = previous + frame_length
                if super_index < end:
                    return frame_index, super_index - previous, previous
                else:
                    previous = end

    def find_frame_indices(self, super_indices):
        with self.cache():
            super_indices = list(super_indices)
            for index, super_index in enumerate(super_indices):
                # Check if index is in range.
                if super_index >= self.length or (super_index + self._length) < 0:
                    raise IndexError("index is out of range")

                # Change negative indexing into positive.
                if super_index < 0:
                    super_indices[index] = self.length - super_index

            # Find
            indices = [None] * len(super_indices)
            previous = 0
            for frame_index, frame_length in enumerate(self.lengths):
                end = previous + frame_length
                for index, super_index in enumerate(super_indices):
                    if previous <= super_index < end:
                        indices[index] = [frame_index, super_index - previous, previous]
                previous = end

            return indices

    # Get a Range of Frames
    def get_range(self, start=None, stop=None, step=None, frame=None):
        if start is not None and stop is not None:
            start_index, stop_index = self.find_frame_indices([start, stop])
        elif start is not None:
            start_index = self.find_frame_index(start)
            stop_index = [None, None, None]
        elif stop is not None:
            stop_index = self.find_frame_index(stop)
            start_index = [None, None, None]
        else:
            start_index = [None, None, None]
            stop_index = [None, None, None]

        start_frame, start_inner, _ = start_index
        stop_frame, stop_inner, _ = stop_index

        return self.get_range_frame(start_frame, stop_frame, start_inner, stop_inner, step, frame)

    def get_range_frame(self, frame_start=None, frame_stop=None, inner_start=None, inner_stop=None, step=None,
                        frame=None):
        if (frame is None and self.returns_frame) or frame:
            data = self.return_frame_type(frames=[self.frames[frame_start:frame_stop]])
        else:
            if frame_start is None:
                frame_start = 0
            if frame_stop is None:
                frame_stop = -1

            if (frame_start + 1) == frame_stop or (frame_start + 1) == len(self.frames) + frame_stop:
                data = self.frames[frame_start][inner_start:inner_stop:step]
            else:
                data = self.frames[frame_start][inner_start::step]
                for fi in range(frame_start + 1, frame_stop):
                    data = self.smart_append(data, self.frames[fi][::step])
                data = self.smart_append(data, self.frames[frame_stop][:inner_stop:step])
        return data

    # Get a Range of Frames with a Slice
    def get_range_slice(self, item, frame=None):
        return self.get_range(item.start, item.stop, item.step, frame)

    # Get a Range of Frames with Slices
    def get_ranges(self, slices, axis=None, frame=None):
        if axis is None:
            axis = self.axis

        slices = list(slices)
        slice_ = slices[axis]
        start = slice_.start
        stop = slice_.stop
        step = slice_.step

        if start is not None and stop is not None:
            start_index, stop_index = self.find_frame_indices([start, stop])
        elif start is not None:
            start_index = self.find_frame_index(start)
            stop_index = [None, None, None]
        elif stop is not None:
            stop_index = self.find_frame_index(stop)
            start_index = [None, None, None]
        else:
            start_index = [None, None, None]
            stop_index = [None, None, None]

        start_frame, start_inner, _ = start_index
        stop_frame, stop_inner, _ = stop_index

        slices[axis] = slice(start_inner, stop_inner, step)

        return self.get_ranges_frame(start_frame, stop_frame, slices, axis, frame)

    def get_ranges_frame(self, frame_start=None, frame_stop=None, slices=None, axis=None, frame=None):
        if (frame is None and self.returns_frame) or frame:
            data = self.return_frame_type(frames=[self.frames[frame_start:frame_stop]])
        else:
            if axis is None:
                axis = self.axis
            if frame_start is None:
                frame_start = 0
            if frame_stop is None:
                frame_stop = -1

            if frame_start == frame_stop:
                data = self.frames[frame_start][slices]
            else:
                slice_ = slices[axis]

                first = list(slices)
                inner = list(slices)
                last = list(slices)

                first[axis] = slice(slice_.start, None, slice_.step)
                inner[axis] = slice(None, None, slice_.step)
                last[axis] = slice(None, slice_.stop, slice_.step)

                data = self.frames[frame_start][tuple(first)]
                for fi in range(frame_start + 1, frame_stop):
                    data = self.smart_append(data, self.frames[fi][tuple(inner)])
                data = self.smart_append(data, self.frames[frame_stop][tuple(last)])
        return data

    # Get Frame based on Index
    def get_frame(self, index, frame=None):
        if (frame is None and self.returns_frame) or frame:
            return self.frames[index]
        else:
            return self.frames[index][...]

    def get_all_data(self, frame=None):
        if (frame is None and self.returns_frame) or frame:
            return self
        else:
            data = self.frames[0][...]
            for index in range(1, len(self.frames)):
                data = self.smart_append(data, self.frames[index][...])
            return data

    # Get Frame within by Index
    def get_frame_within(self, indices, reverse=False, frame=True):
        if len(indices) == 1:
            item = indices[0]
            if isinstance(item, int):
                return self.get_frame(item, frame)
            elif isinstance(item, slice):
                return self.get_range_slice(item, frame)
            else:
                return self.get_ranges(item, frame=frame)
        else:
            indices = list(indices)
            if not reverse:
                index = indices.pop(0)
            else:
                index = indices.pop()

            return self.frames[index].get_frame_within(indices, reverse, frame)

    def get_range_indices(self, start=None, stop=None, step=None, reverse=False, frame=None):
        if frame is None:
            frame = self.returns_frame

        if start is None and stop is None:
            if frame:
                return self
            else:
                return self[::step]

        frame_start = None
        frame_stop = None
        if start is not None and len(start) > 1:
            start = list(start)
            if not reverse:
                frame_start = start.pop(0)
            else:
                frame_start = start.pop()

        if stop is not None and len(stop) > 1:
            stop = list(stop)
            if not reverse:
                frame_stop = stop.pop(0)
            else:
                frame_stop = stop.pop()

        if frame_start and frame_start and (frame_start + 1) == frame_stop:
            return self[frame_start].get_range_indices(start, stop, step, reverse, frame)

        if start is None:
            if frame:
                start_data = self.return_frame_type(frames=[self.frames[frame_start:]])
            else:
                start_data = self.frames[0][::step]
                for fi in range(1, frame_stop):
                    start_data = self.smart_append(start_data, self.frames[fi][::step])
        elif len(start) == 1:
            start_data = self.get_frame_within(start, reverse, frame)
        else:
            start_data = self.frames[frame_start].get_range_indices(start, None, step, reverse, frame)

        if stop is None:
            if frame:
                stop_data = self.return_frame_type(frames=[self.frames[:frame_stop]])
            else:
                stop_data = self.smart_append(start_data, self.frames[frame_start + 1][::step])
                for fi in range(frame_start + 2, len(self.frames)):
                    stop_data = self.smart_append(stop_data, self.frames[fi][::step])
        elif len(stop) == 1:
            stop_data = self.get_frame_within(stop, reverse, frame)
        else:
            stop_data = self.frames[frame_stop].get_range_indices(None, stop, step, reverse, frame)

        if start is not None and stop is not None:
            if frame:
                data = self.return_frame_type(frames=[self.frames[frame_start:frame_stop]])
                self.smart_append(start_data, data)
            else:
                for fi in range(frame_start + 2, frame_stop):
                    start_data = self.smart_append(start_data, self.frames[fi][::step])

        return self.smart_append(start_data, stop_data)

    # Combine
    def combine_frames(self, start=None, stop=None, step=None):
        combine = self.combine_type()
        for frame in self.frames[start:stop:step]:
            combine.append_frame(frame)
        return combine


class DataFrameInterface(BaseObject):
    # Container Methods
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    # Shape
    @abstractmethod
    def validate_shape(self):
        pass

    @abstractmethod
    def reshape(self, shape=None, **kwargs):
        pass


# Assign Cyclic Definitions
DataFrame.default_return_frame_type = DataFrame
