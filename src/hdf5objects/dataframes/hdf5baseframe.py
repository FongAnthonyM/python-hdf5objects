#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5baseframe.py
Description:
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
from abc import abstractmethod
from functools import singledispatchmethod
import pathlib
from typing import Union

# Third-Party Packages #
from framestructure import FileTimeFrame
import numpy as np

# Local Packages #
from ..fileobjects import BaseHDF5


# Definitions #
# Classes #
class HDF5BaseFrame(FileTimeFrame):
    file_type = BaseHDF5
    default_data_container = None

    # Class Methods #
    @classmethod
    def validate_path(cls, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if path.is_file():
            return cls.file_type.validate_file_type(path)
        else:
            return False

    @classmethod
    def new_validated(cls, path, mode="r+", **kwargs):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if path.is_file():
            file = cls.file_type.new_validated(path, mode=mode)
            if file:
                return cls(file=file, **kwargs)

        return None

    # Instance Methods #
    # File
    def set_file(self, file, **kwargs):
        if isinstance(file, (str, pathlib.Path)):
            self.file = self.file_type(file=file, **kwargs)
        if isinstance(file, self.file_type):
            self.file = file
        else:
            raise ValueError("file must be a path, File, or HDF5File")

    def load_data(self):
        self.file.data.get_all_data()

    # Setters/Getters
    def get_shape(self):
        return self.file.data.shape

    def get_start(self):
        return self.file.start_datetime

    def get_end(self):
        return self.file.end_datetime

    def get_time_axis(self):
        return self.file.time_axis.all_data

    def get_sample_rate(self):
        return self.file.data.sample_rate

    def get_data(self):
        return self.file.data

    def set_data(self, value):
        if self.mode == 'r':
            raise IOError("not writable")
        self.data.replace_data(value)

    def set_time_axis(self, value):
        if self.mode == 'r':
            raise IOError("not writable")
        self.time_axis.replace_data(value)

    # Shape
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

        new_ndarray = self.blank_generator(shape, dtype, **kwargs)
        new_ndarray[tuple(new_slices)] = self.data[tuple(old_slices)]

        self.data.replace_data(new_ndarray)

    # Data
    def append(self, data, time_axis=None, axis=None, tolerance=None, correction=None, **kwargs):
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        if correction is None or (isinstance(correction, bool) and correction):
            correction = self.tail_correction
        elif isinstance(correction, str):
            correction = self.get_correction(correction)

        if correction:
            data, time_axis = correction(data, time_axis, axis=axis, tolerance=tolerance, **kwargs)

        self.data.append(data, axis)
        self.time_axis.append_data(time_axis)

    def add_frames(self, frames, axis=None, truncate=None):
        if self.mode == 'r':
            raise IOError("not writable")

        frames = list(frames)

        if self.data is None:
            frame = frames.pop(0)
            if not frame.validate_sample_rate():
                raise ValueError("the frame's sample rate must be valid")
            self.data.replace_data(frame[...])
            self.time_axis.replace_data(frame.get_time_axis())

        for frame in frames:
            self.append_frame(frame, axis=axis, truncate=truncate)

    def get_intervals(self, start=None, stop=None, step=None):
        return self.file.time_axis.get_intervals(start=start, stop=stop, step=step)

    # Find
    def find_time_index(self, timestamp, aprox=False, tails=False):
        return self.file.time_axis.find_time_index(timestamp=timestamp, aprox=aprox, tails=tails)

    # Get data
    def get_timestamp_range_time(self, start=None, stop=None, step=None, aprox=False, tails=False):
        return self.file.data.get_timstamp_range_time(start=start, stop=stop, step=step, aprox=aprox, tails=tails)

    def get_datetime_range_time(self, start=None, stop=None, step=None, aprox=False, tails=False):
        return self.file.data.get_datetime_range_time(start=start, stop=stop, step=step, aprox=aprox, tails=tails)

    def get_data_range_sample(self, start=None, stop=None, step=None, aprox=False, tails=False):
        return self.file.data.get_data_range_sample(start=start, stop=stop, step=step, aprox=aprox, tails=tails)

    def get_data_range_time(self, start=None, stop=None, step=None, aprox=False, tails=False):
        return self.file.data.get_data_range_time(start=start, stop=stop, step=step, aprox=aprox, tails=tails)

    # Sample Rate
    def resample(self, sample_rate, **kwargs):
        if self.mode == 'r':
            raise IOError("not writable")

        if not self.validate_sample_rate():
            raise ValueError("the data needs to have a uniform sample rate before resampling")

        self.data.replace_data(self.resampler(new_fs=self.sample_rate, **kwargs))
        self.time_axis.replace_data(np.arange(self.time_axis[0], self.time_axis[-1], self.sample_period, dtype="f8"))

    # Time Correction
    def fill_time_correction(self, axis=None, tolerance=None, **kwargs):
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        discontinuities = self.where_discontinuous(tolerance=tolerance)

        if discontinuities:
            offsets = np.empty((0, 2), dtype="i")
            gap_discontinuities = []
            previous_discontinuity = 0
            for discontinuity in discontinuities:
                timestamp = self.time_axis[discontinuity]
                previous = discontinuity - 1
                previous_timestamp = self.time_axis[previous]
                if (timestamp - previous_timestamp) >= (2 * self.sample_period):
                    real = discontinuity - previous_discontinuity
                    blank = round((timestamp - previous_timestamp) * self.sample_rate) - 1
                    offsets = np.append(offsets, [[real, blank]], axis=0)
                    gap_discontinuities.append(discontinuities)
                    previous_discontinuity = discontinuity
            offsets = np.append(offsets, [[self.time_axis - discontinuities[-1], 0]], axis=0)

            new_size = np.sum(offsets)
            new_shape = list(self.data.shape)
            new_shape[axis] = new_size
            old_data = self.data
            old_times = self.time_axis
            self.data.replace_data(self.blank_generator(shape=new_shape, **kwargs))
            self.time_axis.replace_data(np.empty((new_size,), dtype="f8"))
            old_start = 0
            new_start = 0
            for discontinuity, offset in zip(gap_discontinuities, offsets):
                previous = discontinuity - 1
                new_mid = new_start + offset[0]
                new_end = new_mid + offset[1]
                mid_timestamp = old_times[previous] + self.sample_period
                end_timestamp = offset[1] * self.sample_period

                slice_ = slice(start=old_start, stop=old_start + offset[0])
                slices = [slice(None, None)] * len(old_data.shape)
                slices[axis] = slice_

                self.set_range(old_data[tuple(slices)], start=new_start)

                self.time_axis[new_start:new_mid] = old_times[slice_]
                self.time_axis[new_mid:new_end] = np.arange(mid_timestamp, end_timestamp, self.sample_period)

                old_start = discontinuity
                new_start += sum(offset)
