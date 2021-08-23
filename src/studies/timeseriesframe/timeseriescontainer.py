#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" timeseriescontainer.py
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
import datetime
import math

# Downloaded Libraries #
import numpy as np
from scipy import interpolate
from dspobjects import Resample

# Local Libraries #
from ..datacontainer import DataContainer
from .timeseriesframeinterface import TimeSeriesFrameInterface


# Todo: Make an interpolator object
# Todo: Make a time_axis object
# Todo: Make implement data mapping to reduce memory
# Definitions #
# Classes #
class TimeSeriesContainer(DataContainer, TimeSeriesFrameInterface):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Static Methods #
    @staticmethod
    def create_nan_array(shape=None, **kwargs):
        a = np.empty(shape=shape, **kwargs)
        a.fill(np.nan)
        return a

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, frames=None, data=None, time_axis=True, shape=None, sample_rate=None, init=True, **kwargs):
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        # Descriptors #
        # System
        self.switch_algorithm_size = 10000000  # Consider chunking rather than switching

        # Time
        self._date = None
        self.time_tolerance = 0.000001
        self.sample_rate = None
        self._sample_period = None
        self._is_continuous = None

        # Interpolate
        self.interpolate_type = "linear"
        self.interpolate_fill_value = "extrapolate"

        # Object Assignment #
        self.resampler = None

        # Method Assignment #
        self.blank_generator = self.create_nan_array
        self.tail_correction = self.default_tail_correction

        # Containers #
        self.time_axis = None

        # Object Construction #
        if init:
            self.construct(frames=frames, data=data, time_axis=time_axis, shape=shape, sample_rate=sample_rate, **kwargs)

    @property
    def n_samples(self):
        return self.get_length()

    @property
    def start(self):
        return datetime.datetime.fromtimestamp(self.time_axis[0])

    @property
    def end(self):
        return datetime.datetime.fromtimestamp(self.time_axis[-1])

    @property
    def date(self):
        if self.start is None:
            return self._date
        else:
            self.start.date()

    @property
    def sample_period(self):
        if self._sample_period is None or (self.is_updating and not self._cache):
            return self.get_sample_period()
        else:
            return self._sample_period

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, frames=None, data=None, time_axis=None, shape=None, sample_rate=None, **kwargs):
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if time_axis is not None:
            self.time_axis = time_axis

        super().construct(frames=frames, data=data, shape=shape, **kwargs)

        self.resampler = Resample(old_fs=self.sample_rate, axis=self.axis)

    # Getters
    def get_time_axis(self):
        return self.time_axis

    def get_sample_period(self):
        self._sample_period = 1 / self.sample_rate
        return self._sample_period

    def get_correction(self, name):
        name.lower()
        if name == "none":
            return None
        elif name == "tail":
            return self.tail_correction
        elif name == "default tail":
            return self.default_tail_correction
        elif name == "nearest end":
            return self.shift_to_nearest_sample_end
        elif name == "end":
            return self.shift_to_the_end

    def set_tail_correction(self, obj):
        if isinstance(obj, str):
            self.tail_correction = self.get_correction(obj)
        else:
            self.tail_correction = obj

    def set_blank_generator(self, obj):
        if isinstance(obj, str):
            obj = obj.lower()
            if obj == "nan":
                self.blank_generator = self.create_nan_array
            elif obj == "empty":
                self.blank_generator = np.empty
            elif obj == "zeros":
                self.blank_generator = np.zeros
            elif obj == "ones":
                self.blank_generator = np.ones
            elif obj == "full":
                self.blank_generator = np.full
        else:
            self.blank_generator = obj

    # Other Data Methods
    def interpolate_shift_other(self, y, x, shift, interp_type=None, axis=0, fill_value=None, **kwargs):
        if interp_type is None:
            interp_type = self.interpolate_type

        if fill_value is None:
            fill_value = self.interpolate_fill_value

        interpolator = interpolate.interp1d(x, y, interp_type, axis, fill_value=fill_value, **kwargs)
        new_x = x + shift
        new_y = interpolator(new_x)

        return new_x, new_y

    def shift_to_nearest_sample_end(self, data, time_axis, axis=None, tolerance=None, **kwargs):
        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        shift = time_axis[0] - self.time_axis[-1]
        if shift < 0:
            raise ValueError("cannot shift data to an existing range")
        elif shift - self.sample_period > tolerance:
            if round(shift * self.sample_rate) < 1:
                remain = shift - self.sample_period
            else:
                remain = math.remainder(shift, self.sample_period)
            small_shift = -remain
            data, time_axis = self.interpolate_shift_other(data, time_axis, small_shift, axis=axis, **kwargs)

        return data, time_axis

    def shift_to_the_end(self, data, time_axis, axis=None, tolerance=None, **kwargs):
        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        shift = time_axis[0] - self.time_axis[-1]
        if abs(shift - self.sample_period) > tolerance:
            small_shift = - math.remainder(shift, self.sample_period)
            large_shift = shift - self.sample_period + small_shift

            data, time_axis = self.interpolate_shift_other(data, time_axis, small_shift, axis=axis, **kwargs)
            time_axis -= large_shift

        return data, time_axis

    def default_tail_correction(self, data, time_axis, axis=None, tolerance=None, **kwargs):
        shift = time_axis[0] - self.time_axis[-1]
        if shift >= 0:
            data, time_axis = self.shift_to_nearest_sample_end(data, time_axis, axis, tolerance, **kwargs)
        else:
            data, time_axis = self.shift_to_the_end(data, time_axis, axis, tolerance, **kwargs)
        return data, time_axis

    # Data
    def shift_times(self, shift, start=None, stop=None, step=None):
        if self.mode == 'r':
            raise IOError("not writable")

        self.time_axis[start:stop:step] += shift

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

        self.data = np.append(self.data, data, axis)
        self.time_axis = np.append(self.time_axis, time_axis, 0)

    def append_frame(self, frame, axis=None, truncate=None, correction=None):
        if self.mode == 'r':
            raise IOError("not writable")

        if truncate is None:
            truncate = self.is_truncate

        if not frame.validate_sample_rate() or frame.sample_rate != self.sample_rate:
            raise ValueError("the frame's sample rate does not match this object's")

        shape = self.shape
        slices = ...
        if not frame.validate_shape or frame.shape != shape:
            if not truncate:
                raise ValueError("the frame's shape does not match this object's")
            else:
                slices = [None] * len(shape)
                for index, size in enumerate(shape):
                    slices[index] = slice(None, size)
                slices[axis] = slice(None, None)
                slices = tuple(slices)

        self.append(frame[slices], frame.get_time_axis(), axis, correction=correction)

    def add_frames(self, frames, axis=None, truncate=None):
        if self.mode == 'r':
            raise IOError("not writable")

        frames = list(frames)

        if self.data is None:
            frame = frames.pop(0)
            if not frame.validate_sample_rate():
                raise ValueError("the frame's sample rate must be valid")
            self.data = frame[...]
            self.time_axis = frame.get_time_axis()

        for frame in frames:
            self.append_frame(frame, axis=axis, truncate=truncate)

    def get_time(self, super_index):
        return datetime.datetime.fromtimestamp(self.time_axis[super_index])

    def get_times(self, start=None, stop=None, step=None):
        if start is not None:
            start = int(start)
        if stop is not None:
            stop = int(stop)
        if step is not None:
            step = int(step)

        return self.time_axis[slice(start, stop, step)]

    def get_intervals(self, start=None, stop=None, step=None):
        return np.ediff1d(self.time_axis[slice(start, stop, step)])

    # Find
    def find_time_index(self, timestamp, aprox=False, tails=False):
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()

        samples = self.get_length()
        if timestamp < self.time_axis[0]:
            if tails:
                return 0, self.start
        elif timestamp > self.time_axis[-1]:
            if tails:
                return samples, self.end
        else:
            index = np.searchsorted(self.time_axis, timestamp, side="right") - 1
            if aprox or timestamp == self.time_axis[index]:
                return index, datetime.datetime.fromtimestamp(self.time_axis[index])

        return -1, datetime.datetime.fromtimestamp(timestamp)

    def find_time_sample(self, timestamp, aprox=False, tails=False):
        return self.find_time_index(timestamp=timestamp, aprox=aprox, tails=tails)

    # Sample Rate
    def validate_sample_rate(self, tolerance=None):
        return self.validate_continuous(tolerance=tolerance)

    def resample(self, sample_rate, **kwargs):
        if self.mode == 'r':
            raise IOError("not writable")

        if not self.validate_sample_rate():
            raise ValueError("the data needs to have a uniform sample rate before resampling")

        # Todo: Make Resample for multiple frames (maybe edit resampler from an outer layer)
        self.data = self.resampler(data=self.data[...], new_fs=self.sample_rate, **kwargs)
        self.time_axis = np.arange(self.time_axis[0], self.time_axis[-1], self.sample_period, dtype="f8")

    # Continuous Data
    def evaluate_continuity(self, tolerance=None):
        if tolerance is None:
            tolerance = self.time_tolerance

        if self.time_axis.shape[0] > self.switch_algorithm_size:
            for index in range(0, len(self.time_axis) - 1):
                interval = self.time_axis[index + 1] - self.time_axis[index]
                if abs(interval - self.sample_period) > tolerance:
                    return False
        elif False in np.abs(np.ediff1d(self.time_axis) - self.sample_period) <= tolerance:
            return False

        return True

    def where_discontinuous(self, tolerance=None):
        # Todo: Get discontinuity type and make report
        if tolerance is None:
            tolerance = self.time_tolerance

        if self.time_axis.shape[0] > self.switch_algorithm_size:
            discontinuous = []
            for index in range(0, len(self.time_axis)-1):
                interval = self.time_axis[index] - self.time_axis[index-1]
                if abs(interval - self.sample_period) >= tolerance:
                    discontinuous.append(index)
        else:
            discontinuous = list(np.where(np.abs(np.ediff1d(self.time_axis) - self.sample_period) > tolerance)[0] + 1)

        if discontinuous:
            return discontinuous
        else:
            return None

    def validate_continuous(self, tolerance=None):
        if self._is_continuous is None or not self.is_cache:
            self._is_continuous = self.evaluate_continuity(tolerance=tolerance)
        return self._is_continuous

    def make_continuous(self, axis=None, tolerance=None):
        self.time_correction_interpolate(axis=axis, tolerance=tolerance)
        self.fill_time_correction(axis=axis, tolerance=tolerance)

    # Time Correction
    def time_correction_interpolate(self, axis=None, interp_type=None, fill_value=None, tolerance=None, **kwargs):
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if interp_type is None:
            interp_type = self.interpolate_type

        if fill_value is None:
            fill_value = self.interpolate_fill_value

        discontinuities = self.where_discontinuous(tolerance=tolerance)
        while discontinuities:
            discontinuity = discontinuities.pop(0)
            timestamp = self.time_axis[discontinuity]
            previous = discontinuity - 1
            previous_timestamp = self.time_axis[previous]

            if (timestamp - previous_timestamp) < (2 * self.sample_period):
                consecutive = [previous, discontinuity]
                start = previous_timestamp + self.sample_period
            else:
                consecutive = [discontinuity]
                nearest = round((timestamp - previous_timestamp) * self.sample_rate)
                start = previous_timestamp + self.sample_period * nearest

            if discontinuities:
                for next_d in discontinuities:
                    if (self.time_axis[next_d] - self.time_axis[next_d - 1]) < (2 * self.sample_period):
                        consecutive.append(discontinuities.pop(0))
                    else:
                        consecutive.append(next_d - 1)
                        break
            else:
                consecutive.append(len(self.time_axis) - 1)

            new_size = consecutive[-1] + 1 - consecutive[0]
            end = start + self.sample_period * (new_size - 1)
            new_times = np.arange(start, end, self.sample_period)
            if new_size > 1:
                times = self.time_axis[consecutive[0]: consecutive[-1] + 1]
                data = self.get_range(consecutive[0], consecutive[-1] + 1)
                interpolator = interpolate.interp1d(times, data, interp_type, axis, fill_value=fill_value, **kwargs)
                self.set_range(interpolator(new_times), start=discontinuity)
            else:
                self.time_axis[discontinuity] = start

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
            self.data = self.blank_generator(shape=new_shape, **kwargs)
            self.time_axis = np.empty((new_size,), dtype="f8")
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
