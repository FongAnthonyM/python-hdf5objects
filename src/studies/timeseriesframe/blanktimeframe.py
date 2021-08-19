#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" blanktimeframe.py
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

# Local Libraries #
from ..blankdataframe import BlankDataFrame
from .timeseriesframeinterface import TimeSeriesFrameInterface


# Definitions #
# Classes #
class BlankTimeFrame(BlankDataFrame, TimeSeriesFrameInterface):
    # Magic Methods
    # Construction/Destruction
    def __init__(self, start=None, end=None, sample_rate=None, sample_period=None, shape=None, init=True):
        super().__init__(init=False)
        self._start = None
        self._true_end = None
        self._assigned_end = None

        self._sample_rate = None

        if init:
            self.construct(start=start, end=end, sample_rate=sample_rate, sample_period=sample_period, shape=shape)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        self.refresh()

    @property
    def n_samples(self):
        return self.get_length()

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        if isinstance(value, datetime.datetime):
            self._start = value
        else:
            self._start = datetime.datetime.fromtimestamp(value)
        self.refresh()

    @property
    def true_end(self):
        return self._true_end

    @true_end.setter
    def true_end(self, value):
        if isinstance(value, datetime.datetime):
            self._true_end = value
        else:
            self._true_end = datetime.datetime.fromtimestamp(value)

    @property
    def assigned_end(self):
        return self._assigned_end

    @assigned_end.setter
    def assigned_end(self, value):
        if isinstance(value, datetime.datetime):
            self._assigned_end = value
        else:
            self._assigned_end = datetime.datetime.fromtimestamp(value)
        self.refresh()

    @property
    def end(self):
        return self.true_end

    @end.setter
    def end(self, value):
        self.assigned_end = value

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value
        self.refresh()

    @property
    def sample_period(self):
        return 1 / self.sample_rate

    @sample_period.setter
    def sample_period(self, value):
        self.sample_rate = 1 / value

    @property
    def is_continuous(self):
        return self.validate_continuous()

    # Instance Methods
    # Constructors/Destructors
    def construct(self, start=None, end=None, sample_rate=None, sample_period=None, shape=None, dtype=None):
        if start is not None:
            if isinstance(start, datetime.datetime):
                self._start = start
            else:
                self._start = datetime.datetime.fromtimestamp(start)

        if end is not None:
            if isinstance(end, datetime.datetime):
                self._assigned_end = end
            else:
                self._assigned_end = datetime.datetime.fromtimestamp(end)

        if sample_period is not None:
            self._sample_rate = 1 / sample_period

        if sample_rate is not None:
            self._sample_rate = sample_rate

        super().construct(shape=shape, dtype=dtype)

        self.refresh()

    # Getters
    def get_length(self):
        start = self.start.timestamp()
        end = self.assigned_end.timestamp()

        size = math.floor((end - start) * self.sample_rate)
        r = math.remainder((end - start), self.sample_period)
        remain = r if r >= 0 else self.sample_period + r
        self.true_end = end - remain
        self.shape[self.axis] = size

        return size

    def get_time_axis(self):
        return self.create_times()

    def refresh(self):
        try:
            self.get_length()
        except AttributeError:
            pass

    # Data
    def create_times(self, start=None, stop=None, step=None, dtype=None):
        samples = self.get_length()
        frame_start = self.start.timestamp()

        if dtype is None:
            dtype = "f8"

        if start is None:
            start = 0

        if stop is None:
            stop = samples
        elif stop < 0:
            stop = samples + stop

        if step is None:
            step = 1

        if start >= samples or stop < 0:
            raise IndexError("index is out of range")

        start_timestamp = frame_start + self.sample_rate * start
        stop_timestamp = frame_start + self.sample_rate * stop
        period = self.sample_period * step

        return np.arange(start_timestamp, stop_timestamp, period, dtype=dtype)

    def create_times_slice(self, slice_, dtype=None):
        return self.create_times(start=slice_.start, stop=slice_.stop, step=slice_.step, dtype=dtype)

    def get_time(self, super_index):
        return self.create_times(start=super_index, stop=super_index + 1)

    def get_times(self, start=None, stop=None, step=None):
        return self.create_times(start=start, stop=stop, step=step)

    # Find
    def find_time_index(self, timestamp, aprox=False, tails=False):
        return self.find_time_sample(timestamp=timestamp, aprox=aprox, tails=tails)

    def find_time_sample(self, timestamp, aprox=False, tails=False):
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()

        samples = self.get_length()
        if timestamp < self.start.timestamp():
            if tails:
                return 0, self.start
        elif timestamp > self.end.timestamp():
            if tails:
                return samples, self.end
        else:
            remain, sample = math.modf((timestamp - self.start.timestamp()) * self.sample_rate)
            if aprox or remain == 0:
                true_timestamp = sample / self.sample_rate + self.start.timestamp()
                return int(sample), datetime.datetime.fromtimestamp(true_timestamp)

        return -1, datetime.datetime.fromtimestamp(timestamp)

    # Shape
    def validate_shape(self):
        self.refresh()
        return True

    def change_size(self, shape=None, **kwargs):
        self.shape = shape
        self.refresh()

    # Sample Rate
    def validate_sample_rate(self):
        self.refresh()
        return True

    def resample(self, new_rate, **kwargs):
        self.sample_rate = new_rate
        self.refresh()

    # Continuous Data
    def validate_continuous(self):
        self.refresh()
        return True

    def make_continuous(self):
        self.refresh()
