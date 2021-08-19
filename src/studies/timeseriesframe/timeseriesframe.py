#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" timeseriesframe.py
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
import datetime  # Todo: Consider Pandas Timestamp for nanosecond resolution

# Downloaded Libraries #
import numpy as np

# Local Libraries #
from ..dataframe import DataFrame
from .blanktimeframe import BlankTimeFrame


# Definitions #
# Classes #
class TimeSeriesFrame(DataFrame):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_fill_type = BlankTimeFrame

    # Magic Methods
    # Construction/Destruction
    def __init__(self, frames=None, update=True, init=True):
        super().__init__(init=False)

        self._date = None
        self.date_format = "%Y-%m-%d"

        self._sample_rates = None
        self._sample_rate = None
        self._sample_period = None
        self.target_sample_rate = None

        self._is_continuous = None
        self.time_tolerance = 0.000001
        self.fill_type = self.default_fill_type

        self._start = None
        self._end = None
        self.start_sample = None
        self.end_sample = None

        if init:
            self.construct(frames, update)

    @property
    def start(self):
        if self._start is None or (self.is_updating and not self._cache):
            return self.get_start()
        else:
            return self._start

    @property
    def end(self):
        if self._end is None or (self.is_updating and not self._cache):
            return self.get_end()
        else:
            return self._end

    @property
    def date(self):
        if self.start is None:
            return self._date
        else:
            self.start.date()

    @property
    def sample_rates(self):
        if self._sample_rates is None or (self.is_updating and not self._cache):
            return self.get_sample_rates()
        else:
            return self._sample_rates

    @property
    def sample_rate(self):
        if self._sample_rate is None or (self.is_updating and not self._cache):
            return self.get_sample_rate()
        else:
            return self._sample_rate

    @property
    def sample_period(self):
        if self._sample_period is None or (self.is_updating and not self._cache):
            return self.get_sample_period()
        else:
            return self._sample_period

    @property
    def is_continuous(self):
        if self._is_continuous is None or (self.is_updating and not self._cache):
            return self.get_is_continuous()
        else:
            return self._is_continuous

    # Instance Methods
    # Constructors/Destructors
    def construct(self, frames=None, update=True):
        super().construct(frames=frames, update=update)

    def frame_sort_key(self, frame):
        return frame.start

    # Cache and Memory
    def refresh(self):
        super().refresh()
        self.get_start()
        self.get_end()
        self.get_sample_rates()
        self.get_sample_rate()
        self.get_sample_period()
        self.get_is_continuous()

    # Getters
    def get_start(self):
        if self.frames:
            start = self.frames[0].start
            if isinstance(start, datetime.datetime):
                self._start = start
            else:
                self._start = datetime.datetime.fromtimestamp(start)
        else:
            self._start = None

        return self._start

    def get_end(self):
        if self.frames:
            end = self.frames[-1].end
            if isinstance(end, datetime.datetime):
                self._end = end
            else:
                self._end = datetime.datetime.fromtimestamp(end)
        else:
            self._end = None

        return self._end

    def get_start_timestamps(self):
        starts = np.empty(len(self.frames))
        for index, frame in enumerate(self.frames):
            starts[index] = frame.start.timestamp()
        return starts

    def get_end_timestamps(self):
        ends = np.empty(len(self.frames))
        for index, frame in self.frames:
            ends[index] = frame.end.timestamp()
        return ends

    def get_time_axis(self):
        time_axis = self.frames[0].get_time_axis
        for index in range(1, len(self.frames)):
            time_axis = self.smart_append(time_axis, self.frames[index].get_time_axis, axis=0)
        return time_axis

    def get_sample_rates(self):
        self._sample_rates = (frame.sample_rate for frame in self.frames)
        return self._sample_rates

    def get_sample_rate(self):
        sample_rates = list(self.get_sample_rates())
        if sample_rates:
            rate = sample_rates.pop()
            if rate:
                for sample_rate in sample_rates:
                    if not sample_rate or rate != sample_rate:
                        self._sample_rate = False
            self._sample_rate = rate
        else:
            self._sample_rate = False
        return self._sample_rate

    def get_sample_period(self):
        self._sample_period = self.get_sample_rate()
        if not isinstance(self._sample_period, bool):
            self._sample_period = 1 / self._sample_period
        return self._sample_period

    def get_is_continuous(self):
        self._is_continuous = self.validate_continuous()
        return self._is_continuous

    # Data
    def get_time(self, super_index):
        with self.cache():
            # Check if index is in range.
            if super_index >= self.length or (super_index + self.length) < 0:
                raise IndexError("index is out of range")

            # Change negative indexing into positive.
            if super_index < 0:
                super_index = self.length + super_index

            # Find
            previous = 0
            for frame_index, frame_length in enumerate(self.lengths):
                end = previous + frame_length
                if super_index < end:
                    return self.frames[frame_index].get_time(int(super_index - previous))
                else:
                    previous = end

    def get_times(self, start=None, stop=None, step=None):
        if start is not None and stop is not None:
            start_index, stop_index = self.find_frame_indices([start, stop])
        elif start is not None:
            start_index = self.find_frame_index(start)
            stop_index = [len(self.frames) - 1, None, None]
        elif stop is not None:
            stop_index = self.find_frame_index(stop)
            start_index = [0, None, None]
        else:
            start_index = [0, None, None]
            stop_index = [len(self.frames) - 1, None, None]

        frame_start, inner_start, _ = start_index
        frame_stop, inner_stop, _ = stop_index

        if frame_start == frame_stop:
            times = self.frames[frame_start].get_times(inner_start, inner_stop, step)
        else:
            times = self.frames[frame_start].get_times(inner_start, None, step)
            for fi in range(frame_start + 1, frame_stop):
                times = self.smart_append(times, self.frames[fi].get_times(None, None, step))
            times = self.smart_append(times, self.frames[frame_stop].get_times(None, inner_stop, step))

        return times

    # Find Time
    def find_frame_time(self, timestamp, tails=False):
        # Setup
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()
        index = None
        times = self.get_start_timestamps()

        if timestamp < self.start.timestamp():
            if tails:
                index = 0
        elif timestamp > self.end.timestamp():
            if tails:
                index = times.shape[0] - 1
        else:
            index = np.searchsorted(times, timestamp, side="right") - 1

        return index

    def find_time_index(self, timestamp, aprox=False, tails=False):
        if isinstance(timestamp, float):
            timestamp = datetime.datetime.fromtimestamp(timestamp)
        index = self.find_frame_time(timestamp, tails)
        location = []
        true_timestamp = timestamp

        if index:
            frame = self.frames[index]
            if timestamp <= frame.end:
                location, true_timestamp = frame.find_time_index(timestamp=timestamp, aprox=aprox)
            else:
                index = None

        return [index] + location, true_timestamp

    def find_time_sample(self, timestamp, aprox=False, tails=False):
        with self.cache():
            if isinstance(timestamp, float):
                timestamp = datetime.datetime.fromtimestamp(timestamp)
            index = self.find_frame_time(timestamp, tails)
            frame_samples = sum(self.lengths[:index])
            inner_samples = 0
            true_timestamp = timestamp

            if index is not None:
                frame = self.frames[index]
                if timestamp <= frame.end:
                    inner_samples, true_timestamp = frame.find_time_sample(timestamp=timestamp, aprox=aprox)
                else:
                    frame_samples = -1

            return frame_samples + inner_samples, true_timestamp

    # Get with Time
    def get_time_range(self, start=None, end=None, aprox=False, tails=False):
        start_sample, true_start = self.find_time_sample(start, aprox, tails)
        end_sample, true_end = self.find_time_sample(end, aprox, tails)

        return self.get_times(start_sample, end_sample), true_start, true_end

    def data_range_time(self, start=None, end=None, aprox=False, tails=False):
        start_sample, true_start = self.find_time_sample(start, aprox, tails)
        end_sample, true_end = self.find_time_sample(end, aprox, tails)

        return self.get_range(start_sample, end_sample), true_start, true_end

    # Sample Rate
    def validate_sample_rate(self):
        sample_rates = list(self.get_sample_rates())
        if sample_rates:
            rate = sample_rates.pop()
            if rate:
                for sample_rate in sample_rates:
                    if not sample_rate or rate != sample_rate:
                        return False
        return True

    def resample(self, sample_rate=None, combine=None, **kwargs):
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        if combine is None:
            combine = self.is_combine

        for index, frame in enumerate(self.frames):
            if not frame.validate_sample_rate() or frame.sample_rate != sample_rate:
                if frame.mode == 'r':
                    self.frames[index] = frame.editable_copy()
                elif combine:
                    try:
                        self.frames[index] = frame.combine_frames()
                    except AttributeError:
                        pass

                self.frames[index].resample(sample_rate=sample_rate, **kwargs)

    # Continuous Data
    def where_discontinuous(self, tolerance=None):
        if tolerance is None:
            tolerance = self.time_tolerance

        discontinuities = []
        for index, frame in enumerate(self.frames):
            discontinuities.append(frame.where_discontinuous())

            if index + 1 < len(self.frames):
                if isinstance(frame.end, datetime.datetime):
                    first = frame.end.timestamp()
                else:
                    first = frame.end

                if isinstance(self.frames[index + 1].start, datetime.datetime):
                    second = self.frames[index + 1].start.timestamp()
                else:
                    second = self.frames[index + 1].start

                if abs((second - first) - self.sample_period) > tolerance:
                    discontinuities.append(index + 1)
                else:
                    discontinuities.append(None)
        return discontinuities

    def validate_continuous(self, tolerance=None):
        if tolerance is None:
            tolerance = self.time_tolerance

        for index, frame in enumerate(self.frames):
            if not frame.validate_continuous():
                return False

            if index + 1 < len(self.frames):
                if isinstance(frame.end, datetime.datetime):
                    first = frame.end.timestamp()
                else:
                    first = frame.end

                if isinstance(self.frames[index + 1].start, datetime.datetime):
                    second = self.frames[index + 1].start.timestamp()
                else:
                    second = self.frames[index + 1].start

                if abs((second - first) - self.sample_period) > tolerance:
                    return False

        return True

    def make_continuous(self):
        fill_frames = []
        if self.validate_sample_rate():
            sample_rate = self.sample_rate
            sample_period = self.sample_period
        else:
            sample_rate = self.target_sample_rate
            sample_period = 1 / sample_rate

        if self.validate_shape():
            shape = self.shape
        else:
            shape = self.target_shape

        for index, frame in enumerate(self.frames):
            if not frame.validate_continuous():
                frame.make_continuous()

            if index + 1 < len(self.frames):
                if isinstance(frame.end, datetime.datetime):
                    first = frame.end.timestamp()
                else:
                    first = frame.end

                if isinstance(self.frames[index + 1].start, datetime.datetime):
                    second = self.frames[index + 1].start.timestamp()
                else:
                    second = self.frames[index + 1].start

                if (second - first) - sample_period > self.time_tolerance:
                    start = first + sample_period
                    end = second + sample_period
                    fill_frames.append(self.fill_type(start=start, end=end, sample_rate=sample_rate, shape=shape))

        if fill_frames:
            self.frames += fill_frames
            self.sort_frames()
            self.refresh()


# Assign Cyclic Definitions
TimeSeriesFrame.default_return_frame_type = TimeSeriesFrame
