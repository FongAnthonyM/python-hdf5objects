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
from abc import abstractmethod
import datetime
import math

# Downloaded Libraries #
import numpy as np

# Local Libraries #
from .dataframe import DataFrame, DataFrameInterface


# Definitions #
# Classes #
class TimeSeriesFrameInterface(DataFrameInterface):
    # Magic Methods
    # Construction/Destruction
    def __init__(self, data=None, times=True, init=True):
        self.axis = 0
        self.sample_rate = 0

        self.data = None
        self.times = None

        if init:
            self.construct(data, times)

    # Container Methods
    def __len__(self):
        return self.data.shape[self.axis]

    def __getitem__(self, item):
        return self.data[item]

    # Instance Methods
    # Constructors/Destructors
    def construct(self, data=None, times=True):
        if data is not None:
            self.data = data

        if times is not None:
            self.times = times

    # Data
    @abstractmethod
    def get_range(self, start=None, stop=None, step=None):
        pass

    @abstractmethod
    def get_times(self, start=None, stop=None, step=None):
        return self.times[slice(start, stop, step)]

    # Find
    @abstractmethod
    def find_time_index(self, timestamp, aprox=False, tails=False):
        pass

    @abstractmethod
    def find_time_sample(self, timestamp, aprox=False, tails=False):
        pass

    # Get with Time
    def get_time_range(self, start=None, stop=None, aprox=False, tails=False):
        start_sample, true_start = self.find_time_sample(timestamp=start, aprox=aprox, tails=tails)
        end_sample, true_end = self.find_time_sample(timestamp=stop, aprox=aprox, tails=tails)

        return self.get_times(start_sample, end_sample), true_start, true_end

    def data_range_time(self, start=None, stop=None, aprox=False, tails=False):
        start_sample, true_start = self.find_time_sample(timestamp=start, aprox=aprox, tails=tails)
        end_sample, true_end = self.find_time_sample(timestamp=stop, aprox=aprox, tails=tails)

        return self.get_range(start_sample, end_sample), true_start, true_end

    # Shape
    @abstractmethod
    def validate_shape(self):
        pass

    @abstractmethod
    def reshape(self, shape=None, **kwargs):
        pass

    # Sample Rate
    @abstractmethod
    def validate_sample_rate(self):
        pass

    @abstractmethod
    def resample(self, new_rate, **kwargs):
        pass

    # Continuous Data
    @abstractmethod
    def validate_continuous(self):
        pass

    @abstractmethod
    def make_continuous(self):
        pass


class BlankTimeFrame(TimeSeriesFrameInterface):
    # Static Methods
    @staticmethod
    def create_nan_array(shape=None, **kwargs):
        a = np.empty(shape=shape, **kwargs)
        a.fill(np.nan)
        return a

    # Magic Methods
    # Construction/Destruction
    def __init__(self, start=None, end=None, sample_rate=None, sample_period=None, shape=None, init=True):
        super().__init__(init=False)
        self.is_cache = True

        self.dtype = "f4"
        self._shape = None

        self._start = None
        self._true_end = None
        self._assigned_end = None

        self._sample_rate = None

        self.generate_data = self.create_nan_array

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
        return self.get_n_samples()

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

    # Container Methods
    def __len__(self):
        return self.data.shape[self.axis]

    def __getitem__(self, item):
        return self.get_item(item)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, start=None, end=None, sample_rate=None, sample_period=None, shape=None):
        if start is not None:
            self.start = start

        if end is not None:
            self.end = end

        if sample_period is not None:
            self.sample_period = sample_period

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if shape is not None:
            self.shape = shape

        self.refresh()

    # Setters
    def set_data_generator(self, obj):
        if isinstance(obj, str):
            if obj.lower() == "nan":
                self.generate_data = self.create_nan_array
            elif obj.lower() == "empty":
                self.generate_data = np.empty
            elif obj.lower() == "zeros":
                self.generate_data = np.zeros
            elif obj.lower() == "ones":
                self.generate_data = np.ones
            elif obj.lower() == "full":
                self.generate_data = np.full
        else:
            self.generate_data = obj

    # Getters
    def get_item(self, item):
        if isinstance(item, slice):
            return self.create_data_slice(item)
        elif isinstance(item, (tuple, list)):
            return self.create_data_slice(item)
        elif isinstance(item, ...):
            return self.create_data()

    def get_n_samples(self):
        start = self.start.timestamp()
        end = self.assigned_end.timestamp()

        remain, samples = math.modf((end - start) / self.sample_rate)
        self.true_end = end - remain
        self.shape[self.axis] = samples

        return samples

    def refresh(self):
        try:
            self.get_n_samples()
        except AttributeError:
            pass

    # Data
    def create_times(self, start=None, stop=None, step=None, dtype=None):
        samples = self.get_n_samples()
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

    def create_data(self, start=None, stop=None, step=None, dtype=None, **kwargs):
        samples = self.get_n_samples()
        shape = self.shape

        if dtype is None:
            dtype = self.dtype

        if start is None:
            start = 0

        if stop is None:
            stop = samples
        elif stop < 0:
            stop = samples + stop

        if start >= samples or stop < 0:
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
        samples = self.get_n_samples()
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
            stop = samples
        elif stop < 0:
            stop = samples + stop

        if step is None:
            step = 1

        if start >= samples or stop < 0:
            raise IndexError("index is out of range")

        size = stop - start
        shape[self.axis] = size
        slices[self.axis] = slice(None, None, step)

        return self.generate_data(shape=self.shape, dtype=dtype, **kwargs)[tuple(slices)]

    def get_range(self, start=None, stop=None, step=None):
        return self.create_data(start=start, stop=stop, step=step)

    def get_times(self, start=None, stop=None, step=None):
        return self.create_times(start=start, stop=stop, step=step)

    # Find
    def find_time_index(self, timestamp, aprox=False, tails=False):
        return self.find_time_sample(timestamp=timestamp, aprox=aprox, tails=tails)

    def find_time_sample(self, timestamp, aprox=False, tails=False):
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()

        samples = self.get_n_samples()
        difference = timestamp - self.start.timestamp()
        if difference < 0:
            if tails:
                return 0, self.start
        elif difference > samples:
            if tails:
                return samples, self.end
        else:
            remain, sample = math.modf(difference * self.sample_rate)
            if aprox or remain == 0:
                true_timestamp = sample / self.sample_rate + self.start.timestamp()
                return int(sample), datetime.datetime.fromtimestamp(true_timestamp)

        return -1, datetime.datetime.fromtimestamp(timestamp)

    # Shape
    def validate_shape(self):
        self.refresh()
        return True

    def reshape(self, shape=None, **kwargs):
        self.shape = shape

    # Sample Rate
    def validate_sample_rate(self):
        self.refresh()
        return True

    def resample(self, new_rate, **kwargs):
        self.sample_rate = new_rate

    # Continuous Data
    def validate_continuous(self):
        self.refresh()
        return True

    def make_continuous(self):
        self.refresh()


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

        self.start = None
        self.end = None
        self.start_sample = None
        self.end_sample = None

        if init:
            self.construct(frames, update)

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
        super().construct(frames, update)

        if self.frames:
            self.get_time_info()

    def frame_sort_key(self, frame):
        return frame.start

    # Cache and Memory
    def refresh(self):
        super().refresh()
        self.get_sample_rates()
        self.get_sample_rate()
        self.get_sample_period()
        self.get_is_continuous()

    # Getters
    def get_start_timestamps(self):
        starts = np.empty(len(self.frames))
        for index, frame in self.frames:
            starts[index] = frame.start.timestamp()
        return starts

    def get_end_timestamps(self):
        ends = np.empty(len(self.frames))
        for index, frame in self.frames:
            ends[index] = frame.end.timestamp()
        return ends

    def get_time_info(self):
        if len(self.frames) > 0:
            self.start = self.frames[0].start
            self.end = self.frames[-1].end
            self.start_sample = self.frames[0].start_sample
            self.end_sample = self.frames[-1].end_sample
            self._date = self.start.date()
        return self.date, self.start, self.end

    def get_sample_rates(self):
        self._sample_rates = [frame.sample_rate for frame in self.frames]
        return self._sample_rates

    def get_sample_rate(self):
        self._sample_rate = self.validate_sample_rate()
        sample_rates = self.get_sample_rates()
        if self._sample_rate and len(sample_rates) > 0:
            self._sample_rate = sample_rates[0]
        return self._sample_rate

    def get_sample_period(self):
        self._sample_period = self.get_sample_rate()
        if not isinstance(self._sample_period, bool):
            self._sample_period = 1 / self._sample_period
        return self._sample_rate

    def get_is_continuous(self):
        self._is_continuous = self.validate_continuous()
        return self._is_continuous

    # Data
    def get_times(self, start=None, stop=None, step=None):
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

        frame_start, inner_start, _ = start_index
        frame_stop, inner_stop, _ = stop_index

        if (frame_start + 1) == frame_stop or (frame_start + 1) == len(self.frames) + frame_stop:
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

        if timestamp < times[0]:
            if tails:
                index = 0
        elif timestamp > times[-1]:
            if tails:
                index = times.shape[0]
        elif timestamp in times:
            index = np.where(times == timestamp)[0][0]
        else:
            index = np.where(times > timestamp)[0][0] - 1

        return index

    def find_time_index(self, timestamp, aprox=False, tails=False):
        index = self.find_frame_time(timestamp, tails)
        location = []
        true_timestamp = timestamp

        if index:
            frame = self.frames[index]
            if timestamp <= frame.end.timestamp():
                location, true_timestamp = frame.find_time_index(timestamp=timestamp, aprox=aprox)
            else:
                index = None

        return [index] + location, true_timestamp

    def find_time_sample(self, timestamp, aprox=False, tails=False):
        index = self.find_frame_time(timestamp, tails)
        frame_samples = sum(self.lengths[:index])
        inner_samples = 0
        true_timestamp = timestamp

        if index:
            frame = self.frames[index]
            if timestamp <= frame.end.timestamp():
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
        sample_rates = self.get_sample_rates()
        if sample_rates:
            rate = sample_rates.pop()
            if rate:
                for sample_rate in sample_rates:
                    if not sample_rate or rate != sample_rate:
                        return False
        return True

    def resample(self, sample_rate=None, combine=False, **kwargs):
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        for index, frame in enumerate(self.frames):
            if combine and frame.validate_sample_rate() and frame.sample_rate != sample_rate \
               and frame.validate_continuous():
                self.frames[index] = frame.combine_frames()
            elif not frame.validate_sample_rate() or frame.sample_rate != sample_rate:
                frame.resample(sample_rate, **kwargs)

    # Continuous Data
    def validate_continuous(self):
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

                if (second - first) - self.sample_period > self.time_tolerance:
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
