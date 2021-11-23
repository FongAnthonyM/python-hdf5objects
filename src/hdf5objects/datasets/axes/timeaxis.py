#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" timeaxis.py
Description:
"""
# Package Header #
from ...__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
import datetime

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache_method
import h5py
import numpy as np
import pytz
import tzlocal

# Local Packages #
from .axis import AxisMap, Axis


# Definitions #
# Functions #
def datetimes_to_timestamps(iter_):
    for dt in iter_:
        yield dt.timestamp()


# Classes #
class TimeAxisMap(AxisMap):
    default_attribute_names = {"time_zone": "time_zone"}


class TimeAxis(Axis):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    local_timezone = tzlocal.get_localzone()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, start: datetime.datetime = None, stop: datetime.datetime = None, step = None,
                 rate: float = None, size: int = None, datetimes=None, s_name: str = None,
                 build: bool = None, init: bool = True, **kwargs):
        super().__init__(init=False)
        self.default_kwargs = {"dtype": 'f8', "maxshape": (None,)}
        self._scale_name = "time axis"

        if init:
            self.construct(start=start, stop=stop, step=step, rate=rate, size=size,
                           datetimes=datetimes, s_name=s_name, build=build, **kwargs)

    @property
    def time_zone(self):
        return self.get_time_zone(refresh=False)

    @time_zone.setter
    def time_zone(self, value):
        self.set_time_zone(value)

    @property
    def timestamps(self):
        try:
            return self.get_all_data.caching_call()
        except AttributeError:
            return self.get_all_data()

    @property
    def datetimes(self):
        try:
            return self.get_as_datetimes.caching_call()
        except AttributeError:
            return self.get_as_datetimes()

    @property
    def start_datetime(self):
        return datetime.datetime.fromtimestamp(self.start, self.time_zone)

    @property
    def end_datetime(self):
        return datetime.datetime.fromtimestamp(self.end, self.time_zone)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, start: datetime.datetime = None, stop: datetime.datetime = None, step = None,
                  rate: float = None, size: int = None, datetimes=None, s_name: str = None,
                  build: bool = None, **kwargs):
        if "data" in kwargs:
            kwargs["build"] = build
            build = False

        super().construct(s_name=s_name, **kwargs)

        if build or build is None:
            if datetimes is not None:
                self.from_datetimes(datetimes)
            elif start is not None:
                self.from_range(start, stop, step, rate, size)

    def from_range(self, start: datetime.datetime = None, stop: datetime.datetime = None, step=None,
                   rate: float = None, size: int = None, **kwargs):
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)

        if step is None and rate is not None:
            step = 1 / rate
        elif isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if start is None:
            start = stop - step * size
        elif isinstance(start, datetime.datetime):
            start = start.timestamp()

        if stop is None:
            stop = start + step * size

        if size is not None:
            self.set_data(data=np.linspace(start, stop, size), **d_kwargs)
        else:
            self.set_data(data=np.arange(start, stop, step), **d_kwargs)

    def from_datetimes(self, iter_, **kwargs):
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)

        stamps = np.zeros(shape=(len(iter_),))
        for index, dt in enumerate(iter_):
            stamps[index] = dt.timestamp()
        self.set_data(data=stamps, **d_kwargs)

    # File
    def create(self, name=None, **kwargs):
        super().create(name=name, **kwargs)
        if "time_zone" not in self.attributes:
            tz = self.map.attributes.get("time_zone", None)
            self.set_time_zone(tz)
        return self

    def require(self, name=None, **kwargs):
        super().require(name=name, **kwargs)
        if "time_zone" not in self.attributes:
            tz = self.map.attributes.get("time_zone", None)
            self.set_time_zone(tz)
        return self

    def refresh(self):
        super().refresh()
        self.get_as_datetimes.clear_cache()

    # Getters/Setter
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_all_data(self):
        self.get_as_datetimes.clear_cache()
        with self:
            return self._dataset[...]

    def get_time_zone(self, refresh: bool = True):
        if refresh:
            self.attributes.refresh()
        tz_str = self.attributes.get("time_zone", self.sentinel)
        if tz_str is self.sentinel or isinstance(tz_str, h5py.Empty) or tz_str == "":
            return None
        else:
            return pytz.timezone(tz_str)

    def set_time_zone(self, value: str = None):
        if value is None:
            value = ""
        elif value.lower() == "local":
            value = self.local_timezone
        self.attributes["time_zone"] = value

    def get_timestamps(self):
        return self.get_all_data()

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_as_datetimes(self, tz=None):
        origin_tz = self.time_zone
        timestamps = self.get_all_data()
        if tz is not None:
            return tuple(datetime.datetime.fromtimestamp(t, origin_tz).astimezone(tz) for t in timestamps)
        else:
            return tuple(datetime.datetime.fromtimestamp(t, origin_tz) for t in timestamps)

    def get_datetime(self, index):
        return self.datetimes[index]

    def get_datetime_range(self, start=None, stop=None, step=None):
        if start is not None:
            start = int(start)
        if stop is not None:
            stop = int(stop)
        if step is not None:
            step = int(step)

        return self.datetimes[slice(start, stop, step)]

    def get_timestamp_range(self, start=None, stop=None, step=None):
        if start is not None:
            start = int(start)
        if stop is not None:
            stop = int(stop)
        if step is not None:
            step = int(step)

        return self.timestamps[slice(start, stop, step)]

    # Find
    def find_time_index(self, timestamp, aprox=False, tails=False):
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()

        samples = self.timestamps.shape[0]
        if timestamp < self.timestamps[0]:
            if tails:
                return 0, self.start
        elif timestamp > self.timestamps[-1]:
            if tails:
                return samples, self.end
        else:
            index = int(np.searchsorted(self.timestamps, timestamp, side="right") - 1)
            if aprox or timestamp == self.timestamps[index]:
                return index, datetime.datetime.fromtimestamp(self.timestamps[index])

        return None, datetime.datetime.fromtimestamp(timestamp)

    def get_timestamp_range_time(self, start=None, stop=None, step=None, aprox=False, tails=False):
        start_sample, true_start = self.find_time_index(timestamp=start, aprox=aprox, tails=tails)
        stop_sample, true_stop = self.find_time_index(timestamp=stop, aprox=aprox, tails=tails)

        return self.get_timestamp_range(start=start_sample, stop=stop_sample, step=step), true_start, true_stop

    def get_datetime_range_time(self, start=None, stop=None, step=None, aprox=False, tails=False):
        start_sample, true_start = self.find_time_index(timestamp=start, aprox=aprox, tails=tails)
        stop_sample, true_stop = self.find_time_index(timestamp=stop, aprox=aprox, tails=tails)

        return self.get_datetime_range(start=start_sample, stop=stop_sample, step=step), true_start, true_stop


# Assign Cyclic Definitions
TimeAxisMap.default_type = TimeAxis
TimeAxis.default_map = TimeAxisMap()
