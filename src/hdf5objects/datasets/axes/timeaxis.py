#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" timeaxis.py
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
                 build: bool = True, init: bool = True, **kwargs):
        super().__init__(init=False)
        self.default_kwargs = {"dtype": 'f8', "maxshape": (None,)}
        self._scale_name = "timestamps"

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
                  build: bool = True, **kwargs):
        if "data" in kwargs:
            kwargs["build"] = build
            build = False

        super().construct(s_name=s_name, **kwargs)

        if build:
            if datetimes is not None:
                self.from_datetimes(datetimes)
            else:
                self.from_range(start, stop, step, rate, size)

    # Getters/Setters
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
            value = h5py.Empty('S')
        elif value.lower() == "local":
            value = self.local_timezone
        self.attributes["time_zone"] = value
        self.get_time_zone.clear_cache()

    def get_timestamps(self):
        return self.get_all_data()

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_as_datetimes(self, tz=None):
        origin_tz = self.time_zone
        timestamps = self.get_all_data()
        if tz is not None:
            return (datetime.datetime.fromtimestamp(t, origin_tz).astimezone(tz) for t in timestamps)
        else:
            return (datetime.datetime.fromtimestamp(t, origin_tz) for t in timestamps)

    # Modification
    def require(self, name=None, **kwargs):
        super().require(name=name, **kwargs)
        if "time_zone" not in self.attributes:
            tz = self.map.attributes.get("time_zone", None)
            self.set_time_zone(tz)
        return self

    def from_range(self, start: datetime.datetime = None, stop: datetime.datetime = None, step=None,
                   rate: float = None, size: int = None, **kwargs):
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)

        if isinstance(start, datetime.datetime):
            start = start.timestamp()

        if isinstance(stop, datetime.datetime):
            stop = stop.timestamp()

        if step is None and rate is not None:
            step = 1 / rate
        elif isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if start is None:
            start = stop - step * size

        if stop is None:
            stop = start + step * size

        if size is not None:
            self.require(data=np.linspace(start, stop, size), **d_kwargs)
        else:
            self.require(data=np.arange(start, stop, step), **d_kwargs)

    def from_datetimes(self, iter_, **kwargs):
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)

        stamps = np.array([])
        for dt in iter_:
            stamps = np.append(stamps, dt.timestamp())
        self.require(data=stamps, **d_kwargs)


# Assign Cyclic Definitions
TimeAxisMap.default_type = TimeAxis
TimeAxis.default_map = TimeAxisMap()
