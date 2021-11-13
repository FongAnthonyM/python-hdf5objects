#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" timeaxis.py
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

# Standard Libraries #
import datetime

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache_method
import h5py
import numpy as np
import pytz
import tzlocal

# Local Packages #
from ..hdf5objects import HDF5Map, HDF5Dataset


# Definitions #
# Functions #
def datetimes_to_timestamps(iter_):
    for dt in iter_:
        yield dt.timestamp()


# Classes #
class TimeAxisMap(HDF5Map):
    default_attributes = {"time_zone": "time_zone"}


class TimeAxis(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_map = None
    local_timezone = tzlocal.get_localzone()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, obj=None, start=None, stop=None, step=None, rate=None, size=None,
                 create=True, init=True, **kwargs):
        super().__init__(init=False)
        self.default_kwargs = {"dtype": 'f8', "maxshape": (None,)}
        self.label = "timestamps"

        self._datetimes = None

        if init:
            self.construct(obj, start=start, stop=stop, step=step, rate=rate, size=size, create=create, **kwargs)

    @property
    def time_zone(self):
        try:
            return self.get_time_zone.caching_call()
        except AttributeError:
            return self.get_time_zone()

    @time_zone.setter
    def time_zone(self, value):
        self.set_time_zone(value)

    @property
    def datetimes(self):
        if self._datetimes is None or self.is_updating:
            self._datetimes = tuple(self.as_datetimes())

        return self._datetimes

    @property
    def start(self):
        try:
            return self.get_start.caching_call()
        except AttributeError:
            return self.get_start()

    @property
    def start_datetime(self):
        return datetime.datetime.fromtimestamp(self.start, self.time_zone)

    @property
    def end(self):
        try:
            return self.get_start.caching_call()
        except AttributeError:
            return self.get_start()

    @property
    def end_datetime(self):
        return datetime.datetime.fromtimestamp(self.end, self.time_zone)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, obj=None, stop=None, step=None, rate=None, size=None, create=True, start=None, **kwargs):
        super().construct(**kwargs)
        if isinstance(obj, (h5py.Dataset, HDF5Dataset)):
            self.set_dataset(obj)
        elif create:
            if obj is None:
                self.from_range(start, stop, step, rate, size)
            elif isinstance(obj, datetime.datetime):
                self.from_range(obj, stop, step, rate, size)
            elif isinstance(obj, h5py.Dataset):
                self._dataset = obj
            elif isinstance(obj, HDF5Dataset):
                self._dataset = obj._dataset
            else:
                self.from_datetimes(obj)

    # Getters/Setters
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_time_zone(self):
        tz_str = self.attributes["time_zone"]
        if isinstance(tz_str, h5py.Empty) or tz_str == "":
            return None
        else:
            return pytz.timezone(tz_str)

    def set_time_zone(self, value):
        if value is None:
            tz_str = h5py.Empty('S')
        else:
            tz_str = value
        self.attributes["time_zone"] = tz_str
        self.get_time_zone.clear_cache()

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_start(self):
        with self:
            return self._dataset[0]

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_end(self):
        with self:
            return self._dataset[-1]

    # Modification
    def from_range(self, start=None, stop=None, step=None, rate=None, size=None, **kwargs):
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)

        if isinstance(start, datetime.datetime):
            start = start.timestamp()

        if isinstance(stop, datetime.datetime):
            stop = stop.timestamp()

        if step is None and rate is not None:
            step = 1 / rate

        if start is None:
            start = stop - step * size

        if stop is None:
            stop = start + step * size

        if size is not None:
            self.require(data=np.linspace(start, stop, size), **d_kwargs)
        else:
            self.require(data=np.arange(start, stop, step), **d_kwargs)

        with self:
            self._dataset.make_scale("timestamps")

    def from_datetimes(self, iter_, **kwargs):
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)

        stamps = np.array([])
        for dt in iter_:
            stamps = np.append(stamps, dt.timestamp())
        self.require(data=stamps, **d_kwargs)

        with self:
            self._dataset.make_scale(self.label)

    def as_datetimes(self, tz=None):
        origin_tz = self.timezone
        if tz is not None and origin_tz is not None:
            return [datetime.datetime.fromtimestamp(t, origin_tz).astimezone(tz) for t in self._dataset]
        else:
            return [datetime.datetime.fromtimestamp(t, origin_tz) for t in self._dataset]

    def require(self, name=None, **kwargs):
        super().require(name=name, **kwargs)
        if "timezone" in self.map.attributes and not self.map.attributes["timezone"] in self.attributes:
            if self._timezone is None:
                self._timezone = self.local_timezone
            self.attributes[self.map.attributes["timezone"]] = self._timezone

        return self


# Assign Cyclic Definitions
TimeAxisMap.default_type = TimeAxis
TimeAxis.default_map = TimeAxisMap()
