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

# Default Libraries #
import datetime

# Downloaded Libraries #
from bidict import bidict
import h5py
import numpy as np
import pytz
import tzlocal

# Local Libraries #
from ..hdf5object import HDF5Map, HDF5Dataset


# Definitions #
# Functions #
def datetimes_to_timestamps(iter_):
    for dt in iter_:
        yield dt.timestamp()


# Classes #
class TimeAxisMap(HDF5Map):
    default_attributes = bidict({"time_zone": "time_zone"})


class TimeAxis(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_map = TimeAxisMap()
    local_timezone = tzlocal.get_localzone()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, obj=None, start=None, stop=None, step=None, rate=None, size=None,
                 create=True, init=True, **kwargs):
        super().__init__(init=False)
        self._timezone = None
        self.is_updating = True
        self.default_kwargs = {"dtype": 'f8', "maxshape": (None,)}
        self.label = "timestamps"

        self._datetimes = None
        self._start = None
        self._end = None

        if init:
            self.construct(obj, start, stop, step, rate, size, create, **kwargs)

    @property
    def timezone(self):
        try:
            tz_str = self.attributes[self.map.attributes["timezone"]]
            if isinstance(tz_str, h5py.Empty) or tz_str == "":
                self._timezone = None
            else:
                self._timezone = pytz.timezone(tz_str)
        finally:
            return self._timezone

    @timezone.setter
    def timezone(self, value):
        try:
            if value is None:
                tz_str = h5py.Empty('S')
            else:
                tz_str = value
            self.attributes[self.map.attributes["timezone"]] = tz_str
        finally:
            self._timezone = None

    @property
    def datetimes(self):
        if self._datetimes is None or self.is_updating:
            self._datetimes = tuple(self.as_datetimes())

        return self._datetimes

    @property
    def start(self):
        if self._start is None or self.is_updating:
            with self:
                self._start = self._dataset[0]

        return self._start

    @property
    def start_datetime(self):
        return datetime.datetime.fromtimestamp(self.start, self.timezone)

    @property
    def end(self):
        if self._end is None or self.is_updating:
            with self:
                self._end = self._dataset[-1]

        return self._end

    @property
    def end_datetime(self):
        return datetime.datetime.fromtimestamp(self.end, self.timezone)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, obj=None, stop=None, step=None, rate=None, size=None, create=True, start=None, **kwargs):
        super().construct(**kwargs)
        if isinstance(obj, (h5py.Dataset, HDF5Dataset)):
            self.set_dataset(obj)
        elif create:
            if obj is None:
                self.from_range(start, stop, step, rate, size, **kwargs)
            elif isinstance(obj, datetime.datetime):
                self.from_range(obj, stop, step, rate, size, **kwargs)
            elif isinstance(obj, h5py.Dataset):
                self._dataset = obj
            elif isinstance(obj, HDF5Dataset):
                self._dataset = obj._dataset
            else:
                self.from_datetimes(obj, **kwargs)

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
        if not self.map.attributes["timezone"] in self.attributes:
            if self._timezone is None:
                self._timezone = self.local_timezone
            self.attributes[self.map.attributes["timezone"]] = self._timezone

        return self

# Assign Cyclic Definitions
TimeAxisMap.default_type = TimeAxis
TimeAxis.default_map = TimeAxisMap()
