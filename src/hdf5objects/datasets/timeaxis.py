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
import h5py
import numpy as np
import pytz
import tzlocal

# Local Libraries #
from ..hdf5object import HDF5Dataset


# Definitions #
# Functions #
def datetimes_to_timestamps(iter_):
    for dt in iter_:
        yield dt.timestamp()


# Classes #
class TimeAxis(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    attribute_map = {"time_zone": "time_zone"}
    local_timezone = tzlocal.get_localzone()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, obj=None, start=None, stop=None, step=None, rate=None, size=None,
                 create=True, init=True, **kwargs):
        super().__init__(init=False)
        self._timezone = self.local_timezone
        self.is_updating = True

        self._datetimes = None

        if init:
            self.construct(obj, start, stop, step, rate, size, create, **kwargs)

    @property
    def timezone(self):
        try:
            tz_str = self.attributes[self.attribute_map["timezone"]]
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
            self.attributes[self.attribute_map["timezone"]] = tz_str
        except:
            pass

    @property
    def datetimes(self):
        if self._datetimes is None or self.is_updating:
            self._datetimes = tuple(self.as_datetimes())

        return self._datetimes

    # Instance Methods
    # Constructors/Destructors
    def construct(self, obj=None, start=None, stop=None, step=None, rate=None, size=None, create=True, **kwargs):
        super().construct(**kwargs)
        if isinstance(obj, h5py.Dataset):
            self._dataset = obj
        elif isinstance(obj, HDF5Dataset):
            self._dataset = obj._dataset
        elif create:
            if obj is not None:
                self.from_datetimes(obj, **kwargs)
            else:
                self.from_range(start, stop, step, rate, size)



    def from_range(self, start=None, stop=None, step=None, rate=None, size=None, **kwargs):
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

        if step is not None:
            self.require(data=np.arange(start, stop, step), **kwargs)
        else:
            self.require(data=np.linspace(start, stop, size), **kwargs)

    def from_datetimes(self, iter_, **kwargs):
        stamps = np.array([])
        for dt in iter_:
            stamps = np.append(stamps, dt.timestamp())
        self.require(data=stamps, **kwargs)

    def as_datetimes(self, tz=None):
        origin_tz = self.timezone
        if tz is not None and origin_tz is not None:
            return [datetime.datetime.fromtimestamp(t, origin_tz).astimezone(tz) for t in self._dataset]
        else:
            return [datetime.datetime.fromtimestamp(t, origin_tz) for t in self._dataset]
