#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5xltekframe.py
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
import pathlib

# Downloaded Libraries #
from ...studies import FileTimeFrame
from multipledispatch import dispatch

# Local Libraries #
from ..objects import HDF5XLTEK


# Definitions #
# Classes #
class HDF5XLTEKFrame(FileTimeFrame):
    file_type = HDF5XLTEK
    default_data_container = None

    # Magic Methods
    # Construction/Destruction
    def __init__(self, file=None, s_id=None, s_dir=None, start=None, init=True, **kwargs):
        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(file=file, s_id=s_id, s_dir=s_dir, start=start, **kwargs)

    @property
    def date(self):
        if self.start is None:
            return self._date
        else:
            self.start.date()

    # Instance Methods
    # Constructors/Destructors
    def construct(self, file=None, s_id=None, s_dir=None, start=None, **kwargs):
        super().construct(file=None)

        if file is not None:
            self.set_file(file, s_id=s_id, s_dir=s_dir, start=start, **kwargs)
        elif s_id is not None or s_dir is not None or start is not None or kwargs:
            self.file = self.file_type(s_id=s_id, s_dir=s_dir, start=start, **kwargs)

    # File
    @dispatch(object)
    def set_file(self, file, **kwargs):
        if isinstance(file, self.file_type):
            self.file = file
        else:
            raise ValueError("file must be a path, File, or HDF5Object")

    @dispatch((str, pathlib.Path))
    def set_file(self, file, **kwargs):
        self.file = self.file_type(file=file, *kwargs)

    def load_data(self):
        self._data = self.file.eeg_data

    def load_time_axis(self):
        self._time_axis = self.file.eeg_data

    # Getters
    def get_start(self):
        self._start = self.file.eeg_data.start

        return self._start

    def get_end(self):
        self._end = None

        return self._end

    def get_time_axis(self):
        return self.time_axis[...]

    def get_sample_rate(self):
        self._sample_rate = 1
        return self._sample_rate

    def get_sample_period(self):
        self._sample_period = 1 / self.get_sample_rate()
        return self._sample_period

    def get_is_continuous(self):
        self._is_continuous = self.validate_continuous()
        return self._is_continuous

    # Setters
    def set_data(self, value):
        if self.mode == 'r':
            raise IOError("not writable")
        self.data.replace_data(value)

    def set_time_axis(self, value):
        if self.mode == 'r':
            raise IOError("not writable")
        self.time_axis.replace_data(value)

    # Data
    def get_range(self, start=None, stop=None, step=None):
        pass

    def get_times(self, start=None, stop=None, step=None):
        pass  # return self.times[slice(start, stop, step)]

    # Find
    def find_time_index(self, timestamp, aprox=False, tails=False):
        pass

    def find_time_sample(self, timestamp, aprox=False, tails=False):
        pass

    # Shape
    def validate_shape(self):
        pass

    def change_size(self, shape=None, **kwargs):
        pass

