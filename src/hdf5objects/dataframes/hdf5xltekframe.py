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
from multipledispatch import dispatch

# Local Libraries #
from ..objects import HDF5XLTEK
from .hdf5baseframe import HDF5BaseFrame


# Definitions #
# Classes #
class HDF5XLTEKFrame(HDF5BaseFrame):
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

    # Instance Methods
    # Constructors/Destructors
    def construct(self, file=None, s_id=None, s_dir=None, start=None, **kwargs):
        if file is not None:
            self.set_file(file, s_id=s_id, s_dir=s_dir, start=start, **kwargs)
        elif s_id is not None or s_dir is not None or start is not None or kwargs:
            self.file = self.file_type(s_id=s_id, s_dir=s_dir, start=start, **kwargs)

        super().construct(file=None)

    # File
    def load_data(self):
        self._data = self.file.eeg_data
        return self._data

    def load_time_axis(self):
        self._time_axis = self.file.time_axis
        return self._time_axis

    # Getters
    def get_start(self):
        self._start = self.file.time_axis.start_datetime
        return self._start

    def get_end(self):
        self._end = self.file.time_axis.end_datetime
        return self._end

    def get_sample_rate(self):
        self._sample_rate = self.file.eeg_data.sample_rate
        return self._sample_rate

