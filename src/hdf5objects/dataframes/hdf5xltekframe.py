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
from baseobjects.cachingtools import timed_keyless_cache_method
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
    def __init__(self, file=None, s_id=None, s_dir=None, start=None, mode='r', init=True, **kwargs):
        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(file=file, s_id=s_id, s_dir=s_dir, start=start, mode=mode, **kwargs)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, file=None, s_id=None, s_dir=None, start=None, mode=None, **kwargs):
        if mode is not None:
            self.mode = mode

        if file is not None:
            self.set_file(file, s_id=s_id, s_dir=s_dir, start=start, mode=self.mode, **kwargs)
        elif s_id is not None or s_dir is not None or start is not None or kwargs:
            self.file = self.file_type(s_id=s_id, s_dir=s_dir, start=start, mode=self.mode, **kwargs)

        super().construct(file=None)

    # File
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def load_data(self):
        return self.file.eeg_data

    # Getters
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_shape(self):
        return self.file["ECoG Array"].shape

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_start(self):
        return self.file.time_axis.start_datetime

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_end(self):
        return self.file.time_axis.end_datetime

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_time_axis(self):
        return self.file.time_axis[...]

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_sample_rate(self):
        return self.file.eeg_data.sample_rate
