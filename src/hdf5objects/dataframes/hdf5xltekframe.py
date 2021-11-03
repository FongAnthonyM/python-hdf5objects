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
import h5py
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
            if isinstance(file, h5py.File):
                self.file = file
            else:
                self.file = h5py.File(file, mode=mode)
        elif s_id is not None or s_dir is not None or start is not None or kwargs:
            path = pathlib.Path()
            self.file = h5py.File(path, mode=self.mode)

        super().construct(file=None)

    def open(self, mode=None, **kwargs):
        if mode is None:
            mode = self.mode
        return self

    # File
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def load_data(self):
        return self.file["ECoG Array"]

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_time_axis(self):
        return self.file["timestamp vector"][...]

    # Getters
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_shape(self):
        return self.file["ECoG Array"].shape

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_start(self):
        return datetime.datetime.fromtimestamp(self.file.attrs["start time"])

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_end(self):
        return datetime.datetime.fromtimestamp(self.file.attrs["end time"])

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_sample_rate(self):
        return self.file["ECoG Array"].attrs["Sampling Rate"]
