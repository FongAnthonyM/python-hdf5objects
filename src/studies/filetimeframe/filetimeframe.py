#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" filetimeframe.py
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
import pathlib

# Downloaded Libraries #

# Local Libraries #
from ..timeseriesframe import TimeSeriesContainer
from ..directorytimeframe import DirectoryTimeFrameInterface


# Definitions #
# Classes #
class FileTimeFrame(TimeSeriesContainer, DirectoryTimeFrameInterface):
    file_type = None
    default_editable_type = TimeSeriesContainer

    # Class Methods #
    @classmethod
    @abstractmethod
    def validate_path(cls, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        return path.is_file()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, file=None, frames=None, init=True):
        # Overriding Property Attributes #
        self._start = None
        self._end = None
        self._sample_rate = None

        self._time_axis = None
        self._data = None

        # Parent Attributes #
        super().__init__(init=False)
        self.is_updating = False

        # New Attributes #
        # Containers #
        self.file = None

        # Object Construction #
        if init:
            self.construct(file=file, frames=frames)

    @property
    def data(self):
        if self._data is None or (self.is_updating and not self._cache):
            return self.load_data()
        else:
            return self._data

    @data.setter
    def data(self, value):
        if value is not None:
            self.set_data(value)

    @property
    def time_axis(self):
        if self._time_axis is None or (self.is_updating and not self._cache):
            return self.load_time_axis()
        else:
            return self._time_axis

    @time_axis.setter
    def time_axis(self, value):
        if value is not None:
            self.set_time_axis(value)

    @property
    def shape(self):
        return self.data.shape

    @property
    def start(self):
        if self._start is None or (self.is_updating and not self._cache):
            return self.get_start()
        else:
            return self._start

    @property
    def end(self):
        if self._end is None or (self.is_updating and not self._cache):
            return self.get_end()
        else:
            return self._end

    @property
    def sample_rate(self):
        if self._sample_rate is None or (self.is_updating and not self._cache):
            return self.get_sample_rate()
        else:
            return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value is not None:
            raise AttributeError("can't set attribute")

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
    def construct(self, file=None, frames=None, **kwargs):
        # New Assignment
        if file is not None:
            self.set_file(file)

        # Parent Construction
        super().construct(frames=frames)

    # Cache and Memory
    def refresh(self):
        self.load_data()
        self.load_time_axis()
        self.get_start()
        self.get_end()
        self.get_sample_rate()
        self.get_sample_period()
        self.get_is_continuous()

    # File
    def set_file(self, file):
        if isinstance(file, self.file_type):
            self.file = file
        else:
            raise ValueError("file must be a path or str")

    def open(self, mode='a', **kwargs):
        self.file.open(mode, **kwargs)
        self.refresh()
        return self

    def close(self):
        self.file.close()

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def load_time_axis(self):
        pass

    # Getters
    def get_start(self):
        self._start = self.file.start
        return self._start

    def get_end(self):
        self._end = self.file.end
        return self._end

    def get_time_axis(self):
        return self.time_axis[...]

    def get_sample_rate(self):
        self._sample_rate = self.file.sample_rate
        return self._sample_rate

    def get_sample_period(self):
        self._sample_period = 1 / self.get_sample_rate()
        return self._sample_period

    def get_is_continuous(self):
        self._is_continuous = self.validate_continuous()
        return self._is_continuous

    # Setters
    @abstractmethod
    def set_data(self, value):
        if self.mode == 'r':
            raise IOError("not writable")

    @abstractmethod
    def set_time_axis(self, value):
        if self.mode == 'r':
            raise IOError("not writable")
