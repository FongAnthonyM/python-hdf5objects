#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" directorytimeframe.py
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
import pathlib
import datetime

# Downloaded Libraries #
import numpy as np

# Local Libraries #
from .timeseriesframe import TimeSeriesFrame


# Definitions #
# Classes #
class DirectoryTimeFrame(TimeSeriesFrame):
    default_return_frame_type = TimeSeriesFrame
    default_frame_type = None

    # Magic Methods
    # Construction/Destruction
    def __init__(self, name, name_dir, date=None, path=None, init=True):
        super().__init__(init=False)
        self._path = None

        self.name = name
        self.glob_condition = None

        self.is_updating_all = False
        self.is_updating_last = True

        self.name_dir = name_dir

        self.frame_names = set()

        if init:
            self.construct()

    @property
    def name_dir(self):
        return self._name_dir

    @name_dir.setter
    def name_dir(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._name_dir = value
        else:
            self._name_dir = pathlib.Path(value)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, **kwargs):
        super().construct()

        self.construct_frames()
        if self.frames:
            self.get_time_info()
        else:
            try:
                self.date_from_path()
            except (ValueError, IndexError):
                pass

    def construct_frames(self):
        for path in self.path.glob(self.glob_condition):
            if path not in self.frame_names:
                if self.frame_condition(path):
                    self.frames.append(self.frame_type(self.path))
                    self.frame_names.add(path)
        self.frames.sort(key=lambda frame: frame.start)

    # File
    def frame_condition(self, path):
        return True

    def require_path(self):
        if not self.path.is_dir():
            self.path.mkdir()

    def date_from_path(self):
        date_string = self.path.parts[-1].split('_')[1]
        self._date = datetime.datetime.strptime(date_string, self.date_format).date()

    def new_file(self, entry):
        file = HDF5XLTEK(self.name, path=self.path, entry=entry)
        self.frames.append(file)
        return file

