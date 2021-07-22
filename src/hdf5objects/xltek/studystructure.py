#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" studystructure.py
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
from baseobjects import BaseObject
import numpy as np

# Local Libraries #
from .eegframe import EEGFrame
from src.hdf5objects.studies.dataframe import DataFrame
from ..objects.hdf5xltek import HDF5XLTEK


# Definitions #
# Classes #
class StructureDataclass(BaseObject):
    # Magic Methods
    # Construction/Destruction
    def __init__(self):
        self.start = None
        self.end = None

        self.structure = None


class DirectoryFrame(DataFrame):
    frame_type = None

    # Magic Methods
    # Construction/Destruction
    def __init__(self, name, name_dir, date=None, path=None, init=True):
        super().__init__(init=False)
        self._path = None
        self._date = None
        self.date_format = "%Y-%m-%d"

        self.name = name
        self.glob_condition = None

        self.is_updating_all = False
        self.is_updating_last = True

        self.start = None
        self.end = None
        self.start_sample = None
        self.end_sample = None

        self.name_dir = name_dir

        self.frame_names = set()
        self.times = []

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

    @property
    def date(self):
        if self.start is None:
            return self._date
        else:
            self.start.date()

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

    def frame_sort_key(self, frame):
        return frame.start

    def frame_condition(self, path):
        return True

    def require_path(self):
        if not self.path.is_dir():
            self.path.mkdir()

    def get_start_timestamps(self):
        starts = np.empty(len(self.frames))
        for index, frame in self.frames:
            starts[index] = frame.start.timestamp()
        return starts

    def get_end_timestamps(self):
        ends = np.empty(len(self.frames))
        for index, frame in self.frames:
            ends[index] = frame.end.timestamp()
        return ends

    def date_from_path(self):
        date_string = self.path.parts[-1].split('_')[1]
        self._date = datetime.datetime.strptime(date_string, self.date_format).date()

    def get_time_info(self):
        if len(self.frames) > 0:
            self.start = self.frames[0].start
            self.end = self.frames[-1].end
            self.start_sample = self.frames[0].start_sample
            self.end_sample = self.frames[-1].end_sample
            self._date = self.start.date()
        return self.date, self.start, self.end

    def new_file(self, entry):
        file = HDF5XLTEK(self.name, path=self.path, entry=entry)
        self.frames.append(file)
        return file

    def find_frame_time(self, timestamp, tails=False):
        # Setup
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()
        index = None
        times = self.get_start_timestamps()

        if timestamp < times[0]:
            if tails:
                index = 0
        elif timestamp > times[-1]:
            if tails:
                index = times.shape[0]
        elif timestamp in times:
            index = np.where(times == timestamp)[0][0]
        else:
            index = np.where(times > timestamp)[0][0] - 1

        return index

    def find_time_index(self, timestamp, aprox=False, tails=False):
        index = self.find_frame_time(timestamp, tails)

        if index:
            frame = self.frames[index]
            if timestamp <= frame.end.timestamp():
                location = frame.find_time_index(timestamp=timestamp, aprox=aprox)
            else:
                index = None

        return [index] + location

    def find_time_sample(self, timestamp, aprox=False, tails=False):
        index = self.find_frame_time(timestamp, tails)
        frame_samples = sum(self.lengths[:index])
        inner_samples = 0

        if index:
            frame = self.frames[index]
            if timestamp <= frame.end.timestamp():
                inner_samples = frame.find_time_sample(timestamp=timestamp, aprox=aprox)
            else:
                frame_samples = -1

        return frame_samples + inner_samples
















    def find_sample(self, s):
        for i, file in enumerate(self.frames):
            index = file.find_sample(s)
            if index is not None:
                return file, i, index
        return None, None, None


    def data_range_sample(self, s=None, e=None, tails=False):
        collect = False
        stop = False
        start = 0
        end = -1
        s_file = None
        s_fi = None
        e_file = None
        e_fi = None
        d_array = []

        # Allow nonspecific start sample and before scope option
        if s is None or (tails and s < self.frames[0].start_sample):
            s_file = self.frames[0]
            s_fi = 0
            collect = True

        # Search for data and append to list
        for i, file in enumerate(self.frames):
            with file as f:
                if s is not None and f.start_sample <= s <= f.end_sample:
                    s_file = file
                    s_fi = i
                    start = f.find_sample(s)
                    collect = True
                if e is not None and f.start_sample <= e <= f.end_sample:
                    e_file = file
                    e_fi = i
                    end = f.find_sample(e)
                    stop = True
                if collect:
                    d_array.append(f.data[start:end])
            if stop:
                break

        # Allow nonspecific end sample and after scope option
        if e is None or (tails and e > self.frames[-1].end_sample):
            e_file = self.frames[-1]
            e_fi = -1

        return d_array, [s_file, s_fi, start], [e_file, e_fi, end]

    def data_range_time(self, s=None, e=None, rnd=False, tails=False, frame=False):
        collect = False
        stop = False
        start = 0
        end = -1
        last_time = None
        s_file = None
        s_fi = None
        e_file = None
        e_fi = None
        f_array = []
        d_array = []

        # Allow nonspecific start time and before scope option
        if s is None or (tails and s < self.frames[0].start):
            s_file = self.frames[0]
            s_fi = 0
            collect = True

        # Search for data and append to list
        for i, file in enumerate(self.frames):
            with file as f:
                d_frame = file
                st = 0
                if s is not None:
                    if f.start <= s <= f.end:
                        s_file = file
                        s_fi = i
                        start = f.find_time(s)
                        st = start
                        d_frame = f.data[st:end]
                        collect = True
                    elif rnd and last_time is not None and last_time < s < f.start:
                        s_file = file
                        s_fi = i
                        start = 0
                        if not frame:
                            st = 0
                            d_frame = f.data[st:end]
                        collect = True
                if e is not None:
                    if f.start <= e <= f.end:
                        e_file = file
                        e_fi = i
                        end = f.find_time(e)
                        d_frame = f.data[st:end]
                        stop = True
                    elif rnd and last_time is not None and last_time < e < f.start:
                        e_file = file
                        e_fi = i
                        end = 0
                        if not frame:
                            d_frame = f.data[st:end]
                        stop = True
                if collect:
                    if frame:
                        f_array.append(d_frame)
                    else:
                        d_array.append(f.data[st:end])
                last_time = f.end
            if stop:
                break

        # Allow nonspecific end time and after scope option
        if e is None or (tails and e > self.frames[-1].end):
            e_file = self.frames[-1]
            e_fi = -1

        if frame:
            d_array = EEGFrame(f_array)
        return d_array, [s_fi, start], [e_fi, end]


