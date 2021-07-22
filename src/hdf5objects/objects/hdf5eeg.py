#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5eeg.py
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
from classversioning import VersionType, TriNumberVersion
from bidict import bidict
import numpy as np

# Local Libraries #
from .basehdf5 import BaseHDF5Map, BaseHDF5
from ..datasets import TimeSeriesMap, ChannelAxisMap, SampleAxisMap, TimeAxisMap


# Definitions #
# Classes #
class HDF5EEGMap(BaseHDF5Map):
    default_attributes = bidict({"file_type": "FileType",
                                 "file_version": "FileVersion",
                                 "subject_id": "subject_id",
                                 "start": "start",
                                 "end": "end"})
    default_containers = bidict({"data": "EEG Array",
                                 "channel_axis": "channel_axis",
                                 "sample_axis": "sample_axis",
                                 "time_axis": "time_axis"})
    default_maps = {"data": TimeSeriesMap(name="data"),
                    "channel_axis": ChannelAxisMap(name="channel_axis"),
                    "sample_axis": SampleAxisMap(name="sample_axis"),
                    "time_axis": TimeAxisMap(name="time_axis")}


class HDF5EEG(BaseHDF5):
    _registration = False
    _VERSION_TYPE = VersionType(name="HDF5EEG", class_=TriNumberVersion)
    VERSION = TriNumberVersion(0, 0, 0)
    FILE_TYPE = "EEG"
    default_map = HDF5EEGMap()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, file=None, s_id=None, s_dir=None, start=None, create=True, init=True, **kwargs):
        super().__init__(init=False)
        self._subject_id = ""
        self._subject_dir = None
        self._start = 0.0
        self._end = 0.0
        self._subject_dir = None

        self.eeg_data = None

        if init:
            self.construct(file, s_id, s_dir, start, create, **kwargs)

    @property
    def start(self):
        try:
            self._start = self.attributes[self.map.attributes["start"]]
        finally:
            return datetime.datetime.fromtimestamp(self._start)

    @start.setter
    def start(self, value):
        if isinstance(value, datetime.datetime):
            value = value.timestamp()
        try:
            self.attributes[self.map.attributes["start"]] = value
        finally:
            self._start = value

    @property
    def end(self):
        try:
            self._end = self.attributes[self.map.attributes["end"]]
        finally:
            return datetime.datetime.fromtimestamp(self._end)

    @end.setter
    def end(self, value):
        if isinstance(value, datetime.datetime):
            value = value.timestamp()
        try:
            self.attributes[self.map.attributes["end"]] = value
        finally:
            self._end = value

    @property
    def subject_dir(self):
        """:obj:`Path`: The path to the file.

        The setter casts objects that are not Path to path before setting
        """
        return self._subject_dir

    @subject_dir.setter
    def subject_dir(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._subject_dir = value
        else:
            self._subject_dir = pathlib.Path(value)

    @property
    def sample_rate(self):
        return self.eeg_data.sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self.eeg_data.sample_rate = value

    @property
    def n_samples(self):
        return self.eeg_data.n_samples

    @n_samples.setter
    def n_samples(self, value):
        self.eeg_data.n_samples = value

    @property
    def channel_axis(self):
        return self.eeg_data.channel_axis

    @channel_axis.setter
    def channel_axis(self, value):
        self.eeg_data.channel_axis = value

    @property
    def sample_axis(self):
        return self.eeg_data.sample_axis

    @sample_axis.setter
    def sample_axis(self, value):
        self.eeg_data.sample_axis = value

    @property
    def time_axis(self):
        return self.eeg_data.time_axis

    @time_axis.setter
    def time_axis(self, value):
        self.eeg_data.time_axis = value

    # Representation
    def __repr__(self):
        return repr(self.start)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, file=None, s_id=None, s_dir=None, start=None, create=True, **kwargs):
        """Constructs this object.

        Args:
            obj: An object to build this object from. It can be the path to the file or a File object.
            update (bool): Determines if this object should constantly open the file for updating attributes.
            open_ (bool): Determines if this object will remain open after construction.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if s_dir is not None:
            self.subject_dir = s_dir

        super().construct(obj=file, create=False, **kwargs)

        if self.path is not None:
            self.load_eeg_data()
        else:
            if s_id is not None:
                self._subject_id = s_id
            if isinstance(start, datetime.datetime):
                self._start = start.timestamp()
            elif isinstance(start, float):
                self._start = start

            if create and self._subject_id is not None and self._start is not None and self._subject_dir is not None:
                self.create_file()

    # File Creation/Construction
    def generate_file_name(self):
        return self.subject_id + '_' + self.start.isoformat('_', 'seconds').replace(':', '~') + ".h5"

    def create_file(self, s_name=None, s_dir=None, start=None, f_path=None, **kwargs):
        if s_name is not None:
            self._subject_id = s_name
        if s_dir is not None:
            self.subject_dir = s_dir
        if start is not None:
            self._start = start

        if f_path is None:
            f_path = self.subject_dir.joinpath(self.generate_file_name())

        self.path = f_path

        super().create_file(**kwargs)

    def construct_file_attributes(self, **kwargs):
        self.attributes[self.map.attributes["subject_id"]] = self._subject_id
        self.attributes[self.map.attributes["start"]] = self._start
        self.attributes[self.map.attributes["end"]] = self._end
        super().construct_file_attributes(**kwargs)

    # EEG Data
    def create_eeg_dataset(self, data=None, shape=None, maxshape=None, dtype=None, **kwargs):
        if maxshape is None:
            maxshape = (None, None)

        if data is None:
            if shape is None:
                shape = (0, 0)
            if dtype is None:
                dtype = "f4"

        kwargs["data"] = data
        kwargs["shape"] = shape
        kwargs["maxshape"] = maxshape
        kwargs["dtype"] = dtype

        self.eeg_data = self.map["data"].create_object(name=self.map.containers["data"], file=self, **kwargs)
        self.eeg_data.axis_map["channel_axis"] = self.map.containers["channel_axis"]
        self.eeg_data.axis_map["sample_axis"] = self.map.containers["sample_axis"]
        self.eeg_data.axis_map["time_axis"] = self.map.containers["time_axis"]
        self.structure[self.map.containers["data"]].object = self.eeg_data

    def load_eeg_data(self):
        with self.temp_open():
            d_name = self.map.containers["data"]
            if d_name in self.h5_fobj:
                self.create_eeg_dataset(dataset=self.h5_fobj[d_name])
            else:
                self.create_eeg_dataset(name=d_name, create=False)

    def set_eeg_data(self, data, sample_rate=None, start_sample=None, end_sample=None,
                     start_time=None, end_time=None, dtype='f4', maxshape=(None, None), **kwargs):
        d_kwargs = self.default_file_attributes.copy()
        d_kwargs.update(kwargs)
        d_name = self.map.containers["data"]
        with self.temp_open():
            if self.eeg_data is None:
                self.create_eeg_dataset(name=d_name, create=False)
            if sample_rate is not None:
                self.eeg_data.sample_rate = sample_rate

            self.eeg_data.set_data(self, data, sample_rate, start_sample, end_sample, start_time, end_time,
                                   dtype=dtype, maxshape=maxshape, **d_kwargs)

    # Data Manipulation
    def find_sample(self, sample, aprox=False, tails=False):
        # Setup
        index = None

        # Find
        with self.temp_open():
            if sample in self.sample_axis:
                index = np.where(self.sample_axis[...] == sample)[0][0]
            elif aprox or tails:
                if sample < self.sample_axis[0]:
                    if tails:
                        index = 0
                elif sample > self.sample_axis[-1]:
                    if tails:
                        index = self.sample_axis.shape[0]
                else:
                    index = np.where(self.sample_axis[...] > sample)[0][0] - 1  # Floor to the closest index

            return index, self.sample_axis[index]

    def find_sample_range(self, start=None, end=None, aprox=False, tails=False):
        with self.temp_open():
            start_index, true_start = self.find_sample(start, aprox, tails)
            end_index, true_end = self.find_sample(end, aprox, tails)

            return self.sample_axis[start_index:end_index], true_start, true_end

    def data_range_sample(self, start=None, end=None, aprox=False, tails=False):
        with self.temp_open():
            start_index, true_start = self.find_sample(start, aprox, tails)
            end_index, true_end = self.find_sample(end, aprox, tails)

            return self.eeg_data[start_index:end_index], true_start, true_end

    def find_time(self, timestamp, aprox=False, tails=False):
        # Setup
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()
        index = None

        # Find
        with self.temp_open():
            if timestamp in self.time_axis:
                index = np.where(self.time_axis[...] == timestamp)[0][0]
            elif aprox or tails:
                if timestamp < self.time_axis[0]:
                    if tails:
                        index = 0
                elif timestamp > self.time_axis[-1]:
                    if tails:
                        index = self.time_axis.shape[0]
                else:
                    index = np.where(self.time_axis[...] > timestamp)[0][0] - 1  # Floor to the closest

            return index, self.time_axis[index]

    def find_time_range(self, start=None, end=None, aprox=False, tails=False):
        with self.temp_open():
            start_index, true_start = self.find_time(start, aprox, tails)
            end_index, true_end = self.find_time(end, aprox, tails)

            return self.time_axis[start_index:end_index], true_start, true_end

    def data_range_time(self, start=None, end=None, aprox=False, tails=False):
        with self.temp_open():
            start_index, true_start = self.find_time(start, aprox, tails)
            end_index, true_end = self.find_time(end, aprox, tails)

            return self.eeg_data[start_index:end_index], true_start, true_end
