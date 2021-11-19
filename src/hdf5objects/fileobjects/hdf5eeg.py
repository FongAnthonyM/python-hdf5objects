#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5eeg.py
Description:
"""
# Package Header #
from ..__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
import pathlib
import datetime

# Third-Party Packages #
from classversioning import VersionType, TriNumberVersion
import numpy as np

# Local Packages #
from .basehdf5 import BaseHDF5Map, BaseHDF5
from ..datasets import TimeSeriesMap, ChannelAxisMap, SampleAxisMap, TimeAxisMap


# Definitions #
# Classes #
class HDF5EEGMap(BaseHDF5Map):
    default_attribute_names = {"file_type": "FileType",
                               "file_version": "FileVersion",
                               "subject_id": "subject_id",
                               "start": "start",
                               "end": "end"}
    default_map_names = {"data": "EEG Array"}
    default_maps = {"data": TimeSeriesMap()}


class HDF5EEG(BaseHDF5):
    _registration = False
    _VERSION_TYPE = VersionType(name="HDF5EEG", class_=TriNumberVersion)
    VERSION = TriNumberVersion(0, 0, 0)
    FILE_TYPE = "EEG"
    default_map = HDF5EEGMap()

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, file=None, s_id=None, s_dir=None, start=None, init=True, **kwargs):
        super().__init__(init=False)
        self._subject_id = ""
        self._subject_dir = None

        if init:
            self.construct(file=file, s_id=s_id, s_dir=s_dir, start=start, **kwargs)

    @property
    def subject_id(self):
        return self.attributes["subject_id"]

    @subject_id.setter
    def subject_id(self, value):
        self.attributes.set_attribute("subject_id", value)
        self._subject_id = value

    @property
    def start(self):
        return self.attributes["start"]

    @start.setter
    def start(self, value):
        self.attributes.set_attribute("start", value)

    @property
    def end(self):
        return self.attributes["end"]

    @end.setter
    def end(self, value):
        self.attributes.set_attribute("end", value)

    @property
    def subject_dir(self):
        """:obj:`Path`: The path to the file.

        The setter casts fileobjects that are not Path to path before setting
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
        return self["data"].sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self["data"].sample_rate = value

    @property
    def n_samples(self):
        return self["data"].n_samples

    @property
    def channel_axis(self):
        return self["data"].channel_axis

    @property
    def sample_axis(self):
        return self["data"].sample_axis

    @property
    def time_axis(self):
        return self["data"].time_axis

    @property
    def data(self):
        return self["data"]

    # Representation
    def __hash__(self):
        """Overrides hash to make the class hashable.

        Returns:
            The system ID of the class.
        """
        return id(self)

    # Comparison
    def __eq__(self, other):
        if isinstance(other, HDF5EEG):
            return self.start == other.start
        else:
            return self.start == other

    def __ne__(self, other):
        if isinstance(other, HDF5EEG):
            return self.start != other.start
        else:
            return self.start != other

    def __lt__(self, other):
        if isinstance(other, HDF5EEG):
            return self.start < other.start
        else:
            return self.start < other

    def __gt__(self, other):
        if isinstance(other, HDF5EEG):
            return self.start > other.start
        else:
            return self.start > other

    def __le__(self, other):
        if isinstance(other, HDF5EEG):
            return self.start <= other.start
        else:
            return self.start <= other

    def __ge__(self, other):
        if isinstance(other, HDF5EEG):
            return self.start >= other.start
        else:
            return self.start >= other

    # Instance Methods
    # Constructors/Destructors
    def construct(self, file=None, s_id=None, s_dir=None, start=None, **kwargs):
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

        if s_id is not None:
            self._subject_id = s_id

        if file is None and self.path is None and start is not None:
            self.path = self.subject_dir.joinpath(self.generate_file_name(s_id=s_id, start=start))

        super().construct(file=file, **kwargs)

        return self

    def construct_file_attributes(self, start=None):
        super().construct_file_attributes()
        if isinstance(start, datetime.datetime):
            self.attributes["start"] = start.timestamp()
        elif isinstance(start, float):
            self.attributes["start"] = start
        self.attributes["subject_id"] = self._subject_id

    def construct_dataset(self, load=False, build=False, **kwargs):
        self._group_.get_member(name="data", load=load, build=build, **kwargs)

    # File
    def generate_file_name(self, s_id=None, start=None):
        if s_id is None:
            s_id = self.subject_id

        if start is None:
            start = self.start

        if isinstance(start, float):
            start = datetime.datetime.fromtimestamp(start)

        return s_id + '_' + start.isoformat('_', 'seconds').replace(':', '~') + ".h5"

    def create_file(self, name=None, s_id=None, s_dir=None, start=None, **kwargs):
        if s_id is not None:
            self._subject_id = s_id
        if s_dir is not None:
            self.subject_dir = s_dir

        if name is None and self.path is None and start is not None:
            self.path = self.subject_dir.joinpath(self.generate_file_name(s_id=s_id, start=start))

        super().create_file(name=name, **kwargs)

    # Attributes Modification
    def validate_attributes(self):
        return self.start == self.data._time_axis.start and self.end == self.data._time_axis.end

    def standardize_attributes(self):
        if self.data.exists:
            self.data.standardize_attributes()
            self.start = self.data._time_axis.start
            self.end = self.data._time_axis.end

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

            return self.data[start_index:end_index], true_start, true_end

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

            return self.data[start_index:end_index], true_start, true_end
