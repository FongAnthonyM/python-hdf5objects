#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" timeseriesdataset.py
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

# Downloaded Libraries #
from bidict import bidict
import numpy as np

# Local Libraries #
from ..hdf5object import HDF5Map, HDF5Dataset
from .channelaxis import ChannelAxis
from .sampleaxis import SampleAxis
from .timeaxis import TimeAxis


# Definitions #
# Classes #
class TimeSeriesMap(HDF5Map):
    default_attributes = bidict({"sample_rate": "samplerate",
                                 "n_samples": "n_samples",
                                 "c_axis": "c_axis",
                                 "t_axis": "t_axis"})


class TimeSeriesDataset(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    _axis_map = {"channel_axis": "channel_axis",
                 "sample_axis": "sample_axis",
                 "time_axis": "time_axis"}
    default_map = TimeSeriesMap()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, data=None, sample_rate=None, create=True, init=True, **kwargs):
        super().__init__(init=False)
        self._sample_rate = 0
        self._n_samples = 0
        self._c_axis = 1
        self._t_axis = 0
        self.axis_map = self._axis_map.copy()

        self.channel_axis = None
        self.sample_axis = None
        self.time_axis = None

        self.channel_axis_label = "channels"
        self.sample_axis_label = "samples"
        self.time_axis_label = "timestamps"

        if init:
            self.construct(data, sample_rate, create, **kwargs)

    @property
    def sample_rate(self):
        try:
            self._sample_rate = self.attributes[self.map.attributes["sample_rate"]]
        finally:
            return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        try:
            self.attributes[self.map.attributes["sample_rate"]] = value
        except:
            pass

    @property
    def n_samples(self):
        try:
            self._n_samples = self.attributes[self.map.attributes["n_samples"]]
        finally:
            return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        try:
            self.attributes[self.map.attributes["n_samples"]] = value
        except:
            pass

    @property
    def c_axis(self):
        try:
            self._c_axis = self.attributes[self.map.attributes["c_axis"]]
        finally:
            return self._c_axis

    @c_axis.setter
    def c_axis(self, value):
        try:
            self.attributes[self.map.attributes["c_axis"]] = value
        except:
            pass

    @property
    def t_axis(self):
        try:
            self._t_axis = self.attributes[self.map.attributes["t_axis"]]
        finally:
            return self._t_axis

    @t_axis.setter
    def t_axis(self, value):
        try:
            self.attributes[self.map.attributes["t_axis"]] = value
        except:
            pass

    # Instance Methods
    # Constructors/Destructors
    def construct(self, data=None, sample_rate=None, start_sample=None, end_sample=None, start_time=None, end_time=None,
                  channels=None, create=True, **kwargs):
        super().construct(create=create, data=data, **kwargs)
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]

        self.load_axes()

        if self.sample_axis is None and start_sample is not None and end_sample is not None:
            self.create_sample_axis(start_sample, end_sample)

        if self.time_axis is None and start_time is not None and end_time is not None:
            self.create_time_axis(start_time, end_time)

        if channels is not None:
            if isinstance(channels, bool) and channels:
                self.create_channel_axis(0, self.n_samples)
            else:
                self.attach_sample_axis(channels)

    # Axes
    def create_channel_axis(self, start=None, stop=None, step=1, size=None, axis=None, **kwargs):
        if axis is None:
            axis = self.c_axis
        if size is None:
            size = self._dataset[self.c_axis]
        if "name" not in kwargs:
            kwargs["name"] = self._dataset.parent + self.axis_map["channel_axis"]

        self.channel_axis = ChannelAxis(start=start, stop=stop, step=step, size=size, file=self._file, **kwargs)
        self.channel_axis_label = self.channel_axis_label
        self.attach_axis(self.channel_axis, axis)

    def attach_channel_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.c_axis
        self.attach_axis(dataset, axis)
        self.channel_axis = dataset

    def detach_channel_axis(self, axis=None):
        if axis is None:
            axis = self.c_axis
        self.detach_axis(self.channel_axis, axis)
        self.channel_axis = None

    def create_sample_axis(self, start=None, stop=None, step=1, rate=None, size=None, axis=None, **kwargs):
        if axis is None:
            axis = self.t_axis
        if size is None:
            size = self.n_samples
        if rate is None:
            rate = self.sample_rate
        if "name" not in kwargs:
            kwargs["name"] = self._dataset.parent + self.axis_map["sample_axis"]

        self.sample_axis = SampleAxis(start=start, stop=stop, step=step,
                                      rate=rate, size=size, file=self._file, **kwargs)
        self.sample_axis_label = self.sample_axis_label
        self.attach_axis(self.sample_axis, axis)

    def attach_sample_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.t_axis
        self.attach_axis(dataset, axis)
        self.sample_axis = dataset

    def detach_sample_axis(self, axis=None):
        if axis is None:
            axis = self.t_axis
        self.detach_axis(self.sample_axis, axis)
        self.sample_axis = None

    def create_time_axis(self, start=None, stop=None, step=None, rate=None, size=None, axis=None, **kwargs):
        if axis is None:
            axis = self.t_axis
        if size is None:
            size = self.n_samples
        if rate is None:
            rate = self.sample_rate
        if "name" not in kwargs:
            kwargs["name"] = self._dataset.parent + self.axis_map["time_axis"]

        self.time_axis = TimeAxis(start=start, stop=stop, step=step, rate=rate, size=size, file=self._file, **kwargs)
        self.time_axis.label = self.time_axis_label
        self.attach_axis(self.time_axis, axis)

    def attach_time_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.t_axis
        self.attach_axis(dataset, axis)
        self.time_axis = dataset

    def detach_time_axis(self, axis=None):
        if axis is None:
            axis = self.t_axis
        self.detach_axis(self.time_axis, axis)
        self.time_axis = None

    def load_axes(self):
        with self:
            if self.channel_axis_label in self._dataset.dims[self.c_axis]:
                self.channel_axis = ChannelAxis(dataset=self._dataset.dims[self.c_axis][self.channel_axis_label],
                                                file=self._file)
                self.channel_axis.label = self.channel_axis_label

            if self.sample_axis_label in self._dataset.dims[self.t_axis]:
                self.sample_axis = SampleAxis(dataset=self._dataset.dims[self.t_axis][self.sample_axis_label],
                                              file=self._file)
                self.sample_axis.label = self.sample_axis_label

            if self.time_axis_label in self._dataset.dims[self.t_axis]:
                self.time_axis = TimeAxis(dataset=self._dataset.dims[self.t_axis][self.time_axis_label],
                                          file=self._file)
                self.time_axis.label = self.time_axis_label

    # Data
    def set_data(self, data, sample_rate=None, start_sample=None, end_sample=None,
                 start_time=None, end_time=None, channels=None, **kwargs):
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]

        self.require(data=data, **kwargs)

        if start_sample is not None and end_sample is not None:
            if self.sample_axis is not None:
                self.detach_sample_axis()
            self.create_sample_axis(start_sample, end_sample)

        if start_time is not None and end_time is not None:
            if self.time_axis is not None:
                self.detach_time_axis()
            self.create_time_axis(start_time, end_time)

        if channels:
            self.attach_sample_axis(channels)
        else:
            self.create_channel_axis(0, self.n_samples)

    def append_data(self, data):
        self.append(data, axis=self.t_axis)


# Assign Cyclic Definitions
TimeSeriesMap.default_type = TimeSeriesDataset
TimeSeriesDataset.default_map = TimeSeriesMap()
