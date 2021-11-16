#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" timeseriesdataset.py
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

# Third-Party Packages #
import numpy as np

# Local Packages #
from ..hdf5objects import HDF5Map, HDF5Dataset
from .axes import ChannelAxis
from .axes import SampleAxis
from .axes import TimeAxis


# Definitions #
# Classes #
class TimeSeriesMap(HDF5Map):
    default_attribute_names = {"sample_rate": "samplerate",
                               "n_samples": "n_samples",
                               "c_axis": "c_axis",
                               "t_axis": "t_axis"}
    default_attributes = {"sample_rate": 0,
                          "n_samples": 0,
                          "c_axis": 1,
                          "t_axis": 0}
    default_map_names = {"channel_axis": "channel_axis",
                         "sample_axis": "sample_axis",
                         "time_axis": "time_axis"}
    default_maps = {"channel_axis": ChannelAxis,
                    "sample_axis": SampleAxis,
                    "time_axis": TimeAxis}


class TimeSeriesDataset(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_map = TimeSeriesMap()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, data=None, sample_rate=None, create=True, init=True, **kwargs):
        super().__init__(init=False)

        self.channel_axis = None
        self.sample_axis = None
        self.time_axis = None

        self.channel_scale_name = "channels"
        self.sample_scale_name = "samples"
        self.time_scale_name = "timestamps"

        if init:
            self.construct(data, sample_rate, create, **kwargs)

    @property
    def sample_rate(self):
        return self.attributes["sample_rate"]

    @sample_rate.setter
    def sample_rate(self, value):
        self.attributes.set_attribute("sample_rate", value)

    @property
    def n_samples(self):
        return self.attributes["n_samples"]

    @n_samples.setter
    def n_samples(self, value):
        self.attributes.set_attribute("n_samples", value)

    @property
    def c_axis(self):
        return self.attributes["c_axis"]
    
    @c_axis.setter
    def c_axis(self, value):
        self.attributes.set_attribute("c_axis", value)

    @property
    def t_axis(self):
        return self.attributes["t_axis"]

    @t_axis.setter
    def t_axis(self, value):
        self.attributes.set_attribute("t_axis", value)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, data=None, sample_rate=None, channels=None, samples=None, timestamps=None,
                  build=True, **kwargs):
        if data is not None:
            kwargs["data"] = data
            kwargs["build"] = build

        super().construct(**kwargs)

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]

        if build:
            self.construct_axes(channels=channels, samples=samples, timestamps=timestamps)

    def construct_axes(self, channels=None, samples=None, timestamps=None):
        if channels is None and self.channel_axis is None:
            self.create_channel_axis(0, self.shape[self.c_axis])
        else:
            self.attach_sample_axis(channels)

        if samples is None and self.sample_axis is None:
            self.create_sample_axis(0, self.shape[self.t_axis])
        else:
            self.attach_sample_axis(samples)

        if timestamps is not None and self.timestamps is None:
            self.attach_time_axis(timestamps)

    # File
    def load(self):
        self.load_axes()

    # Axes
    def create_channel_axis(self, start=None, stop=None, step=1, rate=None, size=None, axis=None, **kwargs):
        if axis is None:
            axis = self.c_axis
        if size is None:
            size = self._dataset[self.c_axis]
        if "name" not in kwargs:
            kwargs["name"] = self._full_name + "_" + self.map.map_names["channel_axis"]
        
        self.channel_axis = self.map["channel_axis"](start=start, stop=stop, step=step, rate=rate, size=size, 
                                                     s_name=self.channel_scale_name, build=True, file=self._file, 
                                                     **kwargs)
        self.attach_axis(self.channel_axis, axis)

    def attach_channel_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.c_axis
        self.attach_axis(dataset, axis)
        self.channel_axis = dataset
        self.channel_scale_name = getattr(dataset, "scale_name", None)

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
            kwargs["name"] = self._full_name + "_" + self.map.map_names["sample_axis"]

        self.sample_axis = self.map["sample_axis"](start=start, stop=stop, step=step, rate=rate, size=size,
                                                   s_name=self.channel_scale_name, build=True, file=self._file, 
                                                   **kwargs)
        self.attach_axis(self.sample_axis, axis)

    def attach_sample_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.t_axis
        self.attach_axis(dataset, axis)
        self.sample_axis = dataset
        self.sample_scale_name = getattr(dataset, "scale_name", None)

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
            kwargs["name"] = self._full_name + "_" + self.map.map_names["time_axis"]

        self.time_axis = self.map["time_axis"](start=start, stop=stop, step=step, rate=rate, size=size,
                                               s_name=self.channel_scale_name, build=True, file=self._file, **kwargs)
        self.attach_axis(self.time_axis, axis)

    def attach_time_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.t_axis
        self.attach_axis(dataset, axis)
        self.time_axis = dataset
        self.time_scale_name = getattr(dataset, "scale_name", None)

    def detach_time_axis(self, axis=None):
        if axis is None:
            axis = self.t_axis
        self.detach_axis(self.time_axis, axis)
        self.time_axis = None

    def load_axes(self):
        with self:
            if self.channel_axis_label in self._dataset.dims[self.c_axis]:
                dataset = self._dataset.dims[self.c_axis][self.channel_axis_label]
                self.channel_axis = self.map["channel_axis"](dataset=dataset, s_name=self.channel_scale_name, 
                                                             file=self._file)

            if self.sample_axis_label in self._dataset.dims[self.t_axis]:
                dataset = self._dataset.dims[self.t_axis][self.sample_axis_label]
                self.sample_axis = self.map["sample_axis"](dataset=dataset, s_name=self.sample_scale_name,
                                                           file=self._file)

            if self.time_axis_label in self._dataset.dims[self.t_axis]:
                dataset = self._dataset.dims[self.t_axis][self.time_axis_label]
                self.time_axis = self.map["time_axis"](dataset=dataset, s_name=self.time_scale_name,
                                                       file=self._file)

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
