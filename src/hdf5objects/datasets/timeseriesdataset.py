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
from .axes import ChannelAxisMap, ChannelAxis
from .axes import SampleAxisMap, SampleAxis
from .axes import TimeAxisMap, TimeAxis


# Definitions #
# Classes #
class TimeSeriesMap(HDF5Map):
    default_attribute_names = {"sample_rate": "samplerate",
                               "n_samples": "n_samples",
                               "c_axis": "c_axis",
                               "t_axis": "t_axis"}
    default_attributes = {"n_samples": 0,
                          "c_axis": 1,
                          "t_axis": 0}
    default_map_names = {"channel_axis": "channel_axis",
                         "sample_axis": "sample_axis",
                         "time_axis": "time_axis"}
    default_maps = {"channel_axis": ChannelAxisMap(),
                    "sample_axis": SampleAxisMap(),
                    "time_axis": TimeAxisMap()}


class TimeSeriesDataset(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_map = TimeSeriesMap()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, data=None, sample_rate=None, channels=None, samples=None, timestamps=None,
                 load=False, build=False, init=True, **kwargs):
        super().__init__(init=False)

        self._channel_axis = None
        self._sample_axis = None
        self._time_axis = None

        self.channel_scale_name = "channel axis"
        self.sample_scale_name = "sample axis"
        self.time_scale_name = "time axis"

        if init:
            self.construct(data=data, sample_rate=sample_rate, channels=channels, samples=samples, 
                           timestamps=timestamps, load=load, build=build, **kwargs)

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
    
    @property
    def channel_axis(self):
        if self._channel_axis is None:
            self.load_axes()
        return self._channel_axis

    @property
    def sample_axis(self):
        if self._sample_axis is None:
            self.load_axes()
        return self._sample_axis

    @property
    def time_axis(self):
        if self._time_axis is None:
            self.load_axes()
        return self._time_axis
    
    # Instance Methods
    # Constructors/Destructors
    def construct(self, data=None, sample_rate=None, channels=None, samples=None, timestamps=None,
                  dataset=None,  name: str = None, map_: HDF5Map = None, file=None,
                  load=False, build=False, parent: str = None, **kwargs):
        if data is not None:
            kwargs["data"] = data
            kwargs["build"] = build

        super().construct(dataset=dataset, name=name, map_=map_, file=file, parent=parent)

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]
        
        if load and self.exists:
            self.load()

        if build and data is not None:
            self.construct_dataset(data=data, channels=channels, samples=samples, timestamps=timestamps, **kwargs)
    
    def construct_channel_axis(self, channels=None):
        if isinstance(channels, ChannelAxis):
            self.attach_channel_axis(channels)
        elif self._channel_axis is None:
            if channels is None and self.exists:
                kwargs = {"start": 0,}
            elif isinstance(channels, dict):
                kwargs = channels
            else:
                kwargs = {}
            self.create_channel_axis(**kwargs)
            if channels is not None and not isinstance(channels, dict):
                self._channel_axis.set_data(data=channels)
        elif channels is not None:
            if isinstance(channels, dict):
                self._channel_axis.from_range(**channels)
            else:
                self._channel_axis.set_data(data=channels)
                
    def construct_sample_axis(self, samples=None):
        if isinstance(samples, ChannelAxis):
            self.attach_sample_axis(samples)
        elif self._sample_axis is None:
            if samples is None and self.exists:
                kwargs = {"start": 0}
            elif isinstance(samples, dict):
                kwargs = samples
            else:
                kwargs = {}
            self.create_sample_axis(**kwargs)
            if samples is not None and not isinstance(samples, dict):
                self._sample_axis.set_data(data=samples)
        elif samples is not None:
            if isinstance(samples, dict):
                self._sample_axis.from_range(**samples)
            else:
                self._sample_axis.set_data(data=samples)
                
    def construct_time_axis(self, timestamps=None):
        if isinstance(timestamps, TimeAxis):
            self.attach_time_axis(timestamps)
        elif self._time_axis is None:
            kwargs = timestamps if isinstance(timestamps, dict) else {}
            self.create_time_axis(**kwargs)
            if timestamps is not None and not isinstance(timestamps, dict):
                self._time_axis.set_data(data=timestamps)
        elif timestamps is not None:
            if isinstance(timestamps, dict):
                self._time_axis.from_range(**timestamps)
            else:
                self._time_axis.set_data(data=timestamps)
    
    def construct_axes(self, channels=None, samples=None, timestamps=None):
        self.construct_channel_axis(channels=channels)
        self.construct_sample_axis(samples=samples)
        self.construct_time_axis(timestamps=timestamps)

    # File
    def create(self, data=None, sample_rate=None, start=None, channels=None, samples=None, timestamps=None, **kwargs):
        super().create(data=data, **kwargs)

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]

        if timestamps is None and start is not None:
            timestamps = {"start": start, "rate": self.sample_rate, "size": self.n_samples}

        self.construct_axes(channels=channels, samples=samples, timestamps=timestamps)
        return self

    def require(self, data=None, sample_rate=None, start=None, channels=None, samples=None, timestamps=None, **kwargs):
        existed = self.exists
        super().require(data=data, **kwargs)

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]

        if existed:
            self.load_axes()
        else:
            if timestamps is None and start is not None:
                timestamps = {"start": start, "rate": self.sample_rate, "size": self.n_samples}

            self.construct_axes(channels=channels, samples=samples, timestamps=timestamps)
        return self

    def set_data(self, data=None, sample_rate=None, channels=None, samples=None, timestamps=None, **kwargs):
        if not self.exists:
            self.require(data=data, sample_rate=sample_rate,
                         channels=channels, samples=samples, timestamps=timestamps, **kwargs)
        else:
            self.replace_data(data=data)

            if sample_rate is not None:
                self.sample_rate = sample_rate

            if data is not None:
                self.n_samples = data.shape[self.t_axis]

            self.construct_axes(channels=channels, samples=samples, timestamps=timestamps)

    def load(self):
        super().load()
        self.load_axes()

    def standardize_attributes(self):
        self.attributes["n_samples"] = self.get_shape[self.t_axis]

    # Axes
    def create_channel_axis(self, start=None, stop=None, step=1, rate=None, size=None, axis=None, **kwargs):
        if axis is None:
            axis = self.c_axis
        if size is None:
            size = self.shape[axis]
        if "name" not in kwargs:
            kwargs["name"] = self._full_name + "_" + self.map.map_names["channel_axis"]
        
        self._channel_axis = self.map["channel_axis"].type(start=start, stop=stop, step=step, rate=rate, size=size, 
                                                           s_name=self.channel_scale_name, build=True, file=self._file, 
                                                           **kwargs)
        self.attach_axis(self._channel_axis, axis)

    def attach_channel_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.c_axis
        self.attach_axis(dataset, axis)
        self._channel_axis = dataset
        self.channel_scale_name = getattr(dataset, "scale_name", None)

    def detach_channel_axis(self, axis=None):
        if axis is None:
            axis = self.c_axis
        self.detach_axis(self.channel_axis, axis)
        self._channel_axis = None

    def create_sample_axis(self, start=None, stop=None, step=1, rate=None, size=None, axis=None, **kwargs):
        if axis is None:
            axis = self.t_axis
        if size is None:
            size = self.shape[axis]
        if rate is None:
            rate = self.sample_rate
        if "name" not in kwargs:
            kwargs["name"] = self._full_name + "_" + self.map.map_names["sample_axis"]

        self._sample_axis = self.map["sample_axis"].type(start=start, stop=stop, step=step, rate=rate, size=size,
                                                         s_name=self.sample_scale_name, build=True, file=self._file,
                                                         **kwargs)
        self.attach_axis(self._sample_axis, axis)

    def attach_sample_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.t_axis
        self.attach_axis(dataset, axis)
        self._sample_axis = dataset
        self.sample_scale_name = getattr(dataset, "scale_name", None)

    def detach_sample_axis(self, axis=None):
        if axis is None:
            axis = self.t_axis
        self.detach_axis(self._sample_axis, axis)
        self._sample_axis = None

    def create_time_axis(self, start=None, stop=None, step=None, rate=None, size=None, axis=None, **kwargs):
        if axis is None:
            axis = self.t_axis
        if size is None:
            size = self.n_samples
        if rate is None:
            rate = self.sample_rate
        if "name" not in kwargs:
            kwargs["name"] = self._full_name + "_" + self.map.map_names["time_axis"]

        self._time_axis = self.map["time_axis"].type(start=start, stop=stop, step=step, rate=rate, size=size,
                                                     s_name=self.time_scale_name, build=True, file=self._file,
                                                     **kwargs)
        self.attach_axis(self._time_axis, axis)

    def attach_time_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.t_axis
        self.attach_axis(dataset, axis)
        self._time_axis = dataset
        self.time_scale_name = getattr(dataset, "scale_name", None)

    def detach_time_axis(self, axis=None):
        if axis is None:
            axis = self.t_axis
        self.detach_axis(self._time_axis, axis)
        self._time_axis = None

    def load_axes(self):
        with self:
            if self.channel_scale_name in self._dataset.dims[self.c_axis]:
                dataset = self._dataset.dims[self.c_axis][self.channel_scale_name]
                self._channel_axis = self.map["channel_axis"].type(dataset=dataset, s_name=self.channel_scale_name,
                                                                   file=self._file)

            if self.sample_scale_name in self._dataset.dims[self.t_axis]:
                dataset = self._dataset.dims[self.t_axis][self.sample_scale_name]
                self._sample_axis = self.map["sample_axis"].type(dataset=dataset, s_name=self.sample_scale_name,
                                                                 file=self._file)

            if self.time_scale_name in self._dataset.dims[self.t_axis]:
                dataset = self._dataset.dims[self.t_axis][self.time_scale_name]
                self._time_axis = self.map["time_axis"].type(dataset=dataset, s_name=self.time_scale_name,
                                                             file=self._file)

    # Axis Getters
    def get_datetime(self, index):
        return self.time_axis.get_datetime(index)

    def get_datetime_range(self, start=None, stop=None, step=None):
        return self.time_axis.get_datetime_range(start=start, stop=stop, step=step)

    def get_timestamp_range(self, start=None, stop=None, step=None):
        return self.time_axis.get_timestamp_range(start=start, stop=stop, step=step)

    def get_time_intervals(self, start=None, stop=None, step=None):
        return self.time_axis.get_intervals(start=start, stop=stop, step=step)

    # Get Range
    def get_timestamp_range_time(self, start=None, stop=None, step=None, aprox=False, tails=False):
        return self.time_axis.get_timstamp_range_time(start=start, stop=stop, step=step, aprox=aprox, tails=tails)

    def get_datetime_range_time(self, start=None, stop=None, step=None, aprox=False, tails=False):
        return self.time_axis.get_datetime_range_time(start=start, stop=stop, step=step, aprox=aprox, tails=tails)

    def get_data_range_sample(self, start=None, stop=None, step=None, aprox=False, tails=False):
        start_sample, true_start = self.sample_axis.find_index(index=start, aprox=aprox, tails=tails)
        stop_sample, true_stop = self.sample_axis.find_index(index=stop, aprox=aprox, tails=tails)

        with self:
            return self._dataset[slice(start=start_sample, stop=stop_sample, step=step)], true_start, true_stop

    def get_data_range_time(self, start=None, stop=None, step=None, aprox=False, tails=False):
        start_sample, true_start = self.time_axis.find_time_index(timestamp=start, aprox=aprox, tails=tails)
        stop_sample, true_stop = self.time_axis.find_time_index(timestamp=stop, aprox=aprox, tails=tails)

        with self:
            return self._dataset[slice(start=start_sample, stop=stop_sample, step=step)], true_start, true_stop

    # Manipulation
    def shift_samples(self, shift, start=None, stop=None, step=None):
        self.sample_axis.shift(shift=shift, start=start, stop=stop, step=step)

    def shift_timestamps(self, shift, start=None, stop=None, step=None):
        self.time_axis.shift(shift=shift, start=start, stop=stop, step=step)


# Assign Cyclic Definitions
TimeSeriesMap.default_type = TimeSeriesDataset
TimeSeriesDataset.default_map = TimeSeriesMap()
