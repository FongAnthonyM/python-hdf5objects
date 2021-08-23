#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5xltek.py
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

# Downloaded Libraries #
from classversioning import VersionType, TriNumberVersion
from bidict import bidict
import h5py

# Local Libraries #
from ..datasets import TimeSeriesDataset, TimeSeriesMap, ChannelAxisMap, SampleAxisMap, TimeAxisMap
from ..hdf5object import HDF5Map, HDF5Dataset, HDF5Object
from .hdf5eeg import HDF5EEG, HDF5EEGMap


# Definitions #
# Classes #
class XLTEKDataMap(TimeSeriesMap):
    default_attributes = {"sample_rate": "Sampling Rate",
                          "n_samples": "n_samples",
                          "c_axis": "c_axis",
                          "t_axis": "t_axis"}


class HDF5EXLTEKMap(HDF5EEGMap):
    default_attributes = bidict({"file_type": "type",
                                 "file_version": "version",
                                 "subject_id": "subject_id",
                                 "start": "start time",
                                 "end": "end time",
                                 "start_entry": "start entry",
                                 "end_entry": "end entry",
                                 "total_samples": "total samples"})
    default_containers = bidict({"data": "ECoG Array",
                                 "channel_axis": "channel indices",
                                 "sample_axis": "samplestamp axis",
                                 "time_axis": "timestamp axis",
                                 "entry_axis": "entry vector"})
    default_maps = {"data": XLTEKDataMap(name="data"),
                    "channel_axis": ChannelAxisMap(name="channel_axis"),
                    "sample_axis": SampleAxisMap(name="sample_axis"),
                    "time_axis": TimeAxisMap(name="time_axis"),
                    "entry_axis": HDF5Map(name="entry_axis", type_=HDF5Dataset)}


class HDF5XLTEK(HDF5EEG):
    _registration = True
    _VERSION_TYPE = VersionType(name="HDF5XLTEK", class_=TriNumberVersion)
    VERSION = TriNumberVersion(0, 0, 0)
    FILE_TYPE = "XLTEK_EEG"
    default_map = HDF5EXLTEKMap()

    # File Validation
    @classmethod
    def validate_file_type(cls, obj):
        start_name = cls.default_map.attributes["start"]
        end_name = cls.default_map.attributes["end"]

        if isinstance(obj, (str, pathlib.Path)):
            if not isinstance(obj, pathlib.Path):
                obj = pathlib.Path(obj)

            if obj.is_file():
                try:
                    with h5py.File(obj) as obj:
                        return start_name in obj.attrs and end_name in obj.attrs
                except OSError:
                    return False
            else:
                return False
        elif isinstance(obj, HDF5Object):
            obj = obj.h5_fobj
            return start_name in obj.attrs and end_name in obj.attrs

    @classmethod
    def new_validated(cls, obj, **kwargs):
        start_name = cls.default_map.attributes["start"]
        end_name = cls.default_map.attributes["end"]

        if isinstance(obj, (str, pathlib.Path)):
            if not isinstance(obj, pathlib.Path):
                obj = pathlib.Path(obj)

            if obj.is_file():
                try:
                    obj = h5py.File(obj)
                    if start_name in obj.attrs and end_name in obj.attrs:
                        return cls(file=obj, **kwargs)
                except OSError:
                    return None
            else:
                return None
        elif isinstance(obj, HDF5Object):
            obj = obj.h5_fobj
            if start_name in obj.attrs and end_name in obj.attrs:
                return cls(file=obj, **kwargs)

    def __init__(self, file=None, s_id=None, s_dir=None, start=None, init=True, **kwargs):
        super().__init__(init=False)
        self._start_entry = None
        self._end_entry = None
        self._total_samples = 0

        self.entry_axis = None

        if init:
            self.construct(file=file, s_id=s_id, s_dir=s_dir, start=start, **kwargs)

    def construct_file_attributes(self, **kwargs):
        self.attributes[self.map.attributes["total_samples"]] = self._total_samples
        self.attributes[self.map.attributes["start_entry"]] = self._start_entry
        self.attributes[self.map.attributes["end_entry"]] = self._end_entry
        super().construct_file_attributes(**kwargs)

    # Entry Axis
    def create_entry_axis(self, axis=None, **kwargs):
        if axis is None:
            axis = self.eeg_data.t_axis
        if "name" not in kwargs:
            kwargs["name"] = self.map.containers["entry_axis"]

        self.entry_axis = self.map["entry_axis"].create_object(file=self, dtype='i', maxshape=(None, 4), **kwargs)
        self.entry_axis.make_scale("entry axis")
        self.eeg_data.attach_axis(self.entry_axis, axis)
        return self.entry_axis

    def attach_entry_axis(self, dataset, axis=None):
        if axis is None:
            axis = self.eeg_data.t_axis
        self.eeg_data.attach_axis(dataset, axis)
        self.entry_axis = dataset

    def detach_entry_axis(self, axis=None):
        if axis is None:
            axis = self.eeg_data.t_axis
        self.eeg_data.detach_axis(self.entry_axis, axis)
        self.entry_axis = None

    def load_entry_axis(self):
        with self.temp_open():
            if "entry axis" in self.eeg_data.dims[self.eeg_data.t_axis]:
                self.entry_axis = HDF5Dataset(dataset=self.eeg_data.dims[self.eeg_data.t_axis]["entry axis"], file=self)

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

        self.eeg_data = TimeSeriesDataset(init=False)
        self.eeg_data.map.attributes = self.map["data"].attributes.copy()
        self.eeg_data.channel_axis_label = "channel axis"
        self.eeg_data.sample_axis_label = "sample axis"
        self.eeg_data.time_axis_label = "time axis"
        self.eeg_data.construct(name=self.map.containers["data"], file=self, **kwargs)
        self.eeg_data.axis_map["channel_axis"] = self.map.containers["channel_axis"]
        self.eeg_data.axis_map["sample_axis"] = self.map.containers["sample_axis"]
        self.eeg_data.axis_map["time_axis"] = self.map.containers["time_axis"]
        self.structure[self.map.containers["data"]].object = self.eeg_data

    def load_eeg_data(self):
        with self.temp_open():
            d_name = self.map.containers["data"]
            if d_name in self.h5_fobj:
                self.create_eeg_dataset(dataset=self.h5_fobj[d_name], create=False)
                self.load_entry_axis()
            else:
                self.create_eeg_dataset(name=d_name, create=False)
                self.create_entry_axis()

    def set_eeg_data(self, data, sample_rate=None, start_sample=None, end_sample=None,
                     start_time=None, end_time=None, dtype='f4', maxshape=(None, None), **kwargs):
        d_kwargs = self.default_file_attributes.copy()
        d_kwargs.update(kwargs)
        d_name = self.map.containers["data"]
        with self.temp_open():
            if self.eeg_data is None:
                self.create_eeg_dataset(name=d_name, create=False)
                self.create_entry_axis()
            if sample_rate is not None:
                self.eeg_data.sample_rate = sample_rate

            self.eeg_data.set_data(self, data, sample_rate, start_sample, end_sample, start_time, end_time,
                                   axis=None,
                                   dtype=dtype, maxshape=maxshape, **d_kwargs)

    # XLTEK Entry # Todo: Redesign this.
    # def format_entry(self, entry):
    #     data = entry["data"]
    #     n_channels = data.shape[self.eeg_data.c_axis]
    #     n_samples =data.shape[self.time_axis]
    #
    #     channel_axis = np.arange(0, n_channels)
    #     sample_axis = np.arrange(entry["start_sample"], entry["end_sample"])
    #
    #     entry_info = np.zeros((n_samples, 4), dtype=np.int32)
    #     entry_info[:, :] = entry["entry_info"]
    #
    #     time_axis = np.zeros(n_samples, dtype=np.float64)
    #     for sample, i in enumerate(sample_axis):
    #         delta_t = datetime.timedelta(seconds=((sample - entry["snc_sample"]) * 1.0 / entry['sample_rate']))
    #         time = entry["snc_time"] + delta_t
    #         time_axis[i] = time.timestamp()
    #
    #     return data, sample_axis, time_axis, entry_info, channel_axis, entry['sample_rate']
    #
    # def add_entry(self, entry):
    #     data, samples, times, entry_info, channels, sample_rate = self.format_entry(entry)
    #
    #     self.end_entry = entry_info[0]
    #     self.end = times[-1]
    #
    #     self.eeg_data.append_data(data)
    #     self.sample_axis.append(samples)
    #     self.time_axis.append(times)
    #     self.entry_axis.append(entry_info)
    #
    #     self.total_samples = self.eeg_data.n_samples
