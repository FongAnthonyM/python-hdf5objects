#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5events.py
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
import time
import uuid

# Downloaded Libraries #
from bidict import bidict
import h5py
import numpy as np


# Local Libraries #
from .basehdf5 import BaseHDF5, HDF5hierarchicalDatasets


# Todo: Adapt this to new style
# Definitions #
# Classes #
class HDF5eventLogger(BaseHDF5):
    FILE_TYPE = "EventLog"
    VERSION = "0.0.1"
    EVENT_FIELDS = bidict(Time=0, DeltaTime=1, StartTime=2, Type=3)
    TIME_NAME = "Time"
    DELTA_NAME = "DeltaTime"
    START_NAME = "StartTime"
    TYPE_NAME = "Type"
    LINK_NAME = "LinkID"
    EVENT_DTYPE = np.dtype([(TIME_NAME, np.float),
                            (DELTA_NAME, np.float),
                            (START_NAME, np.float),
                            (TYPE_NAME, h5py.string_dtype(encoding="utf-8")),
                            (LINK_NAME, h5py.string_dtype(encoding="utf-8"))])

    # Instantiation/Destruction
    def __init__(self, path=None, io_trigger=None, init=False):
        super().__init__(path=path)

        self.default_child_kwargs = {}
        self.event_types = {}
        self.start_datetime = None
        self.start_time_counter = None
        self.start_time_offset = None
        self.hierarchy = None
        if io_trigger is None:
            self.io_trigger = AudioTrigger()
        else:
            self.io_trigger = io_trigger

        if init:
            self.construct()

    # Container Magic Methods
    def __len__(self):
        return self.Events.len()

    def __getitem__(self, item):
        if isinstance(item, str):
            return super().__getitem__(item)
        elif isinstance(item, int) or isinstance(item, slice):
            return self.get_item((item, 0))
        else:
            return self.get_item(item)

    # Representations
    def __repr__(self):
        return repr(self.start_datetime)

    # Constructors
    def construct(self, open_=False, **kwargs):
        super().construct(open_=open_, **kwargs)
        self.hierarchy = HDF5hierarchicalDatasets(h5_container=self, dataset=self.Events, name="Events",
                                                  child_name=self.TYPE_NAME, link_name=self.LINK_NAME)

    def create_file(self, open_=False):
        super().create_file(open_=open_)
        self.create_event_dataset(name="Events", dtype=self.EVENT_DTYPE)

    # Datasets
    def create_event_dataset(self, name, dtype=None, data=None, **kwargs):
        if data is not None:
            m = data.shape[0]
            n = data.shape[1]
        else:
            m = 0
            n = 1
        defaults = {"shape": (m, n), "dtype": dtype, "maxshape": (None, n)}
        args = merge_dict(defaults, kwargs)
        return self.create_dataset(name=name, data=data, **args)

    # Sequence Methods
    def get_item(self, item):
        return self.hierarchy.get_items(item)

    def append(self, type_, **kwargs):
        if isinstance(type_, dict):
            self.append_event(type_)
        else:
            event = self.create_event(type_=type_, **kwargs)
            self.append_event(event)

    def insert(self, i, type_, **kwargs):
        if isinstance(type_, dict):
            super().insert(i, type_)
        else:
            event = self.create_event(type_=type_, **kwargs)
            super().insert(i, event)

    def clear(self):
        self.start_datetime = None
        self.start_time_counter = None
        self._path = None
        super().clear()

    # User Event Methods
    def create_event(self,  type_, **kwargs):
        seconds = self.start_time_offset + round(time.perf_counter() - self.start_time_counter, 6)
        now = self.start_datetime + datetime.timedelta(seconds=seconds)
        return {"Time": now, "DeltaTime": seconds, "StartTime": self.start_datetime, self.TYPE_NAME: type_, **kwargs}

    def append_event(self, event, axis=0, child_kwargs=None):
        child_name = event[self.TYPE_NAME]
        if child_name not in self.hierarchy.child_datasets:
            child_event = event.copy()
            for field in self.EVENT_FIELDS.keys():
                if field in child_event:
                    child_event.pop(field)
            if self.LINK_NAME not in child_event:
                child_event[self.LINK_NAME] = str(uuid.uuid4())
            child_dtype = self.event2dtype(child_event)
            if child_kwargs is None:
                child_kwargs = self.default_child_kwargs
            child_dataset = self.create_event_dataset(child_name, dtype=child_dtype, **child_kwargs)
            self.hierarchy.add_child_dataset(child_name, child_dataset)
        self.hierarchy.append_item(event, (child_name,), axis)

    def set_time(self):
        self.start_datetime = datetime.datetime.now()
        self.start_time_counter = time.perf_counter()
        self.start_time_offset = 0
        self.append({"Time": self.start_datetime, "DeltaTime": 0, "StartTime": self.start_datetime, self.TYPE_NAME: "TimeSet"})

    def resume_time(self, name=None, index=None):
        now_datatime = datetime.datetime.now()
        self.start_time_counter = time.perf_counter()
        if name is None:
            name = "TimeSet"
        if index is None:
            index = -1
        start_event = self.hierarchy.get_item(index, name)
        self.start_datetime = datetime.datetime.fromtimestamp(start_event[self.TIME_NAME])
        self.start_time_offset = (now_datatime - self.start_datetime).total_seconds()
        self.append({"Time": now_datatime, "DeltaTime": self.start_time_offset,
                     "StartTime": self.start_datetime, self.TYPE_NAME: "ResumeTime"})

    def get_event_type(self, name, id_info=False):
        return self.hierarchy.get_dataset(name, id_info)

    # Event Querying
    def find_event(self, time_, type_=None, bisect_="bisect"):
        if isinstance(time_, datetime.datetime):
            time_stamp = time_.timestamp()
        else:
            time_stamp = time_

        if type_ is None or type_ == "Events":
            events = self
            times = self.Events[self.TIME_NAME]
        else:
            events = self.hierarchy.get_dataset(name=type_)
            times = [e[self.TIME_NAME] for e in events]

        index = bisect.bisect_left(times, time_stamp)
        if index >= len(times):
            index -= 1
        elif times[index] == time_:
            pass
        elif bisect_ == "bisect":
            if index > 0:
                index -= 1 - (np.abs(times[index-1:index+1]-time_)).argmin()
        elif bisect_ == "left":
            if index > 0:
                index -= 1
        elif bisect_ == "right":
            pass
        else:
            return -1, None

        return index, events[index]

    def find_event_range(self, start, end, type_=None):
        if type_ is None or type_ == "Events":
            events = self
        else:
            events = self.hierarchy.get_dataset(name=type_)

        first, _ = self.find_event(start, type_=type_, bisect_="right")
        last, _ = self.find_event(end, type_=type_, bisect_="left")

        return range(first, last+1), events[first:last+1]

    # Trigger Methods
    def trigger(self):
        self.io_trigger.trigger()

    def trigger_event(self, **kwargs):
        self.trigger()
        self.append(type_="Trigger", **kwargs)

    # Static Methods
    @staticmethod
    def event2dtype(event):
        dtypes = []
        for key, value in event.items():
            if isinstance(value, int):
                dtype = np.int
            elif isinstance(value, float):
                dtype = np.float
            elif isinstance(value, datetime.datetime):
                dtype = np.float
            else:
                dtype = h5py.string_dtype(encoding="utf-8")

            dtypes.append((key, dtype))
        return dtypes


class SubjectEventLogger(HDF5eventLogger):
    FILE_TYPE = "SubjectEventLog"
    VERSION = "0.0.1"
    SUBJECT_NAME = "Subject"
    EXPERIMENT_NAME = "Task"
    EXPERIMENT_NUMBER = "Block"

    # Instantiation/Destruction
    def __init__(self, path=None, subject="", x_name="", x_number="", io_trigger=None, init=False):
        super().__init__(path, io_trigger)
        self._subject = subject
        self._experiment_name = x_name
        self._experiment_number = x_number

        if init:
            self.construct()

    # Constructors
    def construct(self, open_=False, **kwargs):
        super().construct(open_=open_, **kwargs)

    def create_file(self, open_=False):
        super().create_file(open_=open_)
        self.add_file_attributes({self.SUBJECT_NAME: self._subject, self.EXPERIMENT_NAME: self._experiment_name,
                                  self.EXPERIMENT_NUMBER: self._experiment_number})
