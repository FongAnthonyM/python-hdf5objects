#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5xltekstudy.py
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
import os
import pathlib
import configparser
import datetime

# Downloaded Libraries #
import numpy as np

# Local Libraries #
from .eegframe import EEGFrame
from ..objects.hdf5xltek import HDF5XLTEK


# Configurations
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')
if 'DEFAULT' in config and 'TYPE' in config['DEFAULT'] and 'HDF5 XLTEK Study' == config['DEFAULT']['TYPE']:
    studies_dir = config['Studies Directory']
    if studies_dir.getboolean('OSVARIABLE'):
        studies_path = pathlib.Path(os.environ[studies_dir['VARIABLENAME']])
    else:
        studies_path = pathlib.Path(studies_dir['PATH'])
else:
    studies_path = pathlib.Path.cwd()


# Definitions #
# Classes #
class HDF5XLTEKstudy:
    def __init__(self, name, path=None, spath=studies_path):
        self.name = name
        self._studies_path = None
        self.studies_path = spath
        self._path = None
        if path is None:
            self.path = pathlib.Path(self.studies_path, self.name)
        else:
            self.path = path

        self.days = None

        self.build()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['reader'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def studies_path(self):
        return self._studies_path

    @studies_path.setter
    def studies_path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._studies_path = value
        else:
            self._studies_path = pathlib.Path(value)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    def build(self):
        if not self.path.is_dir():
            print('No such subject directory as %s' % (str(self.path)))
            ans = input('Would you like to create a new subject directory? (y/N): ')
            if ans.upper() == 'Y' or ans.upper() == 'Yes':
                self.path.mkdir()
        else:
            self.days = [DayDirectory(self.name, self.path, path=p) for p in self.path.glob(self.name+'_*') if p.is_dir()]
            self.days.sort()

    def make_directories(self, start, end):
        days = end - start
        for i in range(0, days.days + 1):
            day = start + datetime.timedelta(days=i)
            self.days.append(DayDirectory(self.name, self.path, date=day))

    def find_sample(self, s):
        for i, day in enumerate(self.days):
            file, f, index = day.find_sample(s)
            if file is not None:
                return day, file, i, f, index
        return None, None, None, None, None

    def find_time(self, dt, rnd=False, tails=False):
        f_date = dt.date()
        last_time = None

        if tails:
            if f_date < self.days[0].date:
                return self.days[0], 0, 0, 0, 0
            elif f_date > self.days[-1].date:
                return self.days[-1], -1, -1, -1, -1

        for i, day in enumerate(self.days):
            s_date = day.date
            if s_date == f_date:
                file, f, index = day.find_time(dt, rnd=rnd, tails=tails)
                return day, file, i, f, index
            elif rnd and last_time is not None and last_time < f_date < s_date:
                return day, day.files[0], i, 0, 0
            last_time = s_date

        return None, None, None, None, None

    def data_range(self, s=None, e=None, rnd=False, tails=False, frame=False, separate=False):
        if isinstance(s, datetime.datetime) or isinstance(e, datetime.datetime):
            return self.data_range_time(s, e, rnd, tails, frame, separate)
        else:
            return self.data_range_sample(s, e, separate)

    def data_range_sample(self, s=None, e=None, separate=False):
        f_pass = True
        primary = None
        final = None
        data_array = []
        for i, day in enumerate(self.days):
            if e is not None and e <= day.start_sample:
                break

            if s is None or s <= day.end_sample:
                data, first, last = day.data_range_sample(s, e, tails=True)
                if separate:
                    data_array.append(data)
                else:
                    data_array += data
                if f_pass:
                    if separate:
                        primary = [i] + first
                    else:
                        primary = first
                    f_pass = False

                if separate:
                    final = [i] + last
                else:
                    final = last

        return data_array, primary, final

    def data_range_time(self, s=None, e=None, rnd=False, tails=False, frame=False, separate=False):
        f_pass = True
        primary = None
        final = None
        if frame and not separate:
            data_array = EEGFrame()
        else:
            data_array = None

        # Allow nonspecific start time and before scope option
        if s is None or (tails and s < self.days[0].files[0].start):
            s = self.days[0].files[0].start
        # Allow nonspecific end time and after scope option
        if e is None or (tails and e > self.days[-1].files[-1].end):
            e = self.days[-1].files[-1].end

        for i, day in enumerate(self.days):
            if e <= day.start:
                break

            if s <= day.end:
                data, first, last = day.data_range_time(s, e, rnd=True, tails=True, frame=frame)
                if separate:
                    data_array.append(data)
                elif data_array is None:
                    data_array = data
                else:
                    data_array = np.concatenate([data_array, data], axis=0)
                if f_pass:
                    if separate:
                        primary = [i] + first
                    else:
                        primary = first
                    f_pass = False

                if separate:
                    final = [i] + last
                else:
                    final = last

        return data_array, primary, final

    # Todo: Finish making this Method
    # def export_range_as_file(self, s=None, e=None, rnd=False, tails=False, frame=False, separate=False):
    #     d, f, l, = self.data_range(s, e, rnd, tails, frame, separate)


class DayDirectory:
    def __init__(self, name, name_dir, date=None, path=None):
        self.name = name
        self.date = date
        self.start = None
        self.end = None
        self.start_sample = None
        self.end_sample = None
        self.name_dir = name_dir
        if path is None:
            self.path = pathlib.Path(self.name_dir, name+'_'+str(date))
        else:
            self.path = path

        self.files = []

        self.build()
        if date is None:
            self.get_time_info()

    def __eq__(self, other):
        try:
            return self.start == other.start
        except AttributeError:
            return self.start == other

    def __ne__(self, other):
        try:
            return self.start != other.start
        except AttributeError:
            return self.start != other

    def __le__(self, other):
        try:
            return self.start <= other.start
        except AttributeError:
            return self.start <= other

    def __lt__(self, other):
        try:
            return self.start < other.start
        except AttributeError:
            return self.start < other

    def __ge__(self, other):
        try:
            return self.start >= other.start
        except AttributeError:
            return self.start >= other

    def __gt__(self, other):
        try:
            return self.start > other.start
        except AttributeError:
            return self.start > other

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

    def build(self):
        if self.path.is_dir():
            self.files = [HDF5XLTEK(self.name, path=p) for p in self.path.glob('*.h5')]
            self.files.sort(key=lambda study: study.start)
        else:
            self.path.mkdir()

    def get_time_info(self):
        if len(self.files) > 0:
            start = self.files[0].start
            end = self.files[-1].end
            start_sample = self.files[0].start_sample
            end_sample = self.files[-1].end_sample
            date = start.date()
        else:
            start = None
            end = None
            start_sample = None
            end_sample = None
            date_string = self.path.parts[-1].split('_')[1]
            date = datetime.datetime.strptime(date_string, '%Y-%m-%d').date()
        self.start = start
        self.end = end
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.date = date
        return date, start, end

    def new_file(self, entry):
        file = HDF5XLTEK(self.name, path=self.path, entry=entry)
        self.files.append(file)
        return file

    def find_sample(self, s):
        for i, file in enumerate(self.files):
            index = file.find_sample(s)
            if index is not None:
                return file, i, index
        return None, None, None

    def find_time(self, dt, rnd=False, tails=False):
        last_time = None

        if tails:
            if dt < self.files[0].start:
                return self.files[0], 0, 0,
            elif dt > self.files[-1].end:
                return self.files[-1], -1, -1

        for i, file in enumerate(self.files):
            if file.start <= dt <= file.end:
                return file, i, file.find_time(dt)
            elif rnd and last_time is not None and last_time < dt < file.start:
                return file, i, 0
            last_time = file.end
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
        if s is None or (tails and s < self.files[0].start_sample):
            s_file = self.files[0]
            s_fi = 0
            collect = True

        # Search for data and append to list
        for i, file in enumerate(self.files):
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
        if e is None or (tails and e > self.files[-1].end_sample):
            e_file = self.files[-1]
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
        d_array = None

        # Allow nonspecific start time and before scope option
        if s is None or (tails and s < self.files[0].start):
            s_file = self.files[0]
            s_fi = 0
            collect = True

        # Search for data and append to list
        for i, file in enumerate(self.files):
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
                    elif d_array is None:
                        d_array = f.data[st:end]
                    else:
                        d_array = np.concatenate([d_array, f.data[st:end]], axis=0)
                last_time = f.end
            if stop:
                break

        # Allow nonspecific end time and after scope option
        if e is None or (tails and e > self.files[-1].end):
            e_file = self.files[-1]
            e_fi = -1

        if frame:
            d_array = EEGFrame(f_array)
        return d_array, [s_fi, start], [e_fi, end]
