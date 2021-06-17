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
import h5py

# Local Libraries #


# Todo: Inherit from the AdvanceHDF5 object
# Todo: Make one big HDF5 Package
# Definitions #
# Classes #
class HDF5EEG:
    structure = {'type': 'type',
                 'name': 'name',
                 'ID': 'ID',
                 'start': 'start',
                 'end': 'end',
                 'data': 'EEG Array',
                 'samplerate': 'samplerate',
                 'nsamples': 'nsamples',
                 'saxis': 'saxis',
                 'taxis': 'taxis'}

    def __init__(self, name, ID=None, path=None, init=False):
        self._type = 'EEG'
        self._name = name
        self._ID = ID
        self._path = None
        self.path = path
        self.is_open = False

        self._start = None
        self._end = None

        self._data = None
        self._start_sample = None
        self._end_sample = None
        self._sample_rate = None
        self._n_samples = None
        self._s_axis = None
        self._t_axis = None
        self._dt_axis = None

        self.h5_fobj = None
        
        self.cargs = {'compression': 'gzip', 'compression_opts': 4}
        self.is_updating = False

        if self.path.is_file():
            self.open()
            if not init:
                self.close()
        elif init:
            self.build()

    def __repr__(self):
        return repr(self.start)

    def __getstate__(self):
        state = self.__dict__.copy()
        name = str(state['path'])
        open = state['is_open']
        if open:
            fobj = state['h5_fobj']
            fobj.flush()
            fobj.close()
        state['h5_fobj'] = (name, open)
        return state

    def __setstate__(self, state):
        name, open = state['h5_fobj']
        state['h5_fobj'] = h5py.File(str(name), 'r+')
        if not open:
            state['h5_fobj'].close()
        self.__dict__.is_updating(state)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        op = self.is_open
        self.open()

        data = self.data[item]

        if not op:
            self.close()
        return data

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    @property
    def type(self):
        op = self.is_open
        self.open()

        if self.structure['type'] in self.h5_fobj.attrs and (self._type is None or self.is_updating):
            self._type = self.h5_fobj.attrs[self.structure['type']]

        if not op:
            self.close()

        return self._type

    @property
    def name(self):
        op = self.is_open
        self.open()

        if self.structure['name'] in self.h5_fobj.attrs and (self._name is None or self.is_updating):
            self._name = self.h5_fobj.attrs[self.structure['name']]

        if not op:
            self.close()

        return self._name

    @property
    def ID(self):
        op = self.is_open
        self.open()

        if self.structure['ID'] in self.h5_fobj.attrs and (self._ID is None or self.is_updating):
            self._ID = self.h5_fobj.attrs[self.structure['ID']]

        if not op:
            self.close()

        return self._ID

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
    def start(self):
        op = self.is_open
        self.open()

        if self.structure['start'] in self.h5_fobj.attrs and (self._start is None or self.is_updating):
            self._start = datetime.datetime.fromtimestamp(self.h5_fobj.attrs[self.structure['start']])

        if not op:
            self.close()

        return self._start

    @property
    def end(self):
        op = self.is_open
        self.open()

        if self.structure['end'] in self.h5_fobj.attrs and (self._end is None or self.is_updating):
            self._end = datetime.datetime.fromtimestamp(self.h5_fobj.attrs[self.structure['end']])

        if not op:
            self.close()

        return self._end

    @property
    def data(self):
        op = self.is_open
        self.open()

        if self.structure['data'] in self.h5_fobj:
            self._data = self.h5_fobj[self.structure['data']]

        if not op:
            self.close()

        return self._data

    @property
    def start_sample(self):
        op = self.is_open
        self.open()

        if self.structure['saxis'] in self.h5_fobj and (self._start_sample is None or self.is_updating):
            self._start_sample = self.h5_fobj[self.structure['saxis']][0]

        if not op:
            self.close()

        return self._start_sample

    @property
    def end_sample(self):
        op = self.is_open
        self.open()

        if self.structure['saxis'] in self.h5_fobj and (self._end_sample is None or self.is_updating):
            self._end_sample = self.h5_fobj[self.structure['saxis']][-1]

        if not op:
            self.close()

        return self._end_sample

    @property
    def sample_rate(self):
        op = self.is_open
        self.open()

        if self.structure['data'] in self.h5_fobj and \
           self.structure['samplerate'] in self.h5_fobj[self.structure['data']].attrs and \
           (self._sample_rate is None or self.is_updating):
            self._sample_rate = self.h5_fobj[self.structure['data']].attrs[self.structure['samplerate']]

        if not op:
            self.close()

        return self._sample_rate

    @property
    def n_samples(self):
        op = self.is_open
        self.open()

        if self.structure['data'] in self.h5_fobj and \
           self.structure['nsamples'] in self.h5_fobj[self.structure['data']].attrs and \
           (self._n_samples is None or self.is_updating):
            self._n_samples = self.h5_fobj[self.structure['data']].attrs[self.structure['nsamples']]

        if not op:
            self.close()

        return self._n_samples

    @property
    def s_axis(self):
        op = self.is_open
        self.open()

        if self.structure['saxis'] in self.h5_fobj and (self._s_axis is None or self.is_updating):
            self._s_axis = self.h5_fobj[self.structure['saxis']][...]

        if not op:
            self.close()

        return self._s_axis

    @property
    def t_axis(self):
        op = self.is_open
        self.open()

        if self.structure['taxis'] in self.h5_fobj and (self._t_axis is None or self.is_updating):
            self._t_axis = self.h5_fobj[self.structure['taxis']][...]

        if not op:
            self.close()

        return self._t_axis

    @property
    def dt_axis(self):
        op = self.is_open
        self.open()

        if self.structure['taxis'] in self.h5_fobj and (self._dt_axis is None or self.is_updating):
            t_axis = self.h5_fobj[self.structure['taxis']][...]
            self._dt_axis = [datetime.datetime.fromtimestamp(t) for t in t_axis]

        if not op:
            self.close()

        return self._dt_axis

    def build(self, start=None, data=None, saxis=None, taxis=None):
        if self.path.is_dir():
            if start is None:
                f_name = self.name
            else:
                f_name = self.name + '_' + start.isoformat('_', 'seconds').replace(':', '~')
            self.path = pathlib.Path(self.path, f_name + '.h5')

        self.h5_fobj = h5py.File(str(self.path))

        self.h5_fobj.attrs[self.structure['type']] = self._type
        self.h5_fobj.attrs[self.structure['name']] = self._name
        self.h5_fobj.attrs[self.structure['ID']] = self._ID
        self.h5_fobj.attrs[self.structure['start']] = 0
        self.h5_fobj.attrs[self.structure['end']] = 0

        ecog = self.h5_fobj.create_dataset(self.structure['data'], dtype='f32', data=data, maxshape=(None, None), **self.cargs)
        ecog.attrs[self.structure['samplerate']] = 0
        ecog.attrs[self.structure['nsamples']] = 0

        sstamps = self.h5_fobj.create_dataset(self.structure['saxis'], dtype='i', data=saxis, maxshape=(None,), **self.cargs)
        tstamps = self.h5_fobj.create_dataset(self.structure['taxis'], dtype='f8', data=taxis, maxshape=(None,), **self.cargs)

        ecog.dims.create_scale(sstamps, 'sample axis')
        ecog.dims.create_scale(tstamps, 'time axis')

        ecog.dims[0].attach_scale(sstamps)
        ecog.dims[0].attach_scale(tstamps)

        self.h5_fobj.flush()

        return self.h5_fobj

    def open(self, exc=False):
        if not self.is_open:
            try:
                self.h5_fobj = h5py.File(str(self.path))
            except:
                if exc:
                    self.is_open = False
                else:
                    raise
            else:
                self.is_open = True
        return self.h5_fobj

    def close(self):
        if self.is_open:
            self.h5_fobj.flush()
            self.h5_fobj.close()
            self.is_open = False
        return not self.is_open

    def find_sample(self, s):
        op = self.is_open
        self.open()

        s_axis = self.s_axis
        if s in s_axis:
            index = list(s_axis).index(s)
        else:
            index = None

        if not op:
            self.close()
        return index

    def find_time(self, dt, tails=False):
        op = self.is_open
        self.open()

        dt_axis = self.dt_axis
        index = None

        if tails and dt < dt_axis[0]:
            index = 0
        elif tails and dt > dt_axis[-1]:
            index = -1
        else:
            for i in range(1, len(dt_axis)):
                if dt_axis[i-1] <= dt <= dt_axis[i]:
                    if dt_axis[i-1] == dt:
                        index = i-1
                    else:
                        index = i
                    break

        if not op:
            self.close()
        return index

    def find_time_range(self, s=None, e=None, tails=False):
        op = self.is_open
        self.open()

        dt_axis = self.dt_axis
        start = None
        end = None

        if s is None or (tails and s < dt_axis[0]):
            start = 0
        if e is None or (tails and e > dt_axis[-1]):
            end = -1

        for i in range(1, len(dt_axis)):
            if dt_axis[i - 1] <= s <= dt_axis[i]:
                if dt_axis[i - 1] == s:
                    start = i - 1
                else:
                    start = i
                if end == -1:
                    break
            if dt_axis[i - 1] <= e <= dt_axis[i]:
                if dt_axis[i - 1] == s:
                    end = i - 1
                else:
                    end = i
                break

        if not op:
            self.close()
        return dt_axis[start:end]

    def data_range(self):
        pass

    def data_range_sample(self, s=None, e=None, tails=False):
        op = self.is_open
        self.open()
        s_axis = list(self.s_axis)
        start = None
        end = None
        d_range = None

        if s is None or (tails and s < s_axis[0]):
            start = 0
        elif s in s_axis:
            start = s_axis.index(s)

        if e is None or (tails and e > s_axis[-1]):
            end = 0
        elif e in s_axis:
            end = s_axis.index(e)

        if start is not None and end is not None:
            d_range = self.data[start:end]

        if not op:
            self.close()
        return d_range, start, end

    def data_range_time(self, s=None, e=None, tails=False):
        op = self.is_open
        self.open()
        dt_axis = self.dt_axis
        start = None
        end = None
        d_range = None

        if s is None or (tails and s < dt_axis[0]):
            start = 0

        if e is None or (tails and e > dt_axis[-1]):
            end = -1

        if start is None or end is None:
            for i in range(1, len(dt_axis)):
                if dt_axis[i-1] <= s <= dt_axis[i]:
                    if dt_axis[i-1] == s:
                        start = i-1
                    else:
                        start = i
                if dt_axis[i-1] <= e <= dt_axis[i]:
                    if dt_axis[i-1] == e:
                        end = i-1
                    else:
                        end = i
                if start is not None and end is not None:
                    break

        if start is not None and end is not None:
            d_range = self.data[start:end]

        if not op:
            self.close()
        return d_range, start, end

