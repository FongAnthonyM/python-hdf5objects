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
import datetime

# Downloaded Libraries #
import numpy as np

# Local Libraries #
from .hdf5eeg import HDF5EEG


class HDF5XLTEK(HDF5EEG):
    structure = {'type': 'type',
                 'name': 'name',
                 'ID': 'ID',
                 'start': 'start time',
                 'end': 'end time',
                 'data': 'ECoG Array',
                 'samplerate': 'Sampling Rate',
                 'nsamples': 'total samples',
                 'saxis': 'samplestamp vector',
                 'taxis': 'timestamp vector'}

    def __init__(self, name, ID="", path=None, init=False, entry=None):
        HDF5EEG.__init__(self, name, ID, path)
        self._type = 'XLTEK_EEG'
        self._start_entry = None
        self._end_entry = None

        if self.path.is_file():
            self.open()
            if not init:
                self.close()
        elif init:
            self.make_file(entry)

    @property
    def start_entry(self):
        op = self.is_open
        self.open()

        if 'start entry' in self.h5_fobj.attrs and (self._start_entry is None or self.is_updating):
            self._start_entry = self.h5_fobj.attrs['start entry']

        if not op:
            self.close()

        return self._start_entry

    @property
    def end_entry(self):
        op = self.is_open
        self.open()

        if 'end entry' in self.h5_fobj.attrs and (self._end_entry is None or self.is_updating):
            self._end_entry = self.h5_fobj.attrs['end entry']

        if not op:
            self.close()

        return self._end_entry

    @property
    def n_samples(self):
        op = self.is_open
        self.open()

        if self.structure['nsamples'] in self.h5_fobj.attrs and (self._n_samples is None or self.is_updating):
            self._n_samples = int(self.h5_fobj.attrs[self.structure['nsamples']])

        if not op:
            self.close()

        return self._n_samples

    def format_entry(self, entry):
        data = entry['data']
        dshape = data.shape
        channels = np.arange(0, dshape[1])
        samples = np.array(range(entry['start_sample'], entry['end_sample']))

        locs = np.zeros((dshape[0], 4), dtype=np.int32)
        locs[:, :] = entry['entry_info']

        times = np.zeros(dshape[0], dtype=np.float64)
        for i in range(0, len(samples)):
            delta_t = datetime.timedelta(seconds=((samples[i] - entry['snc_sample']) * 1.0 / entry['sample_rate']))
            time = entry['snc_time'] + delta_t
            times[i] = time.timestamp()

        return data, dshape, samples, times, locs, channels, entry['sample_rate']

    def make_file(self, entry):
        start = entry['snc_start'].replace(tzinfo=None)
        data, dshape, samples, times, locs, channels, sample_rate = self.format_entry(entry)

        self.build(start, data, samples, times)

        self.h5_fobj.attrs['start time'] = times[0]
        self.h5_fobj.attrs['end time'] = times[-1]
        self.h5_fobj.attrs['total samples'] = len(samples)
        self.h5_fobj.attrs['start entry'] = locs[0]
        self.h5_fobj.attrs['end entry'] = locs[0]

        ecog = self.h5_fobj['ECoG Array']
        estamps = self.h5_fobj.create_dataset('entry vector', dtype='i', data=locs, maxshape=(None,4), **self.cargs)
        cstamps = self.h5_fobj.create_dataset('channel indices', dtype='i', data=channels, maxshape=(None,), **self.cargs)

        ecog.dims.create_scale(estamps, 'entry axis')
        ecog.dims.create_scale(cstamps, 'channel axis')

        ecog.dims[0].attach_scale(estamps)
        ecog.dims[1].attach_scale(cstamps)

        self.h5_fobj.flush()

        return self.h5_fobj

    def add_data(self, entry, h5_fobj=None):
        if h5_fobj is None:
            h5_fobj = self.h5_fobj

        data, dshape, samples, times, locs, channels, sample_rate = self.format_entry(entry)

        h5_fobj.attrs['end entry'] = locs[0]
        h5_fobj.attrs['end time'] = times[-1]

        ecog = h5_fobj['ECoG Array']
        last_total = ecog.shape[0]

        samplestamps = h5_fobj['samplestamp vector']
        timestamps = h5_fobj['timestamp vector']
        entrystamps = h5_fobj['entry vector']

        ecog.resize(last_total+dshape[0], 0)
        samplestamps.resize(last_total+dshape[0], 0)
        timestamps.resize(last_total+dshape[0], 0)
        entrystamps.resize(last_total+dshape[0], 0)

        ecog[-dshape[0]:, :] = data
        samplestamps[-dshape[0]:] = samples
        timestamps[-dshape[0]:] = times
        entrystamps[-dshape[0]:, :] = locs

        h5_fobj.attrs['total samples'] = len(samplestamps)


