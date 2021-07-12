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
import numpy as np

# Local Libraries #
from ..hdf5object import HDF5Dataset


# Definitions #
# Classes #
class TimeSeriesDataset(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    attribute_map = {"sample_rate": "samplerate",
                     "n_samples": "n_samples",
                     "t_axis": "t_axis"}

    # Magic Methods
    # Construction/Destruction
    def __init__(self, data=None, sample_rate=None, t_axis=0, create=True, init=True, **kwargs):
        super().__init__(init=False)
        self._sample_rate = 0
        self._n_samples = 0
        self._t_axis = 0

        self.sample_axis = None
        self.time_axis = None

        if init:
            self.construct(data, sample_rate, t_axis, create, **kwargs)

    @property
    def sample_rate(self):
        try:
            self._sample_rate = self.attributes[self.attribute_map["sample_rate"]]
        finally:
            return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        try:
            self.attributes[self.attribute_map["sample_rate"]] = value
        except:
            pass

    @property
    def n_samples(self):
        try:
            self._n_samples = self.attributes[self.attribute_map["n_samples"]]
        finally:
            return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        try:
            self.attributes[self.attribute_map["n_samples"]] = value
        except:
            pass

    @property
    def t_axis(self):
        try:
            self._t_axis = self.attributes[self.attribute_map["t_axis"]]
        finally:
            return self._t_axis

    @t_axis.setter
    def t_axis(self, value):
        try:
            self.attributes[self.attribute_map["t_axis"]] = value
        except:
            pass

    # Instance Methods
    # Constructors/Destructors
    def construct(self, data=None, sample_rate=None, t_axis=None, create=True, **kwargs):
        super().construct(create=create, data=data, **kwargs)
        if data is not None:
            self.n_samples = data

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if t_axis is not None:
            self.t_axis = t_axis

    # Axes
    def create_sample_axis(self, name, start=None, end=None, dim=None, **kwargs):
        pass


    def attach_sample_axis(self, dataset):
        self.attach_axis(dataset, axis=self.t_axis)
        self.sample_axis = dataset
