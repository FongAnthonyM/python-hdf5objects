#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_hdf5objects.py
Description:
"""
# Package Header #
from src.hdf5objects.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from collections.abc import Mapping
import datetime
import pathlib
import timeit
import uuid
from typing import Any

# Third-Party Packages #
from classversioning import Version, TriNumberVersion
from hdf5objects import HDF5Map
import pytest
import numpy as np

# Local Packages #
from src.hdf5objects import BaseHDF5, BaseHDF5Map, HDF5Dataset, DatasetMap
from src.hdf5objects.dataset import TimeAxisMap
from src.hdf5objects.dataset.components import TimeSeriesComponent


# Definitions #
# Classes #
# Module Implementation
class TimeSeriesTestMap(DatasetMap):
    """Implementation of Dataset."""

    default_attribute_names: Mapping[str, str] = {"t_axis": "t_axis"}
    default_attributes: Mapping[str, Any] = {"t_axis": 0}
    default_component_types = {
        "timeseries": (TimeSeriesComponent, {"scale_name": "time_axis"})
    }
    default_axis_maps = [{"time_axis": TimeAxisMap()}]


class TimeSeriesTestFileMap(BaseHDF5Map):
    """The map for the file which implements a Dataset with a TimeSeriesComponent."""

    default_map_names: Mapping[str, str] = {"test_timeseries": "test_timeseries"}
    default_maps: Mapping[str, HDF5Map] = {
        "test_timeseries": TimeSeriesTestMap(maxshape=(None, None))
    }


class TimeSeriesTestHDF5(BaseHDF5):
    """The file object that implements the Dataset."""

    _registration: bool = True
    FILE_TYPE: str = "TensorModels"
    VERSION: Version = TriNumberVersion(0, 0, 0)
    default_map: HDF5Map = TimeSeriesTestFileMap()


# Module Test
class ClassTest:
    """Default class tests that all classes should pass."""

    class_ = None
    timeit_runs = 2
    speed_tolerance = 200

    def get_log_lines(self, tmp_dir, logger_name):
        path = tmp_dir.joinpath(f"{logger_name}.log")
        with path.open() as f_object:
            lines = f_object.readlines()
        return lines


class TestDataset(ClassTest):
    @pytest.fixture
    def load_file(self, tmp_path):
        return DatasetMap(file=tmp_path)

    @pytest.fixture
    def generate_data(self):
        n_samples = 30
        n_channels = 256
        sample_rate = 1024.0
        start = datetime.datetime.now()
        stop = start.timestamp() + sample_rate * n_samples

        timeseries = np.random.rand(n_samples, n_channels)
        timestamps = np.linspace(start.timestamp(), stop, n_samples)

        return timeseries, timestamps, n_samples, n_channels, sample_rate, start, stop

    def test_require(self, tmp_path, generate_data):
        (
            timeseries_data,
            timestamps,
            n_samples,
            n_channels,
            sample_rate,
            start,
            stop,
        ) = generate_data
        with TimeSeriesTestHDF5(
            file=tmp_path / "test.h5", mode="a", create=True
        ) as test_file:
            test_file.construct_members()
            timeseries = test_file["test_timeseries"]

            time_axis_kwargs = {"data": timestamps, "rate": sample_rate}
            timeseries.require(
                data=timeseries_data, axes_kwargs=[{"time_axis": time_axis_kwargs}]
            )
            existed = timeseries.exists

        assert existed

    # def test_set_item_dict(self, tmp_path):
    #     with DatasetTestHDF5(file=tmp_path / "test.h5", mode='a', create=True, require=True) as test_file:
    #         multi_dataset = test_file["test_dataset"]
    #         multi_dataset.set_item_dict(0, {"First": 2.0, "Second": 3, "Third": "Random"})
    #         assert tuple(multi_dataset[0]) == (2.0, 3, "Random")
    #
    # def test_append_item_dict(self, tmp_path):
    #
    #     test_data = {"First": 2.0, "Second": 3, "Third": "Random", "Fourth": uuid.uuid4()}
    #     with DatasetTestHDF5(file=tmp_path / "test.h5", mode='a', create=True, require=True) as test_file:
    #         multi_dataset = test_file["test_dataset"]
    #         multi_dataset.append_item_dict(test_data)
    #         new_dict = multi_dataset.get_item_dict(-1)
    #         n_items = multi_dataset.shape[0]
    #
    #     assert n_items == 2
    #     assert tuple(test_data) == tuple(new_dict)


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
