#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5model_example.py

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
from typing import Any

# Third-Party Packages #
from src.hdf5objects import HDF5Map, HDF5BaseObject
from src.hdf5objects.datasets import TimeSeriesMap, TimeSeriesDataset  # Remove src when this package is pip installed
from src.hdf5objects.fileobjects import BaseHDF5, BaseHDF5Map  # Remove src when this package is pip installed
import numpy as np

# Local Packages #


# Definitions #
# Classes #
# Define Model Dataset
class TimeTensorMap(TimeSeriesMap):
    """A map for a time Tensor dataset."""
    # Create names for the attributes
    default_attribute_names: Mapping[str, str] = {"sample_rate": "samplerate",
                                                  "n_samples": "n_samples",
                                                  "c_axis": "c_axis",
                                                  "t_axis": "t_axis",
                                                  "r_axis": "r_axis",
                                                  "w_axis": "w_axis"}
    # Create set the initial values for the attributes
    default_attributes: Mapping[str, Any] = {"n_samples": 0,
                                             "c_axis": 1,
                                             "t_axis": 0,
                                             "r_axis": 2,
                                             "w_axis": 3}
    # For this example we do not need to override this attributes because we are not changing them.
    # default_map_names = {"channel_axis": "channel_axis",
    #                      "sample_axis": "sample_axis",
    #                      "time_axis": "time_axis"}
    # default_maps = {"channel_axis": ChannelAxisMap(),
    #                 "sample_axis": SampleAxisMap(),
    #                 "time_axis": TimeAxisMap()}


class TimeTensorDataset(TimeSeriesDataset):
    """A dataset that holds a series of Tensors that are related in time."""
    default_map: HDF5Map = TimeTensorMap()


# Assign Cyclic Definitions
TimeTensorMap.default_type = TimeTensorDataset  # This allows maps to construct the dataset directly


# Define File Type
class TensorModelsMap(BaseHDF5Map):
    """A map for the Tensor Models HDF5 file."""
    # Create names for the attributes
    default_attribute_names: Mapping[str, str] = {"file_type": "FileType",
                                                  "file_version": "FileVersion",
                                                  "subject_id": "subject_id",
                                                  "start": "start",
                                                  "end": "end"}
    # Create names for the contained maps and future objects
    default_map_names: Mapping[str, str] = {"model_1": "model_1", "model_2": "model_2", "model_3": "model_3"}
    # Create the contained maps.
    # Note: For datasets the shape, maxshape, and dtype must be initialized to build the datasets on file creation.
    default_maps: Mapping[str, HDF5Map] = {
        "model_1": TimeTensorMap(shape=(0, 0, 0, 0), maxshape=(None, None, None, None), dtype="f8"),
        "model_2": TimeTensorMap(shape=(0, 0, 0, 0), maxshape=(None, None, None, None), dtype="f8"),
        "model_3": TimeTensorMap(shape=(0, 0, 0, 0), maxshape=(None, None, None, None), dtype="f8"),
    }


class TensorModelsHDF5(BaseHDF5):
    """The Tensor Models HDF5 file object."""
    FILE_TYPE: str = "TensorModels"
    default_map: HDF5Map = TensorModelsMap()


# Main #
if __name__ == '__main__':
    # Define some parameters
    file_name = "test_model_file.h5"
    out_path = pathlib.Path.cwd().joinpath(file_name)

    n_samples_1 = 30
    n_channels = 256
    n_rank = 50
    n_windows = 50

    start = datetime.datetime.now()
    sample_rate = 1024.0
    stop = start.timestamp() + sample_rate * n_samples_1

    tensor_series_1 = np.random.rand(n_samples_1, n_channels, n_rank, n_windows)
    timestamps_1 = np.linspace(start.timestamp(), stop, n_samples_1)

    single_tensor_1 = np.random.rand(n_channels, n_rank, n_windows)
    single_tensor_2 = np.random.rand(n_channels, n_rank, n_windows)

    # Print Map
    print("This is the map of the file that was created:")
    TensorModelsHDF5.default_map.print_tree()

    # Create the file
    with TensorModelsHDF5(file=out_path, mode='a', create=True, build=True) as model_file:
        # Set Single Write Multiple Reader
        # Not that the swr_mode should only be set to true after all the datasets and attributes have been created.
        # model_file.swmr_mode = True

        # Create Dataset and Directly Add Some Data (Full Data and Time)
        model_1_dataset = model_file["model_1"]
        model_1_dataset.require(
            data=tensor_series_1,
            start=start,
            sample_rate=sample_rate,
            timestamps=timestamps_1,  # Can leave blank and timestamps will be generated from start and sample rate.
        )

        print(f"The shape of model 1: {model_1_dataset.shape}")

        # Create Dataset and Directly Add Some Data (Appending Data)
        model_2_dataset = model_file["model_1"]
        model_2_dataset.require(
            maxshape=(None, n_channels, n_rank, n_windows),
            start=start,
            # sample_rate=sample_rate,  # There does not need to be a sample rate
        )

        ts_array = np.array([datetime.datetime.now().timestamp()])
        model_2_dataset.append(data=single_tensor_1, axis=model_2_dataset.t_axis, time_axis=ts_array)

        print(f"The shape of model 2 after first append: {model_2_dataset.shape}")

        ts_array = np.array([datetime.datetime.now().timestamp()])
        model_2_dataset.append(data=single_tensor_2, axis=model_2_dataset.t_axis, time_axis=ts_array)

        print(f"The shape of model 2 after second append: {model_2_dataset.shape}")


