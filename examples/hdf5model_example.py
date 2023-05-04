#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5model_example.py
A detailed example of how to create a file type and use hdf5objects.
"""
# # Package Header #
# from src.hdf5objects.header import *
#
# # Header #
# __author__ = __author__
# __credits__ = __credits__
# __maintainer__ = __maintainer__
# __email__ = __email__


# Imports #
# Standard Libraries #
from collections.abc import Mapping
import datetime
import pathlib
from typing import Any

# Third-Party Packages #
from classversioning import Version, TriNumberVersion
from hdf5objects import HDF5Map, DatasetMap
from hdf5objects.dataset import ChannelAxisMap, SampleAxisMap, TimeAxisMap
from hdf5objects.dataset.components import TimeSeriesComponent
from hdf5objects.fileobjects import BaseHDF5, BaseHDF5Map
import numpy as np

# Local Packages #


# Definitions #
# There should be one definition for each type across all scripts and live in a single static location.
# Any changes should be done in a subclass or on the instance level.
# Classes #
# Define Model Dataset
class TimeTensorMap(DatasetMap):
    """A map for a time Tensor dataset."""
    # Create names for the attributes
    default_attribute_names: Mapping[str, str] = {"n_samples": "n_samples",
                                                  "c_axis": "c_axis",
                                                  "t_axis": "t_axis",
                                                  "r_axis": "r_axis",
                                                  "w_axis": "w_axis"}
    # Create and set the initial values for the attributes
    default_attributes: Mapping[str, Any] = {"n_samples": 0,
                                             "c_axis": 1,
                                             "t_axis": 0,
                                             "r_axis": 2,
                                             "w_axis": 3}
    # For this example we do not need to override this attributes because we are not changing them.
    # default_map_names = {"channel_axis": "channel_axis",
    #                      "sample_axis": "sample_axis",
    #                      "time_axis": "time_axis"}
    default_component_types = {"timeseries": (TimeSeriesComponent, {"scale_name": "time_axis"})}
    default_axis_maps = [
        {"sample_axis": SampleAxisMap(), "time_axis": TimeAxisMap()},
        {"channel_axis": ChannelAxisMap()},
    ]
    # TimeAxisMap(attributes={"sample_rate": h5py.Empty('f8'), "time_zone": ""})}
    # Can set the default sample rate and time zone by filling in the dict values above.


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
    # Note: For dataset the shape, maxshape, and dtype must be initialized to build the dataset on file creation.
    # If you want set the maxshape, do it at the instance level, do not redefine this map just to set it.
    default_maps: Mapping[str, HDF5Map] = {
        "model_1": TimeTensorMap(dtype='f8', object_kwargs={"shape": (0, 0, 0, 0), "maxshape": (None, None, None, None)}),
        "model_2": TimeTensorMap(dtype='f8', object_kwargs={"shape": (0, 0, 0, 0), "maxshape": (None, None, None, None)}),
        "model_3": TimeTensorMap(dtype='f8', object_kwargs={"shape": (0, 0, 0, 0), "maxshape": (None, None, None, None)}),
    }


class TensorModelsHDF5(BaseHDF5):
    """The Tensor Models HDF5 file object."""
    _registration: bool = True  # Version registration, to learn more about versioning ask Anthony.
    FILE_TYPE: str = "TensorModels"
    VERSION: Version = TriNumberVersion(0, 0, 0)  # To learn more about versioning ask Anthony.
    default_map: HDF5Map = TensorModelsMap()


# Main #
if __name__ == '__main__':
    # Define some parameters
    file_name = "test_model_file.h5"
    out_path = pathlib.Path.cwd().joinpath(file_name)  # The file path as a pathlib Path

    n_samples_1 = 30
    n_channels = 256
    n_rank = 50
    n_windows = 50

    start = datetime.datetime.now()
    sample_rate = 1024.0
    stop = start.timestamp() + sample_rate * n_samples_1

    tensor_series_1 = np.random.rand(n_samples_1, n_channels, n_rank, n_windows)
    timestamps_1 = np.linspace(start.timestamp(), stop, n_samples_1)

    tensor_series_2 = np.random.rand(n_samples_1, n_channels, n_rank, n_windows)

    single_tensor_1 = np.random.rand(n_channels, n_rank, n_windows)
    single_tensor_2 = np.random.rand(n_channels, n_rank, n_windows)

    # Start of the actual code.

    # Create New Map with known maxshape (this is optional, but it can save space)
    new_map = TensorModelsMap()
    new_map.maps["model_1"].kwargs.update(maxshape=(None, n_channels, n_rank, n_windows))
    new_map.maps["model_2"].kwargs.update(maxshape=(None, n_channels, n_rank, n_windows))
    new_map.maps["model_3"].kwargs.update(maxshape=(None, n_channels, n_rank, n_windows))

    # Print Map
    print("This is the map of the file:")
    new_map.print_tree()
    # TensorModelsHDF5.default_map.print_tree()  # Prints the default map, but unnecessary since they are the same.
    print("")

    # Check if the File Exists
    # These should print false because the file has not been made.
    print(f"File Exists: {out_path.is_file()}")  # [.is_file()] is a useful pathlib Path method
    print(f"File is Openable: {TensorModelsHDF5.is_openable(out_path)}")

    # Create the file
    # The map_ kwarg overrides the default map, in this case the same map with the maxsahpe changed.
    # The create kwarg determines if the file will be created.
    # The construct kwarg determines if the file's structure will be built.
    # The require kwarg determines if the file's structure will be filled, which is highly suggested for SWMR.
    with TensorModelsHDF5(file=out_path, mode='a', map_=new_map, create=True, construct=True, require=True) as model_file:
        # Note: Caching is off while in write mode. Caching can be turned on using methods covered in the load/read file
        # section. Caching is particularly useful when reading a file.

        # Assign a File Attribute
        # These attributes were not defined as a property in the TensorModelHDF5 class, so they have to be set and get
        # directly like a normal h5py attribute. Ask Anthony how to set these up as properties.
        model_file.attributes["subject_id"] = "ECxx"  # If this was setup as a property: model_file.subject_id = "ECxx"
        model_file.attributes["start"] = start.timestamp()

        # Create Dataset and Directly Add Some Data (Full Data and Time)
        # The method bellow will work on both a constructed and un-constructed file.
        model_1_dataset = model_file.construct_member("model_1")
        model_1_dataset.require(
            data=tensor_series_1,
            axes_kwargs=[{"time_axis": {"data": timestamps_1, "rate": sample_rate}}]
        )
        # Note: Timestamps do not need to be explicitly given, instead they can be interpolated from the start and fs.

        print(f"The shape of model 1: {model_1_dataset.shape}")

        model_2_dataset = model_file["model_2"]
        # Since require was called in the file creation we do not need to require the dataset.
        # However, if require was not called, then the require method would need to be called.
        # model_2_dataset.require(
        #     start=start,
        #     # sample_rate=sample_rate,  # Sample rate does not need to be set
        # )
        model_2_dataset.sample_rate = sample_rate

        # Set Single Write Multiple Reader
        # Note: that the swr_mode should only be set to true after all the dataset and attributes have been set.
        # Once in SWMR only add data to dataset and do not change attributes.
        model_file.swmr_mode = True

        # Directly Add Some Data (Appending Data)
        print(f"The shape of model 2 at start: {model_2_dataset.shape}")

        # Append One Point
        ts_array = np.array([start.timestamp()])  # Create a singe point in time to append.
        t_axis = model_2_dataset.components["timeseries"].t_axis  # Get the axis to append along, default is the time axis.
        model_2_dataset.append(data=single_tensor_1, axis=t_axis, component_kwargs={"timeseries": {"data": ts_array}})

        print(f"The shape of model 2 after first append: {model_2_dataset.shape}")

        # Append Another Point
        ts_array = np.array([datetime.datetime.now().timestamp()])  # Create a singe point in time to append.
        model_2_dataset.append(data=single_tensor_2, axis=t_axis, component_kwargs={"timeseries": {"data": ts_array}})

        print(f"The shape of model 2 after second append: {model_2_dataset.shape}")

        # Append Array with multiple time points
        now = datetime.datetime.now().timestamp()
        stop = now + sample_rate * n_samples_1
        model_2_dataset.append(data=tensor_series_2, axis=t_axis, component_kwargs={"timeseries": {"data": np.linspace(now, stop, n_samples_1)}})

        print(f"The shape of model 2 after third append: {model_2_dataset.shape}")

    print("")

    # After Closing Check if the File Exists
    print(f"File Exists: {out_path.is_file()}")
    print(f"File is Openable: {TensorModelsHDF5.is_openable(out_path)}")
    print(f"File Type is Valid: {TensorModelsHDF5.validate_file_type(out_path)}")
    # In the class definition a validation method can be defined so that any features can be used in validating.
    # For example, the presence of a particular attribute or its value could be used.
    # print(f"File is Valid: {TensorModelsHDF5.validate_file(out_path)}")
    print("")

    # Open File
    # read mode is the default mode.
    # The load kwarg determines if the whole file structure will be loaded in. This is useful if you plan on looking at
    # everything in the file, but if load is False or not set it will load parts of the structure on demand which is
    # more efficient if you are looking at specific parts and not checking others.
    # [map_=new_map] does not need to be set here because we are loading a file and not requiring it.
    with TensorModelsHDF5(file=out_path, load=True, swmr=True) as model_file:
        # Caching is on when in read mode.
        # In normal read mode, once data is loaded into cache it has to manually be told refresh the cache.
        # In SWMR mode the cache will clear and get the values from the file again at a predefined interval.
        # There are methods which can turn caching on and off and can set the caching interval.

        # Caching
        # There are two versions of the caching methods. There methods that operate on the specific object and ones that
        # change the caching for all objects contained within that object as well. The larger scope methods have _all_
        # in their method name.
        print(f"Caching: {model_file.is_cache}")
        model_file.disable_all_caching()  # Since _all_ is present, this will apply to all contained objects as well.
        print(f"Caching: {model_file.is_cache}")
        model_file.enable_all_caching()
        print(f"Caching: {model_file.is_cache}")
        model_file.timeless_all_caching()  # Caches will not clear on their own.
        model_file.clear_all_caches()  # Clear all the caches.
        model_file.timed_all_caching()  # Caches will clear at regular intervals.
        model_file.set_all_lifetimes(2.0)  # The sets lifetime of the cache before it will clear in seconds.
        print("")

        # Check Data
        print(f"File Type: {model_file.file_type}")
        print(f"File Version: {model_file.file_version}")  # An example of an attribute with a property defined.
        print(f"File Subject ID: {model_file.attributes['subject_id']}")  # An attribute without a property defined.
        print(f"Start: {datetime.datetime.fromtimestamp(model_file.attributes['start'])}")

        model_1_dataset = model_file["model_1"]
        print(f"The shape of model 1: {model_1_dataset.shape}")

        model_2_dataset = model_file["model_2"]
        print(f"The shape of model 2: {model_2_dataset.shape}")
        print("")

    # Alternate File instantiation methods
    # The open_ kwarg determines if the file will remain open after instantiation, True by default.
    model_file = TensorModelsHDF5(file=out_path)
    print(f"The file is open: {model_file.is_open}")
    model_file.close()  # Remember to close the file when finished.

    model_file = TensorModelsHDF5(file=out_path, open_=False, load=True)  # This will close after instantiation
    print(f"The file is open: {model_file.is_open}")

    # Through some wrapping magic you can still access the file even when it is closed.
    # Any time you would access data from the file it will automatically open and close the file.
    # Also, if the values are cached and still within their lifetime those are returned without opening the file.
    model_1_dataset = model_file["model_1"]
    print(f"A data point from model 1: {model_1_dataset[3,3,3,3]}")
    print(f"The file is open: {model_file.is_open}")
    # Be warned, while a cool feature, unfortunately closing the file has a large time overhead. Therefore, if file data
    # is going to be accessed multiple times, it is best to leave the open rather than closing it.
