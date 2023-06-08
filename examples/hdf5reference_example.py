#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5reference_example.py
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
import datetime
import pathlib
import uuid
from collections.abc import Mapping
from typing import Any

import h5py
import numpy as np
from classversioning import TriNumberVersion
from classversioning import Version

from hdf5objects import HDF5Map
from hdf5objects.datasets import RegionReferenceDataset
from hdf5objects.datasets import RegionReferenceMap
from hdf5objects.datasets import TimeReferenceMap
from hdf5objects.fileobjects import BaseHDF5
from hdf5objects.fileobjects import BaseHDF5Map

# Third-Party Packages #

# Local Packages #


# Definitions #
# There should be one definition for each type across all scripts and live in a single static location.
# Any changes should be done in a subclass or on the instance level.
# Classes #
# Define Model Dataset
class TimeReferenceMap(TimeReferenceMap):
    """Implementation of TimeReferenceMap."""

    default_attribute_names = RegionReferenceMap.default_attribute_names | {
        "test_attribute": "TestAttribute"
    }
    default_dtype = (
        ("ID", uuid.UUID),
        ("Text", str),
        ("multiple_object", h5py.ref_dtype),
        ("multiple_region", h5py.regionref_dtype),
        ("single_region", h5py.regionref_dtype),
    )
    default_single_reference_fields = {
        "test_single": ("test_attribute", "single_region")
    }
    default_multiple_reference_fields = {
        "test_multiple": ("multiple_object", "multiple_region")
    }
    default_primary_reference_field = "test_single"


class RegionReferenceDatasetTestFileMap(BaseHDF5Map):
    """The map for the file which implements RegionReferenceDataset."""

    # Create names for the contained maps and future objects
    default_map_names: Mapping[str, str] = {"main_dataset": "main_dataset"}
    # Create the contained maps.
    # Note: For dataset the shape, maxshape, and dtype must be initialized to build the dataset on file creation.
    # If you want set the maxshape, do it at the instance level, do not redefine this map just to set it.
    default_maps: Mapping[str, HDF5Map] = {
        "main_dataset": TimeReferenceMap(shape=(0,), maxshape=(None,)),
    }


# Define File Type
class RegionReferenceDatasetTestHDF5(BaseHDF5):
    """The file object that implements the RegionReferenceDataset."""

    _registration: bool = (
        True  # Version registration, to learn more about versioning ask Anthony.
    )
    FILE_TYPE: str = "TestRegionReference"
    VERSION: Version = TriNumberVersion(
        0, 0, 0
    )  # To learn more about versioning ask Anthony.
    default_map: HDF5Map = RegionReferenceDatasetTestFileMap()


# Main #
if __name__ == "__main__":
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
    new_map.maps["model_1"].kwargs.update(
        maxshape=(None, n_channels, n_rank, n_windows)
    )
    new_map.maps["model_2"].kwargs.update(
        maxshape=(None, n_channels, n_rank, n_windows)
    )
    new_map.maps["model_3"].kwargs.update(
        maxshape=(None, n_channels, n_rank, n_windows)
    )

    # Print Map
    print("This is the map of the file:")
    new_map.print_tree()
    # TensorModelsHDF5.default_map.print_tree()  # Prints the default map, but unnecessary since they are the same.
    print("")

    # Check if the File Exists
    # These should print false because the file has not been made.
    print(
        f"File Exists: {out_path.is_file()}"
    )  # [.is_file()] is a useful pathlib Path method
    print(f"File is Openable: {TensorModelsHDF5.is_openable(out_path)}")

    # Create the file
    # The map_ kwarg overrides the default map, in this case the same map with the maxsahpe changed.
    # The create kwarg determines if the file will be created.
    # The require kwarg determines if the file's structure will be built, which is highly suggested for SWMR.
    with TensorModelsHDF5(
        file=out_path, mode="a", map_=new_map, create=True, require=True
    ) as model_file:
        # Note: Caching is off while in write mode. Caching can be turned on using methods covered in the load/read file
        # section. Caching is particularly useful when writing to a file.

        # Assign a File Attribute
        # These attributes were not defined as a property in the TensorModelHDF5 class, so they have to be set and get
        # directly like a normal h5py attribute. Ask Anthony how to set these up as properties.
        model_file.attributes[
            "subject_id"
        ] = "ECxx"  # If this was setup as a property: model_file.subject_id = "ECxx"
        model_file.attributes["start"] = start.timestamp()

        # Create Dataset and Directly Add Some Data (Full Data and Time)
        # The method bellow will work on both a built file and an un-built file.
        model_1_dataset = model_file["model_1"]
        model_1_dataset.require(
            data=tensor_series_1,
            sample_rate=sample_rate,
            timestamps=timestamps_1,
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
        ts_array = np.array(
            [start.timestamp()]
        )  # Create a singe point in time to append.
        t_axis = (
            model_2_dataset.t_axis
        )  # Get the axis to append along, default is the time axis.
        model_2_dataset.append(data=single_tensor_1, axis=t_axis, time_axis=ts_array)

        print(f"The shape of model 2 after first append: {model_2_dataset.shape}")

        # Append Another Point
        ts_array = np.array(
            [datetime.datetime.now().timestamp()]
        )  # Create a singe point in time to append.
        model_2_dataset.append(data=single_tensor_2, time_axis=ts_array)

        print(f"The shape of model 2 after second append: {model_2_dataset.shape}")

        # Append Array with multiple time points
        now = datetime.datetime.now().timestamp()
        stop = now + sample_rate * n_samples_1
        model_2_dataset.append(
            data=tensor_series_2, time_axis=np.linspace(now, stop, n_samples_1)
        )

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
        model_file.set_all_lifetimes(
            2.0
        )  # The sets lifetime of the cache before it will clear in seconds.
        print("")

        # Check Data
        print(f"File Type: {model_file.file_type}")
        print(
            f"File Version: {model_file.file_version}"
        )  # An example of an attribute with a property defined.
        print(
            f"File Subject ID: {model_file.attributes['subject_id']}"
        )  # An attribute without a property defined.
        print(
            f"Start: {datetime.datetime.fromtimestamp(model_file.attributes['start'])}"
        )

        model_1_dataset = model_file["model_1"]
        print(f"The shape of model 1: {model_1_dataset.shape}")

        model_2_dataset = model_file["model_2"]
        print(f"The shape of model 2: {model_2_dataset.shape}")
        print("")

        a = model_1_dataset.ref
        thing = model_file[a]

    # Alternate File instantiation methods
    # The open_ kwarg determines if the file will remain open after instantiation, True by default.
    model_file = TensorModelsHDF5(file=out_path)
    print(f"The file is open: {model_file.is_open}")
    model_file.close()  # Remember to close the file when finished.

    model_file = TensorModelsHDF5(
        file=out_path, open_=False, load=True
    )  # This will close after instantiation
    print(f"The file is open: {model_file.is_open}")

    # Through some wrapping magic you can still access the file even when it is closed.
    # Any time you would access data from the file it will automatically open and close the file.
    # Also, if the values are cached and still within their lifetime those are returned without opening the file.
    model_1_dataset = model_file["model_1"]
    print(f"A data point from model 1: {model_1_dataset[3,3,3,3]}")
    print(f"The file is open: {model_file.is_open}")
    # Be warned, while a cool feature, unfortunately closing the file has a large time overhead. Therefore, if file data
    # is going to be accessed multiple times, it is best to leave the open rather than closing it.
