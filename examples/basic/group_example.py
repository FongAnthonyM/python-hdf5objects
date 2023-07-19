#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" group_example.py
An example of creating a file map with groups.
"""

# Imports #
# Standard Libraries #
import pathlib

# Third-Party Packages #
from hdf5objects import FileMap, GroupMap, DatasetMap
from hdf5objects import HDF5File

import h5py
import numpy as np


# Definitions #
# Classes #
class ExampleGroupMap(GroupMap):
    """A map for an example group."""

    default_attribute_names = {"label": "Label", "sample_rate": "sample_rate"}
    default_attributes = {"label": "", "sample_rate": h5py.Empty("f8")}
    default_map_names = {"raw_data": "raw_data", "processed_data": "processed_data"}
    default_maps = {
        "raw_data": DatasetMap(shape=(0, 0), maxshape=(None, None)),
        "processed_data": DatasetMap(shape=(0, 0, 0), maxshape=(None, None, None)),
    }


class ExampleFileMap(FileMap):
    """A map for an example file."""

    default_attribute_names = {"python_name": "File Name"}
    default_attributes = {"python_name": "Timmy"}
    default_map_names = {"group_1": "Group 1", "group_2": "Group 2"}
    default_maps = {
        "group_1": ExampleGroupMap(attributes={"label": "default group label 1"}),
        "group_2": ExampleGroupMap(attributes={"label": "default group label 2"}),
    }


class ExampleFile(HDF5File):
    """An example file."""

    # Set the default map of this object
    default_map = ExampleFileMap()


# Main #
if __name__ == "__main__":
    # Parameters #
    file_name = "group_example_file.h5"
    out_path = pathlib.Path.cwd() / file_name  # The file path as a pathlib Path

    raw_data = np.random.rand(10, 10)

    # Start of the actual code.

    # Map Information #
    print("Map Information:")
    print("This is the map of the file:")
    ExampleFileMap.print_tree_class()
    # TensorModelsHDF5.default_map.print_tree()  # Prints the default map, but unnecessary since they are the same.
    print("")

    # Check if the File Exists
    # These should print false because the file has not been made.
    print(f"File Exists: {out_path.is_file()}")
    print(f"File is Openable: {ExampleFile.is_openable(out_path)}")
    print("")

    # Create the file #
    # The create kwarg determines if the file will be created.
    # The construct kwarg determines if the file's structure will be built, which is highly suggested for SWMR.
    with ExampleFile(file=out_path, mode="a", create=True, construct=True) as file:
        # Go Through Groups
        for i, (name, value) in enumerate(file.items()):
            print(f"Group Python Name: {name}")
            print(f"Original Label: {value.attributes['label']}")
            value.attributes["label"] = f"New Label {i}"
            print(f"Edited Label: {value.attributes['label']}")

            print(f"Original Sample Rate: {value.attributes['sample_rate']}")
            value.attributes["sample_rate"] = (i + 1) * 512
            print(f"Edited Sample Rate: {value.attributes['sample_rate']}")

            # Go through Group contents
            for j, (n, v) in enumerate(value.items()):
                print(f"Group Content {i}: {n}")
                print(f"Shape: {v.shape}")

            print("")
