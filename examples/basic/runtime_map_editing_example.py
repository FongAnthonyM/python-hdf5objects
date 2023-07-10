#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" group_example.py
An example of editing a map before creating a file, at runtime.

Runtime map editing is useful because default parameters can be set without permanently changing the structure.
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
    file_name = "runtime_map_editing_example.h5"
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

    # Edit Map
    # Create New Map with known maxshape (this is optional, but it can save space)
    new_map = ExampleFileMap()
    new_map["group_1"].attributes["sample_rate"] = 512
    new_map["group_1"]["raw_data"].kwargs.update(shape=(10, 10), maxshape=(None, 10))

    new_map["group_2"].attributes["sample_rate"] = 1024
    new_map["group_2"]["raw_data"].kwargs.update(shape=(10, 10), maxshape=(None, 10))

    # Create the file #
    # The map_ kwarg overrides the default map, in this case the same map with the maxshape changed.
    # The create kwarg determines if the file will be created.
    # The require kwarg determines if the file's structure will be built, which is highly suggested for SWMR.
    with ExampleFile(file=out_path, mode="a", map_=new_map, create=True, require=True) as file:
        # Go Through Groups
        for i, (name, value) in enumerate(file.items()):
            print(f"Group Python Name: {name}")
            print(f"Sample Rate: {value.attributes['sample_rate']}")

            # Go through Group contents
            for j, (n, v) in enumerate(value.items()):
                print(f"Group Content {i}: {n}")
                print(f"Shape: {v.shape}")

            print("")

    # After Closing Check if the File Exists
    print(f"File Exists: {out_path.is_file()}")
    print(f"File is Openable: {ExampleFile.is_openable(out_path)}")
    print("")

    # Open File
    # read mode is the default mode.
    # The load kwarg determines if the whole file structure will be loaded in. This is useful if you plan on looking at
    # everything in the file, but if load is False or not set it will load parts of the structure on demand which is
    # more efficient if you are looking at specific parts and not checking others.
    # [map_=new_map] does not need to be set here because we are loading a file and not requiring it.
    with ExampleFile(file=out_path, load=True, swmr=True) as file:
        # Go Through Groups
        for i, (name, value) in enumerate(file.items()):
            print(f"Group Python Name: {name}")
            print(f"Sample Rate: {value.attributes['sample_rate']}")

            # Go through Group contents
            for j, (n, v) in enumerate(value.items()):
                print(f"Group Content {i}: {n}")
                print(f"Shape: {v.shape}")

            print("")
