#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" first_example.py
Description: 
"""

# Imports #
# Standard Libraries #
import pathlib

# Third-Party Packages #
from hdf5objects import FileMap, DatasetMap
from hdf5objects import HDF5File

import numpy as np


# Definitions #
# Classes #
class ExampleFileMap(FileMap):
    """A map for an example file."""

    # Define Attributes
    default_attribute_names = {"python_name": "File Name"}
    default_attributes = {"python_name": "Timmy"}

    # Define Child Maps
    default_map_names = {"data": "Main Array"}
    default_maps = {"data": DatasetMap(shape=(0, 0), maxshape=(None, None))}


class ExampleFile(HDF5File):
    """An example file."""

    # Set the default map of this object
    default_map = ExampleFileMap()


# Main #
if __name__ == "__main__":
    # Parameters #
    file_name = "first_example_file.h5"
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
    # The require kwarg determines if the file's structure will be built, which is highly suggested for SWMR.
    with ExampleFile(file=out_path, mode="a", create=True, require=True) as file:
        # Validate specifications were created
        print(f"Attribute: {file.attributes['python_name'] == 'Timmy'}")
        print(f"Data Shape: {file['data'].shape == (0, 0)}")
        print("")

        # Manipulate Data
        print("Data Manipulation:")

        file_data = file["data"]
        print(f"Original Shape: {file_data.shape}")

        # Set Data
        file_data.resize((10, 10))
        file_data[:, :] = raw_data
        print(f"Shape After Resize: {file_data.shape}")

        # Append
        file_data.append(raw_data)
        print(f"Shape After Append: {file_data.shape}")

