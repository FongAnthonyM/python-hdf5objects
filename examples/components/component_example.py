#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" first_example.py
A basic example, which introduces Maps and file creation/reading.
"""

# Imports #
# Standard Libraries #
import pathlib

# Third-Party Packages #
from hdf5objects import FileMap, DatasetMap
from hdf5objects import HDF5File
from hdf5objects.dataset import BaseDatasetComponent

import numpy as np


# Definitions #
# Classes #
# Components
class ShapesComponent(BaseDatasetComponent):
    def get_min_shape(self, ignore_zeros: bool = False) -> tuple[int, ...]:
        if self.composite.size != 0:
            shapes = self.composite[~np.all(self.composite[...] == 0, axis=1)] if ignore_zeros else self.composite
            return tuple(np.amin(shapes, 0))
        else:
            return (0,)

    def get_max_shape(self) -> tuple[int, ...]:
        return tuple(np.amax(self.composite, 0)) if self.composite.size != 0 else (0,)


# Maps
class ShapesDatasetMap(DatasetMap):
    """An outline which contains shapes and its methods."""

    default_kwargs = {"shape": (0, 0), "maxshape": (None, None), "dtype": "u8"}
    default_component_types = {"shapes": (ShapesComponent, {})}


class ExampleFileMap(FileMap):
    """A map for an example file."""

    # Define Attributes
    default_attribute_names = {"python_name": "File Name"}
    default_attributes = {"python_name": "Timmy"}
    default_map_names = {"data": "Main Array"}
    default_maps = {"data": ShapesDatasetMap()}


class ExampleFile(HDF5File):
    """An example file."""

    # Set the default map of this object
    default_map = ExampleFileMap()


# Main #
if __name__ == "__main__":
    # Parameters #
    file_name = "component_example.h5"
    out_path = pathlib.Path.cwd() / file_name  # The file path as a pathlib Path

    raw_data = np.random.randint(0, 100, size=(10, 10))

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


