#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" first_example.py
A basic example, which introduces Maps and file creation/reading.
"""

# Imports #
# Standard Libraries #
import pathlib

# Third-Party Packages #
from hdf5objects import FileMap
from hdf5objects.dataset import ElectricalSeriesMap
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
    default_maps = {"data": ElectricalSeriesMap(shape=(0, 0), maxshape=(None, None))}


class ExampleFile(HDF5File):
    """An example file."""

    # Set the default map of this object
    default_map = ExampleFileMap()


# Main #
if __name__ == "__main__":
    # Parameters #
    file_name = "electrical_example_file.h5"
    out_path = pathlib.Path.cwd() / file_name  # The file path as a pathlib Path

    n_s = int(2000*300)
    raw_data = np.random.rand(n_s, 7)
    raw_time = ((np.arange(n_s)/2000)*1e9).astype(int)
    raw_lbl = np.array([('AA', 1), ('AA', 2), ('AA', 3), ('BB', 1), ('CC', 1), ('CC', 1), ('CC', 1)])
    print(raw_lbl.shape)
    raw_crd = np.random.randn(7, 3)
	
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
    file = ExampleFile(file=out_path, mode="a", create=True, construct=True)

    # Validate specifications were created
    print(f"Attribute: {file.attributes['python_name'] == 'Timmy'}")
    print(f"Data Shape: {file['data'].shape == (0, 0)}")
    print("")

    # Manipulate Data
    print("Data Manipulation:")

    file_data = file["data"]
    print(f"Original Shape: {file_data.shape}")

    # Set Data
    #file_data.resize(raw_data.shape)
    #file_data[:, :] = raw_data
    #print(f"Shape After Resize: {file_data.shape}")

    # Append
    file_data.append(raw_data, component_kwargs={"timeseries": {"data": raw_time}})
    print(f"Shape After Append: {file_data.shape}")
    print([*file.keys()])

    # View Components and Axes
    print("\nAxes before appending")
    print("file['data'].components: ", file["data"].components)
    print("file['data'].axes: ", file["data"].axes)
    print("file['data'].axes[0]['time_axis']: ", file["data"].axes[0]['time_axis'][...])
    print("file['data'].axes[1]['channellabel_axis']: ", file["data"].axes[1]['channellabel_axis'][...])
    print("file['data'].axes[1]['channelcoord_axis']: ", file["data"].axes[1]['channelcoord_axis'][...])

    print("\nAxes after appending")
    file_data.axes[1]['channellabel_axis'].append(raw_lbl)
    file_data.axes[1]['channelcoord_axis'].append(raw_crd)
    print("channellabel_axis component methods: ", file["data"].axes[1]['channellabel_axis'].components["axis"].channels)
    print("channellabel_axis component methods: ", file["data"].axes[1]['channellabel_axis'].components["axis"].complete_labels)
    print("channellabel_axis component methods: ", file["data"].axes[1]['channellabel_axis'].components["axis"].group_channels_by_sensor())
    print("file['data'].axes[1]['channelcoord_axis']: ", file["data"].axes[1]['channelcoord_axis'][...])
    print("channelcoord_axis component methods: ", file["data"].axes[1]['channelcoord_axis'].components["axis"].calculate_channel_distance())

    print("")

    # After Closing Check if the File Exists
    print(f"File Exists: {out_path.is_file()}")
    print(f"File is Openable: {ExampleFile.is_openable(out_path)}")
    print("")

    file.close()

    # Open File
    # read mode is the default mode.
    # The load kwarg determines if the whole file structure will be loaded in.
    with ExampleFile(file=out_path, load=True, swmr=True) as file:
        # Check Data
        print(f"Attribute: {file.attributes['python_name'] == 'Timmy'}")
        file_data = file["data"]
        print(f"Shape: {file_data.shape}")
        print("file['data'].axes[0]['time_axis']: ", file["data"].axes[0]['time_axis'][...])
        print("file['data'].axes[1]['channellabel_axis']: ", file["data"].axes[1]['channellabel_axis'][...])
        print("file['data'].axes[1]['channelcoord_axis']: ", file["data"].axes[1]['channelcoord_axis'][...])
        print("channelcoord_axis component methods: ", file["data"].axes[1]['channelcoord_axis'].components["axis"].calculate_channel_distance())

