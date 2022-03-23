""" hdf5dataset.py
An object that represents an HDF5 Dataset.
"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import pathlib
from typing import Any

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod
from baseobjects.cachingtools import timed_keyless_cache
import h5py
import numpy as np

# Local Packages #
from .hdf5map import HDF5Map
from .hdf5baseobject import HDF5BaseObject
from .hdf5attributes import HDF5Attributes


# Definitions #
# Classes #
class HDF5Dataset(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 dataset and gives more functionality.

    Class Attributes:
        _wrapped_types: A list of either types or objects to set up wrapping for.
        _wrap_attributes: Attribute names that will contain the objects to wrap where the resolution order is descending
            inheritance.

    Attributes:
        _dataset: The HDF5 dataset to wrap.
        _scale_name: The name of this dataset if it is a scale.
        attributes: The attributes of this dataset.

    Args:
        dataset: The HDF5 dataset to build this dataset around.
        name: The HDF5 name of this object.
        map_: The map for this HDF5 object.
        file: The file object that this dataset object originates from.
        load: Determines if this object will load the dataset from the file on construction.
        build: Determines if this object will create and fill the dataset in the file on construction.
        parent: The HDF5 name of the parent of this HDF5 object.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments to construct the base HDF5 dataset.
    """
    _wrapped_types: list[type | object] = [h5py.Dataset]
    _wrap_attributes: list[str] = ["dataset"]

    # Magic Methods
    # Constructors/Destructors
    def __init__(
        self,
        dataset: h5py.Dataset | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        build: bool = False,
        parent: str | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(file=file, init=False)

        # New Attributes #
        self._dataset: h5py.Dataset | None = None
        self._scale_name: str | None = None
        self.attributes: HDF5Attributes | None = None

        # Object Construction #
        if init:
            self.construct(
                dataset=dataset,
                name=name,
                map_=map_,
                file=file,
                load=load,
                build=build,
                parent=parent,
                **kwargs,
            )

    @property
    def scale_name(self) -> str:
        """The name of this dataset if it is a scale. The setter applies the scale name in the HDF5 file."""
        return self._scale_name

    @scale_name.setter
    def scale_name(self, value: str) -> None:
        self.make_scale(value)

    @property
    def shape(self) -> tuple[int]:
        """Get the shape of the data in this dataset."""
        try:
            return self.get_shape.caching_call()
        except AttributeError:
            return self.get_shape()

    @property
    def all_data(self) -> np.ndarray:
        """Get all the data in this dataset as a numpy array, caching the output."""
        try:
            return self.get_all_data.caching_call()
        except AttributeError:
            return self.get_all_data()

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Return this dataset as a numpy array."""
        with self:
            return self._dataset.__array__(dtype=dtype)

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        dataset: h5py.Dataset | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        build: bool = False,
        parent: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            dataset: The HDF5 dataset to build this dataset around.
            name: The HDF5 name of this object.
            map_: The map for this HDF5 object.
            file: The file object that this dataset object originates from.
            load: Determines if this object will load the dataset from the file on construction.
            build: Determines if this object will create and fill the dataset in the file on construction.
            parent: The HDF5 name of the parent of this HDF5 object.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.
        """
        if file is None and isinstance(dataset, str):
            raise ValueError("A file must be given if giving dataset name")

        super().construct(name=name, map_=map_, file=file, parent=parent)

        if self.map.type is None:
            self.map.type = type(self)

        if dataset is not None:
            self.set_dataset(dataset)

        self.construct_attributes(build=build)

        if load and self.exists:
            self.load()

        if build:
            self.construct_dataset(**kwargs)

    def construct_attributes(self, map_: HDF5Map = None, load: bool = False, build: bool = False) -> None:
        """Creates the attributes for this dataset.

        Args:
            map_: The map to use to create the attributes.
            load: Determines if this object will load the attribute values from the file on construction.
            build: Determines if this object will create and fill the attributes in the file on construction.
        """
        if map_ is None:
            map_ = self.map
        self.attributes = map_.attributes_type(name=self._full_name, map_=map_, file=self._file, load=load, build=build)

    def construct_dataset(self, **kwargs: Any) -> None:
        """Creates and fills the data if it does not exist.

        Args:
            **kwargs: The keyword arguments for constructing a HDF5 Dataset.
        """
        self.require(name=self._full_name, **kwargs)

    # File
    def load(self) -> None:
        """Loads this dataset which is just loading the attributes."""
        self.attributes.load()

    def refresh(self) -> None:
        """Reloads the dataset and attributes."""
        with self:
            self._dataset.refresh()
        self.attributes.refresh()
        self.get_shape.clear_cahce()
        self.get_all_data.clear_cache()

    # Getters/Setters
    @singlekwargdispatchmethod("dataset")
    def set_dataset(self, dataset: h5py.Dataset) -> None:
        """Sets the wrapped dataset.

        Args:
            dataset: The dataset this object will wrap.
        """
        if isinstance(dataset, HDF5Dataset):
            if self._file is None:
                self._file = dataset._file
            self.set_name(dataset._name)
            self._dataset = dataset._dataset
        else:
            raise TypeError(f"{type(dataset)} is not a valid type for set_dataset.")

    @set_dataset.register
    def _(self, dataset: h5py.Dataset) -> None:
        """Sets the wrapped dataset.

        Args:
            dataset: The dataset this object will wrap.
        """
        if not dataset:
            raise ValueError("Dataset needs to be open")
        if self._file is None:
            self._file = self.file_type(dataset.file)
        self.set_name(dataset.name)
        self._dataset = dataset

    @timed_keyless_cache(call_method="clearing_call", collective=False)
    def get_shape(self) -> tuple[int]:
        """Gets the shape of the dataset.

        Returns:
            The shape of the dataset.
        """
        with self:
            return self._dataset.shape

    @timed_keyless_cache(call_method="clearing_call", collective=False)
    def get_all_data(self) -> np.ndarray:
        """Gets all the data in the dataset.

        Returns:
            All the data in the dataset.
        """
        with self:
            return self._dataset[...]

    # Data Modification
    def create(self, name: str = None, **kwargs: Any) -> "HDF5Dataset":
        """Creates and fills the data, gives an error if it already exists.

        Args:
            name: The name of the dataset.
            **kwargs: The keyword arguments for constructing a HDF5 Dataset.

        Returns:
            This object.
        """
        if name is not None:
            self._name = name

        if "data" in kwargs:
            if "shape" not in kwargs:
                kwargs["shape"] = kwargs["data"].shape
            if "maxshape" not in kwargs:
                kwargs["maxshape"] = kwargs["data"].shape

        with self._file.temp_open():
            self._dataset = self._file._file.create_dataset(name=self._full_name, **kwargs)
            self.attributes.construct_attributes()
            self.make_scale()

        return self

    def require(self, name: str = None, **kwargs: Any) -> "HDF5Dataset":
        """Creates and fills the data if it does not exist.

        Args:
            name: The name of the dataset.
            **kwargs: The keyword arguments for constructing a HDF5 Dataset.

        Returns:
            This object.
        """
        if name is not None:
            self._name = name

        if "data" in kwargs:
            if "shape" not in kwargs:
                kwargs["shape"] = kwargs["data"].shape
            if "dtype" not in kwargs:
                kwargs["dtype"] = kwargs["data"].dtype

        with self._file.temp_open():
            existed = self.exists
            self._dataset = self._file._file.require_dataset(name=self._full_name, **kwargs)
            if not existed:
                self.attributes.construct_attributes()
                self.make_scale()

        return self

    def replace_data(self, data: np.ndarray) -> None:
        """Replaces the data in the dataset with new data.

        Args:
            data: A numpy array like object that can be used to replace the data.
        """
        with self:
            # Assign Data
            self._dataset.resize(data.shape)  # resize for new data
            self._dataset[...] = data

    def set_data(self, data: np.ndarray, **kwargs: Any) -> None:
        """Sets the data by either creating it or replacing it.

        Args:
            data: The data to fill the dataset with.
            **kwargs: The keyword arguments for creating the dataset.
        """
        if self.exists:
            self.replace_data(data=data)
        else:
            self.require(data=data, **kwargs)

    def append(self, data: np.ndarray, axis: int = 0) -> None:
        """Append data to the dataset along a specified axis.

        Args:
            data: The data to append.
            axis: The axis to append the data along.
        """
        with self:
            # Get the shapes of the dataset and the new data to be added
            s_shape = self._dataset.shape
            d_shape = data.shape
            # Determine the new shape of the dataset
            new_shape = list(s_shape)
            new_shape[axis] = s_shape[axis] + d_shape[axis]
            # Determine the location where the new data should be assigned
            slicing = tuple(slice(s_shape[ax]) for ax in range(0, axis)) + (-d_shape[axis], ...)

            # Assign Data
            self._dataset.resize(new_shape)  # resize for new data
            self._dataset[slicing] = data    # Assign data to the new location

    # Axes and Scales
    def make_scale(self, name: str = None) -> None:
        """Assigns this dataset as a scale with a scale name.

        Args:
            name: The name to make this scale.
        """
        if name is not None:
            self._scale_name = name
        if self._scale_name is not None and self.exists:
            with self:
                self._dataset.make_scale(self._scale_name)

    def attach_axis(self, dataset: h5py.Dataset, axis: int = 0) -> None:
        """Attaches an axis (scale) to this dataset.

        Args:
            dataset: The dataset to attach as an axis (scale).
            axis: The axis to attach the axis (scale) to.
        """
        if isinstance(dataset, HDF5Dataset):
            dataset = dataset._dataset

        with self:
            self._dataset.dims[axis].attach_scale(dataset)

    def detach_axis(self, dataset: h5py.Dataset, axis: int = 0) -> None:
        """Detaches an axis (scale) from this dataset.

        Args:
            dataset: The dataset to detach as an axis (scale).
            axis: The axis to detach the axis (scale) from.
        """
        if isinstance(dataset, HDF5Dataset):
            dataset = dataset._dataset

        with self:
            self._dataset.dims[axis].detach_scale(dataset)
