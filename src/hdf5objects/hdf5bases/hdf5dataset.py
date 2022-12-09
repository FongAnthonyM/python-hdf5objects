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
from collections.abc import Mapping, Iterable
import pathlib
from typing import Any
import warnings

# Third-Party Packages #
from bidict import bidict
from baseobjects import singlekwargdispatchmethod
from baseobjects.cachingtools import timed_keyless_cache
import h5py
import numpy as np

# Local Packages #
from .hdf5map import HDF5Map
from .hdf5baseobject import HDF5BaseObject
from .hdf5attributes import HDF5Attributes
from .hdf5caster import HDF5Caster


# Definitions #
# Classes #
class DatasetMap(HDF5Map):
    """A general map for HDF5 Datasets."""
    default_attributes_type = HDF5Attributes

    # Instance Methods
    # Constructors/Destructors
    def construct_object(self, **kwargs: Any) -> Any:
        """Constructs the object that this map is for.

        Args:
            **kwargs: The keyword arguments for the object.

        Returns:
            The HDF5Object that this map is for.
        """
        temp_kwargs = self.kwargs | kwargs

        if "require" in temp_kwargs and "data" not in temp_kwargs and (
                "shape" not in temp_kwargs or "maxshape" not in temp_kwargs):
            # Need to warn and skip if these components are missing.
            warnings.warn("Cannot build dataset without data or shape and maxshape - skipping.")
            return None

        self.object = self.type(**temp_kwargs)

        return self.object


class HDF5Dataset(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 dataset and gives more functionality.

    Class Attributes:
        _wrapped_types: A list of either types or objects to set up wrapping for.
        _wrap_attributes: Attribute names that will contain the objects to wrap where the resolution order is descending
            inheritance.
        default_map: The map of this dataset.

    Attributes:
        _dataset: The HDF5 dataset to wrap.
        _scale_name: The name of this dataset if it is a scale.
        attributes: The attributes of this dataset.
        kwargs: The kwargs to use when creating the dataset.
        _dtypes: The dtypes of this dataset if it has multiple data types.
        _types_dict: A mapping of the data type names to their type index.
        _dtype: This dataset's data type.

        casting_kwargs: list[dict[str, Any]] | None = None

    Args:
        data: The data to fill in this dataset.
        dataset: The HDF5 dataset to build this dataset around.
        name: The HDF5 name of this object.
        map_: The map for this HDF5 object.
        file: The file object that this dataset object originates from.
        load: Determines if this object will load the dataset from the file on construction.
        require: Determines if this object will create and fill the dataset in the file on construction.
        parent: The HDF5 name of the parent of this HDF5 object.
        dtype: The dtype of this dataset.
        casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments to construct the base HDF5 dataset.
    """
    _wrapped_types: list[type | object] = [h5py.Dataset]
    _wrap_attributes: list[str] = ["dataset"]
    default_map: HDF5Map = DatasetMap()
    default_dtype: np.dtype | str | tuple[tuple[str, type]] | None = None
    default_casting_kwargs: list[dict[str, Any]] | None = None
    caster = HDF5Caster

    # Magic Methods
    # Constructors/Destructors
    def __init__(
        self,
        data: np.ndarray | None = None,
        dataset: h5py.Dataset | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        require: bool = False,
        parent: str | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(file=file, init=False)

        # New Attributes #
        self._dataset: h5py.Dataset | None = None
        self._scale_name: str | None = None
        self.attributes: HDF5Attributes | None = None
        self.kwargs: dict[str, Any] | None = None

        self._dtypes: tuple[tuple[str, type]] = tuple()
        self._types_dict: bidict = bidict()
        self._dtype: np.dtype | str | tuple[tuple[str, type]] | None = None

        self.casting_kwargs: list[dict[str, Any]] | None = None

        # Object Construction #
        if init:
            self.construct(
                data=data,
                dataset=dataset,
                name=name,
                map_=map_,
                file=file,
                load=load,
                parent=parent,
                require=require,
                dtype=dtype,
                casting_kwargs=casting_kwargs,
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

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            dict: A dictionary of this object's attributes.
        """
        state = super().__getstate__()
        state["_dataset"] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Builds this object based on a dictionary of corresponding attributes.

        Args:
            state: The attributes to build this object from.
        """
        super().__setstate__(state=state)
        with self.file.temp_open:
            self._dataset = self.file._file[self._full_name]

    # Container Methods
    def __getitem__(self, key: Any) -> Any:
        """Ensures HDF5 object is open for getitem"""
        with self:
            return self.get_item(key=key)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Ensures HDF5 object is open for setitem"""
        with self:
            getattr(self, self._wrap_attributes[0])[key] = value

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        dataset: h5py.Dataset | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        require: bool = False,
        parent: str | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The data to fill in this dataset.
            dataset: The HDF5 dataset to build this dataset around.
            name: The HDF5 name of this object.
            map_: The map for this HDF5 object.
            file: The file object that this dataset object originates from.
            load: Determines if this object will load the dataset from the file on construction.
            require: Determines if this object will create and fill the dataset in the file on construction.
            parent: The HDF5 name of the parent of this HDF5 object.
            dtype: The dtype of this dataset.
            casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.
        """
        if file is None and isinstance(dataset, str):
            raise ValueError("A file must be given if giving dataset name")

        super().construct(name=name, map_=map_, file=file, parent=parent)

        if self.map.type is None:
            self.map.type = self.__class__

        if dtype is None:
            if "dtype" in self.map.kwargs:
                dtype = self.map.kwargs["dtype"]
            else:
                dtype = self.default_dtype

        if dtype is not None:
            if not isinstance(dtype, str) and not isinstance(dtype, np.dtype):
                self.set_types(dtype)
                dtype = list(self._dtype)
            else:
                self._dtype = dtype
            kwargs["dtype"] = dtype
        elif require and data is None:
            raise ValueError("Cannot build dataset without data or a given dtype.")

        if casting_kwargs is not None:
            self.casting_kwargs = casting_kwargs
        elif self.default_casting_kwargs is None:
            self.casting_kwargs = ([{}] * len(self._dtypes))
        else:
            self.casting_kwargs = self.default_casting_kwargs.copy()

        if dataset is not None:
            self.set_dataset(dataset)

        if kwargs is not None:
            self.kwargs = kwargs

        self.construct_attributes()

        if load and self.exists:
            self.load()

        if require or data is not None:
            self.require(name=self._full_name, **kwargs)

    def construct_attributes(self, map_: HDF5Map = None, load: bool = False, require: bool = False) -> None:
        """Creates the attributes for this dataset.

        Args:
            map_: The map to use to create the attributes.
            load: Determines if this object will load the attribute values from the file on construction.
            require: Determines if this object will create and fill the attributes in the file on construction.
        """
        if map_ is None:
            map_ = self.map
        self.attributes = map_.attributes_type(
            name=self._full_name,
            map_=map_,
            file=self.file,
            load=load,
            require=require,
        )

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

    # Caching
    def clear_all_caches(self, **kwargs: Any) -> None:
        """Clears all caches in this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the clear caches method.
        """
        self.attributes.clear_caches(**kwargs)
        self.clear_caches(**kwargs)

    def enable_all_caching(self, **kwargs: Any) -> None:
        """Enables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the enable caching method.
        """
        self.attributes.enable_caching(**kwargs)
        self.enable_caching(**kwargs)

    def disable_all_caching(self, **kwargs: Any) -> None:
        """Disables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the disable caching method.
        """
        self.attributes.disable_caching(**kwargs)
        self.disable_caching(**kwargs)

    def timeless_all_caching(self, **kwargs: Any) -> None:
        """Allows timeless caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timeless caching method.
        """
        self.attributes.timeless_caching(**kwargs)
        self.timeless_caching(**kwargs)
        
    def timed_all_caching(self, **kwargs: Any) -> None:
        """Allows timed caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timed caching method.
        """
        self.attributes.timed_caching(**kwargs)
        self.timed_caching(**kwargs)

    def set_all_lifetimes(self, lifetime: int | float | None, **kwargs: Any) -> None:
        """Sets the lifetimes on this object and all contained objects.

        Args:
            lifetime: The lifetime to set all the caches to.
            **kwargs: The keyword arguments for the lifetime caching method.
        """
        self.attributes.set_lifetimes(lifetime=lifetime, **kwargs)
        self.set_lifetimes(lifetime=lifetime, **kwargs)

    # Item Data Types
    def item_to_dict(self, item: Any) -> dict:
        """Translates an item of the dataset's type to a dictionary that multi-type.

        Args:
            item: The item to translate.

        Returns:
            The dictionary representation of the item.
        """
        types = zip(self._dtypes, self.casting_kwargs)
        return {name: self.caster.cast_to(type_, item[i], **kwargs) for i, ((name, type_), kwargs) in enumerate(types)}

    def dict_to_item(self, dict_: dict) -> Any:
        """Translates a dictionary of a multi-type to an item that can be added to the dataset.

        Args:
            dict_: The dictionary to translate.

        Returns:
            The item representation of the dictionary.
        """
        return tuple(self.caster.cast_from(dict_[name]) for i, (name, _) in enumerate(self._dtypes))

    # Getters/Setters
    def set_types(self, types: tuple[tuple[str, type]] | None = None):
        """Sets the dataset to have multiple types with the give types.

        The caster allows python types to be given and translated to an HDF5 compatible type.

        Args:
            types: The types which this dataset will contain.
        """
        self._dtypes = types
        self._types_dict.clear()
        self._types_dict.update({name: i for i, (name, _) in enumerate(types)})
        self._dtype = tuple((name, self.caster.map_type(type_)) for name, type_ in types)

    def get_item(self, key: Any) -> Any:
        """Gets an item or items from the dataset.

        Args:
            key: The key to get an item or items from the dataset.

        Returns:
            The item or items requested.
        """
        return getattr(self, self._wrap_attributes[0])[key]

    def get_item_dict(self, index: int | tuple) -> dict:
        """Gets an item from the given an index and translates a multi-type into a dictionary.

        Args:
            index: The index of the item to translate into a dictionary.

        Returns:
            The item of interest as a dictionary.
        """
        return self.item_to_dict(self[index])

    def set_item(self, key: Any, value: Any) -> None:
        """Sets an item or items from the dataset.

        Args:
            key: The key to set an item or items from the dataset.
            value: The value or values to set in the dataset.
        """
        getattr(self, self._wrap_attributes[0])[key] = value

    def set_item_dict(self, index: int | tuple, dict_: dict) -> None:
        """Sets an item from the given an index to a translated a multi-type from a dictionary.

        Args:
            index: The index of the item to set.
            dict_: The dictionary of a multi-type to set to.
        """
        self[index] = self.dict_to_item(dict_)

    @singlekwargdispatchmethod("dataset")
    def set_dataset(self, dataset: "HDF5Dataset") -> None:
        """Sets the wrapped dataset.

        Args:
            dataset: The dataset this object will wrap.
        """
        if isinstance(dataset, HDF5Dataset):
            if self.file is None:
                self.set_file(dataset.file)
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
        if self.file is None:
            self.set_file(dataset.file)
        self.set_name(dataset.name)
        self._dataset = dataset

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_shape(self) -> tuple[int]:
        """Gets the shape of the dataset.

        Returns:
            The shape of the dataset.
        """
        with self:
            return self._dataset.shape

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_all_data(self) -> np.ndarray:
        """Gets all the data in the dataset.

        Returns:
            All the data in the dataset.
        """
        with self:
            return self._dataset[...]

    # Data Modification
    def create(self, name: str | None = None, **kwargs: Any) -> "HDF5Dataset":
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

        with self.file.temp_open():
            self._dataset = self.file._file.create_dataset(name=self._full_name, **kwargs)
            if self.file._file.swmr_mode:
                if self.file.allow_swmr_create:
                    self.file.close()
                    self.file.open()
                    self.file._file.swmr_mode = True
                else:
                    raise RuntimeError("Creating a new dataset with SWMR mode on causes issues")
            self.attributes.construct_attributes()
            if self.scale_name is not None:
                self.make_scale()

        return self

    def require(self, name: str | None = None, **kwargs: Any) -> "HDF5Dataset":
        """Creates and fills the data if it does not exist.

        Args:
            name: The name of the dataset.
            **kwargs: The keyword arguments for constructing a HDF5 Dataset.

        Returns:
            This object.
        """
        if name is not None:
            self._name = name

        if "data" in kwargs and kwargs["data"] is not None:
            kwargs["shape"] = kwargs["data"].shape

        with self.file.temp_open():
            if not self.exists:
                self._dataset = self.file._file.create_dataset(name=self._full_name, **(self.kwargs | kwargs))
                if self.file._file.swmr_mode:
                    if self.file.allow_swmr_create:
                        self.file.close()
                        self.file.open(mode='a')
                        self.file._file.swmr_mode = True
                    else:
                        raise RuntimeError("Creating a new dataset with SWMR mode on causes issues")
                self.attributes.construct_attributes()
                if self.scale_name is not None:
                    self.make_scale()
            else:
                self._dataset = self.file._file[self._full_name]
                data = kwargs.get("data", None)
                if data is not None:
                    self.replace_data(data=data)

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
            s_shape = np.asarray(self._dataset.shape)
            d_shape = list(data.shape)
            if len(d_shape) == len(s_shape):
                d_extension = d_shape[axis]
            elif len(d_shape) == len(s_shape) - 1:
                d_extension = 1
                d_shape.insert(axis, 1)
            else:
                raise ValueError("Cannot append with two different rank shapes.")

            # Determine the new shape of the dataset
            new_shape = list(s_shape) if s_shape.any() else d_shape.copy()
            new_shape[axis] = s_extension = s_shape[axis] + d_extension
            # Determine the location where the new data should be assigned
            slicing = (slice(None),) * axis + (slice(s_shape[axis], s_extension),)

            # Assign Data
            self._dataset.resize(new_shape)  # resize for new data
            self._dataset[slicing] = data    # Assign data to the new location

    def append_item_dict(self, dict_: dict, axis: int = 0) -> None:
        """Appends a dictionary which would represent a single item to the dataset.

        Args:
            dict_: The dictionary to add as an item to the dataset.
            axis: The axis to add the dictionary along.
        """
        self.append(np.array(self.dict_to_item(dict_), dtype=list(self._dtype)), axis=axis)

    def extend_item_dicts(self, iter_: Iterable[dict], axis: int = 0) -> None:
        """Extends the dataset with an iterable of dictionaries which would represent single items.

        Args:
            iter_: An iterable of dictionaries to append to the dataset.
            axis: The axis to extend the dictionaries to.
        """
        self.append(np.fromiter((self.dict_to_item(item) for item in iter_), dtype=list(self._dtype)), axis=axis)

    # Axes and Scales
    def make_scale(self, name: str | None = None) -> None:
        """Assigns this dataset as a scale with a scale name.

        Args:
            name: The name to make this scale.
        """
        if not self.exists:
            raise ValueError("The dataset must exist before setting it as a scale.")

        if name is not None:
            self._scale_name = name

        if self._scale_name is not None:
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


# Assign Cyclic Definitions
DatasetMap.default_type = HDF5Dataset
