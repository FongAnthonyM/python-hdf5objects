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
import copy
import pathlib
from typing import Any
import warnings
import weakref

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


# Definitions #
# Classes #
class DatasetMap(HDF5Map):
    """A general map for HDF5 Datasets.

    Class Attributes:
        default_attributes_type: The default type for attribute objects in this map.
        default_dtype: The default dtype of the dataset this class will map.
        default_casting_kwargs: The default keyword arguments for the dtype casting.

    Attributes:
        _dtypes: The dtypes of this dataset if it has multiple data types.
        dtypes_dict: A mapping of the data type names to their type index.
        casting_kwargs: The keyword arguments for the dtype casting.

    Args:
        name: The name of this map.
        type_: The type of the hdf5 object this map represents.
        attribute_names: The name map of python name vs hdf5 name of the attribute.
        attributes: The default values of the attributes of the represented hdf5 object.
        map_names: The name map of python name vs hdf5 name of the maps contained within this map.
        maps: The nested maps.
        parent: The parent of this map.
        component_types: The components to add to the HDF5Object when created.
        component_kwargs: The keyword arguments for the components.
        dtype: The dtype of the dataset this object will map.
        casting_kwargs: The keyword arguments for the dtype casting.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the object this map represents.
    """
    __slots__ = HDF5Map.__slots__ | {"_dtypes", "dtypes_dict", "casting_kwargs"}
    default_attributes_type = HDF5Attributes
    default_dtype: np.dtype | str | tuple[tuple[str, type]] | None = None
    default_casting_kwargs: list[dict[str, Any]] | None = None
    default_axis_maps: list[dict[str, Any]] = []

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        name: str | None = None,
        type_: type | None = None,
        attribute_names: Mapping[str, str] | None = None,
        attributes: Mapping[str, Any] | None = None,
        map_names: Mapping[str, str] | None = None,
        maps: Mapping[str, HDF5Map] | None = None,
        parent: str | None = None,
        component_types: dict[str, type] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any
    ) -> None:
        # New Attributes #
        self._dtypes: tuple[tuple[str, type]] = tuple()
        self.dtypes_dict: bidict = bidict()

        self.casting_kwargs: list[dict[str, Any]] | None = None
        self.axis_maps: list[dict[str, Any]] = copy.deepcopy(self.default_axis_maps)

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            name = name if name is not None else self.default_name
            parent = parent if parent is not None else self.default_parent
            self.construct(
                name=name,
                type_=type_,
                attribute_names=attribute_names,
                attributes=attributes,
                map_names=map_names,
                maps=maps,
                parent=parent,
                component_types=component_types,
                component_kwargs=component_kwargs,
                dtype=dtype,
                casting_kwargs=casting_kwargs,
                **kwargs,
            )

    @property
    def dtype(self) -> np.dtype | str | tuple[tuple[str, type]] | None:
        """The dtype of this object's type, stored in the kwargs."""
        return self.kwargs["dtype"]

    @dtype.setter
    def dtype(self, value: np.dtype | str | tuple[tuple[str, type]] | None) -> None:
        self.set_dtype(value)

    @property
    def dtypes(self) -> np.dtype | str | tuple[tuple[str, type]] | None:
        """The dtypes of this object's type, if it has multiple types."""
        return self._dtypes

    @dtypes.setter
    def dtypes(self, value: np.dtype | str | tuple[tuple[str, type]] | None) -> None:
        self.set_dtype(value)

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        name: str | None = None,
        type_: type | None = None,
        attribute_names: Mapping[str, str] | None = None,
        attributes: Mapping[str, Any] | None = None,
        map_names: Mapping[str, str] | None = None,
        maps: Mapping[str, HDF5Map] | None = None,
        parent: str | None = None,
        component_types: dict[str, type] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object, setting attributes and sets nested maps' parents.

        Args:
            name: The name of this map.
            type_: The type of the hdf5 object this map represents.
            attribute_names: The name map of python name vs hdf5 name of the attribute.
            attributes: The default values of the attributes of the represented hdf5 object.
            map_names: The name map of python name vs hdf5 name of the maps contained within this map.
            maps: The nested maps.
            parent: The parent of this map.
            component_types: The components to add to the HDF5Object when created.
            component_kwargs: The keyword arguments for the components.
            dtype: The dtype of the dataset this object will map.
            casting_kwargs: The keyword arguments for the dtype casting.
            **kwargs: The keyword arguments for the object this map represents.
        """
        super().construct(
            name=name,
            type_=type_,
            attribute_names=attribute_names,
            attributes=attributes,
            map_names=map_names,
            maps=maps,
            parent=parent,
            component_types=component_types,
            component_kwargs=component_kwargs,
            **kwargs,
        )

        if dtype is None:
            dtype = self.default_dtype

        if dtype is not None:
            self.set_dtype(dtype)

        if casting_kwargs is not None:
            self.casting_kwargs = casting_kwargs
        elif self.default_casting_kwargs is None:
            self.casting_kwargs = ([{}] * len(self.dtypes))
        else:
            self.casting_kwargs = self.default_casting_kwargs.copy()

    def create_object(self, **kwargs: Any) -> Any:
        """Creates the object that this map is for.

        Args:
            **kwargs: The keyword arguments for the object.

        Returns:
            The HDF5Object that this map is for.
        """
        temp_kwargs = self.kwargs | kwargs

        require = temp_kwargs.get("require", False)
        if require and "data" not in temp_kwargs and ("shape" not in temp_kwargs or "maxshape" not in temp_kwargs):
            # Need to warn and skip if these components are missing.
            warnings.warn("Cannot build dataset without data or shape and maxshape - skipping.")
            return None

        if "map_" not in kwargs:
            kwargs["map_"] = self

        object_ = self.type(**kwargs)
        self.weak_object = weakref.ref(object_)

        return object_

    # Setters
    def set_dtype(self, dtype: tuple[tuple[str, type]] | None = None):
        """Sets the dataset to have multiple types with the give types.

        The caster allows python types to be given and translated to an HDF5 compatible type.

        Args:
            dtype: The dtype which this dataset will contain.
        """
        self.dtypes_dict.clear()
        if not isinstance(dtype, str) and not isinstance(dtype, np.dtype):
            self._dtypes = dtype
            self.dtypes_dict.update({name: i for i, (name, _) in enumerate(dtype)})
            dtype = list((name, self.caster.map_type(type_)) for name, type_ in dtype)
            if self.casting_kwargs is None or len(self.casting_kwargs) != len(self._dtypes):
                if self.default_casting_kwargs is None or len(self.default_casting_kwargs) != len(self._dtypes):
                    self.casting_kwargs = ([{}] * len(self._dtypes))
                else:
                    self.casting_kwargs = self.default_casting_kwargs

        self.kwargs["dtype"] = dtype

    def set_children(self) -> None:
        """Sets the nested maps parents to the correct hierarchy."""
        for name, child in self.maps.items():
            child.set_parent(parent=self.parent)
            child_name = child.name
            if child_name is None:
                child_name = self.map_names.get(name, self.sentinel)
                if child_name is self.sentinel:
                    child_name = name
                child.name = child_name
            child.set_children()

    def print_tree(self, indent: int = 0) -> None:
        """Prints the entire map.

        Args:
            indent: The number of space to print between each layer.
        """
        if self.attribute_names:
            print(f"{' ' * indent}  Attributes:")
            for name in self.attribute_names.values():
                print(f"{' ' * indent}      {name}")
        if self.axis_maps:
            print(f"{' ' * indent}  Axes:")
            for i, dim in enumerate(self.axis_maps):
                print(f"{' ' * indent}    Dimension {i}:")
                for name, map_ in dim.items():
                    print(f"{' ' * indent}    +  {name}: {map_.full_name} {map_.type}")
                    map_.print_tree(indent=indent+5)


class HDF5Dataset(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 dataset and gives more functionality.

    Class Attributes:
        _wrapped_types: A list of either types or objects to set up wrapping for.
        _wrap_attributes: Attribute names that will contain the objects to wrap where the resolution order is descending
            inheritance.
        default_map: The map of this dataset.
        default_axis_type: The default axis type when making an axis.

    Attributes:
        _dataset: The HDF5 dataset to wrap.
        _scale_name: The name of this dataset if it is a scale.
        attributes: The attributes of this dataset.

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
        scale_name: Makes this data an axis with this name.
        casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.
        component_kwargs: The keyword arguments for the components.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments to construct the base HDF5 dataset.
    """
    _wrapped_types: list[type | object] = [h5py.Dataset]
    _wrap_attributes: list[str] = ["dataset"]
    default_map: HDF5Map = DatasetMap()
    default_axis_type: Any = None

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
        scale_name: str | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._dataset: h5py.Dataset | None = None
        self._scale_name: str | None = None
        self.attributes: HDF5Attributes | None = None

        self.axes: list[dict[str, Any]] = []
        self.axes_kwargs: Iterable[dict[str, dict[str, Any]]] = []

        # Parent Attributes #
        super().__init__(init=False)

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
                scale_name=scale_name,
                casting_kwargs=casting_kwargs,
                component_kwargs=component_kwargs,
                **kwargs,
            )

    @property
    def dtypes(self) -> tuple[tuple[str, type]] | None:
        """The dtypes of this object if it has multiple types"""
        return self.map.dtypes

    @dtypes.setter
    def dtypes(self, value: np.dtype | str | tuple[tuple[str, type]] | None) -> None:
        self.map.dtypes = value

    @property
    def dtypes_dict(self) -> bidict:
        """The dictionary mapping of the dtypes of this object if it has multiple types"""
        return self.map.dtypes_dict

    @property
    def casting_kwargs(self) -> list[dict[str, Any]]:
        """The keyword arguments for the dtype casting."""
        return self.map.casting_kwargs

    @casting_kwargs.setter
    def casting_kwargs(self, value: list[dict[str, Any]]) -> None:
        self.map.casting_kwargs = value

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
        scale_name: str | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
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
            scale_name: Makes this data an axis with this name.
            casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.
            component_kwargs: The keyword arguments for the components.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.
        """
        if file is None and isinstance(dataset, str):
            raise ValueError("A file must be given if giving dataset name")

        super().construct(name=name, map_=map_, file=file, parent=parent)

        if self.map.type is None:
            self.map.type = self.__class__

        if dtype is not None:
            self.map.set_dtype(dtype)

        if dataset is not None:
            self.set_dataset(dataset)

        if kwargs is not None:
            self.kwargs.update(kwargs)

        if scale_name is not None:
            self._scale_name = scale_name

        if component_kwargs is None:
            self.construct_components()
        else:
            self.construct_components(**component_kwargs)
        self.construct_attributes()

        if load and self.exists:
            self.load()

        if require or data is not None:
            self.require(name=self._full_name, data=data, **kwargs)

    def construct_attributes(self, load: bool = False, require: bool = False) -> None:
        """Creates the attributes for this dataset.

        Args:
            load: Determines if this object will load the attribute values from the file on construction.
            require: Determines if this object will create and fill the attributes in the file on construction.
        """
        self.attributes = self.map.attributes_type(
            name=self._full_name,
            map_=self.map,
            file=self.file,
            load=load,
            require=require,
        )

    # File
    def load_axes(self) -> None:
        """Loads the axes from file."""
        mapped_dims = len(self.map.axis_maps)

        with self:
            self.axes.extend([{}] * (len(self._dataset.dims) - len(self.axes)))
            for i, dim in enumerate(self._dataset.dims):
                raw_axes = dict(dim.items())

                if i < mapped_dims:
                    mapped_axes = self.map.axis_maps[i]
                else:
                    mapped_axes = {}

                for name, axis_map in mapped_axes.items():
                    dataset = raw_axes.get(name, None)
                    if dataset is not None:
                        self.axes[i][name] = axis_map.get_object(dataset=dataset, scale_name=name, file=self.file)
                        del raw_axes[name]
                    elif False:  # Todo: Set verbose because this does not need to always warn.
                        warnings.warn("A dataset's axis is missing.")

                for name, dataset in raw_axes.items():
                    self.axes[i][name] = self.default_axis_type(dataset=dataset, scale_name=name, file=self.file)

    def load(self) -> None:
        """Loads this dataset which is just loading the attributes."""
        self.attributes.load()
        self.load_axes()

    def refresh(self) -> None:
        """Reloads the dataset and attributes."""
        with self:
            self._dataset.refresh()
        self.attributes.refresh()
        self.get_shape.clear_cahce()
        self.get_all_data.clear_cache()
        for dim in self.axes:
            for axis in dim.values():
                axis.refresh()

    # Caching
    def clear_all_caches(self, **kwargs: Any) -> None:
        """Clears all caches in this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the clear caches method.
        """
        self.clear_caches(**kwargs)
        self.attributes.clear_caches(**kwargs)
        for dim in self.axes:
            for axis in dim.values():
                axis.clear_all_caches(**kwargs)
        for component in self.components.values():
            clear_all_chaches = getattr(component, "clear_all_caches", None)
            if clear_all_chaches is not None:
                clear_all_chaches()

    def enable_all_caching(self, **kwargs: Any) -> None:
        """Enables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the enable caching method.
        """
        self.enable_caching(**kwargs)
        self.attributes.enable_caching(**kwargs)
        for dim in self.axes:
            for axis in dim.values():
                axis.enable_all_caching(**kwargs)

    def disable_all_caching(self, **kwargs: Any) -> None:
        """Disables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the disable caching method.
        """
        self.disable_caching(**kwargs)
        self.attributes.disable_caching(**kwargs)
        for dim in self.axes:
            for axis in dim.values():
                axis.disable_all_caching(**kwargs)

    def timeless_all_caching(self, **kwargs: Any) -> None:
        """Allows timeless caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timeless caching method.
        """
        self.timeless_caching(**kwargs)
        self.attributes.timeless_caching(**kwargs)
        for dim in self.axes:
            for axis in dim.values():
                axis.timeless_all_caching(**kwargs)
        
    def timed_all_caching(self, **kwargs: Any) -> None:
        """Allows timed caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timed caching method.
        """
        self.timed_caching(**kwargs)
        self.attributes.timed_caching(**kwargs)
        for dim in self.axes:
            for axis in dim.values():
                axis.timed_all_caching(**kwargs)

    def set_all_lifetimes(self, lifetime: int | float | None, **kwargs: Any) -> None:
        """Sets the lifetimes on this object and all contained objects.

        Args:
            lifetime: The lifetime to set all the caches to.
            **kwargs: The keyword arguments for the lifetime caching method.
        """
        self.set_lifetimes(lifetime=lifetime, **kwargs)
        self.attributes.set_lifetimes(lifetime=lifetime, **kwargs)
        for dim in self.axes:
            for axis in dim.values():
                axis.set_all_lifetimes(lifetime=lifetime, **kwargs)

    # Item Data Types
    def item_to_dict(self, item: Any, casting_kwargs: list[dict[str | Any]] | None = None) -> dict:
        """Translates an item of the dataset's type to a dictionary that multi-type.

        Args:
            item: The item to translate.
            casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.

        Returns:
            The dictionary representation of the item.
        """
        if isinstance(item, np.ndarray):
            item = item[0]

        if casting_kwargs is None:
            casting_kwargs = self.casting_kwargs

        types = zip(self.dtypes, casting_kwargs)
        return {name: self.caster.cast_to(type_, item[i], **kwargs) for i, ((name, type_), kwargs) in enumerate(types)}

    def dict_to_item(self, dict_: dict) -> Any:
        """Translates a dictionary of a multi-type to an item that can be added to the dataset.

        Args:
            dict_: The dictionary to translate.

        Returns:
            The item representation of the dictionary.
        """
        return tuple(self.caster.cast_from(dict_[name]) for i, (name, _) in enumerate(self.dtypes))

    # Getters/Setters
    def set_map(self, map_: HDF5Map) -> None:
        """Changes the current map with a different one.

        Args:
            map_: The map to replace the current map.
        """
        super().set_map(map_=map_)
        map_.object = self
        if map_.name is None:
            map_.set_name(self._full_name)
        self.file[self._parent].map[self._name] = map_
        self.attributes.set_map(map_)
        self.construct_components()

    def get_item(self, key: Any) -> Any:
        """Gets an item or items from the dataset.

        Args:
            key: The key to get an item or items from the dataset.

        Returns:
            The item or items requested.
        """
        return getattr(self, self._wrap_attributes[0])[key]

    def get_item_dict(self, index: int | tuple | h5py.Reference) -> dict:
        """Gets an item from the given an index and translates a multi-type into a dictionary.

        Args:
            index: The index of the item to translate into a dictionary.

        Returns:
            The item of interest as a dictionary.
        """
        return self.item_to_dict(self[index])

    def get_item_dicts_iter(self, casting_kwargs: list[dict[str | Any]] | None = None) -> Iterable:
        """Gets the item dictionaries as an iterable.

        Args:
            casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.
        """
        if casting_kwargs is None:
            casting_kwargs = self.casting_kwargs

        types = zip(self.dtypes, casting_kwargs)
        return ({name: self.caster.cast_to(type_, item[i], **kwargs) for i, ((name, type_), kwargs) in enumerate(types)}
                for item in self[...])

    def set_item(self, key: Any, value: Any) -> None:
        """Sets an item or items from the dataset.

        Args:
            key: The key to set an item or items from the dataset.
            value: The value or values to set in the dataset.
        """
        getattr(self, self._wrap_attributes[0])[key] = value
        self.clear_all_caches()

    def set_item_dict(self, index: int | tuple | h5py.Reference, dict_: dict) -> None:
        """Sets an item from the given an index to a translated a multi-type from a dictionary.

        Args:
            index: The index of the item to set.
            dict_: The dictionary of a multi-type to set to.
        """
        self[index] = self.dict_to_item(self.get_item_dict(index) | dict_)
        self.clear_all_caches()

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

    def get_field(self, name: str) -> np.ndarray:
        """Gets all the data of a dtype field in the dataset.

        Returns:
            All the data in the dtype field.
        """
        with self:
            return self._dataset[name]

    # Data Modification
    def create_data(self, name: str | None = None, **kwargs: Any) -> None:
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
            if self._scale_name is not None:
                self._dataset.make_scale(self._scale_name)

    def create_axis(self, dim: int, scale_name: str, **kwargs: Any) -> "HDF5Dataset":
        """Creates and fills an axis for this dataset, gives an error if any already exists.

        Args:
            dim: The dimension of the axis.
            scale_name: The name of the axis to create.
            **kwargs: The keyword arguments for creating the axis.
        """
        if len(self.axes) < dim + 1:
            self.axes.extend([{}] * (dim + 1 - len(self.axes)))

        old_kwargs = self.axes_kwargs[dim].get(scale_name, {}) if i < len(self.axes_kwargs) else {}
        temp_kwargs = {
            "name": f"{self._full_name}_{scale_name}",
            "scale_name": name, "require": True,
            "file": self.file,
        }
        if "data" not in kwargs and "data" not in old_kwargs and self.exists:
            temp_kwargs["size"] = self.shape[dim]
        new_kwargs = temp_kwargs | old_kwargs | kwargs

        self.axes[i][name] = axis = self.map.axis_maps[dim][scale_name].create_object(**new_kwargs)
        self._dataset.dims[i].attach_scale(axis._dataset)
        return axis

    def create_axes(self, axes_kwargs: Iterable[dict[str, dict[str, Any]]] = ()) -> None:
        """Creates and fills the axes for this dataset, gives an error if any already exists.

        Args:
            axes_kwargs: The keyword arguments for creating the axes objects.
        """
        if len(self.axes) < len(self.map.axis_maps):
            self.axes.extend([{}] * (len(self.map.axis_maps) - len(self.axes)))

        temp_kwargs = {"require": True, "file": self.file}
        new_kwargs_len = len(axes_kwargs)
        old_kwargs_len = len(self.axes_kwargs)
        for i, dim in enumerate(self.map.axis_maps):
            for name, axis_map in dim.items():
                new_kwargs = axes_kwargs[i].get(name, {}) if i < new_kwargs_len else {}
                old_kwargs = self.axes_kwargs[i].get(name, {}) if i < old_kwargs_len else {}
                temp_kwargs["name"] = f"{self._full_name}_{name}"
                temp_kwargs["scale_name"] = name
                if "data" not in new_kwargs and "data" not in old_kwargs and self.exists:
                    temp_kwargs["component_kwargs"] = {"axis": {"size": self.shape[i]}}
                kwargs = temp_kwargs | old_kwargs | new_kwargs
                self.axes[i][name] = axis = axis_map.create_object(**kwargs)
                self._dataset.dims[i].attach_scale(axis._dataset)

    def create(
        self,
        name: str | None = None,
        axes_kwargs: Iterable[dict[str, dict[str, Any]]] = (),
        component_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> "HDF5Dataset":
        """Creates and fills the data and axes, gives an error if they already exists.

        Args:
            name: The name of the dataset.
            axes_kwargs: The keyword arguments for creating the axes objects.
            component_kwargs: The keyword arguments for the components' create methods.
            **kwargs: The keyword arguments for constructing a HDF5 Dataset.

        Returns:
            This object.
        """
        self.create_data(name=name, **kwargs)
        self.create_axes(axes_kwargs=axes_kwargs)
        self.create_components(**component_kwargs)
        return self

    def require_data(self, name: str | None = None, **kwargs: Any) -> "HDF5Dataset":
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
                self.kwargs.update(kwargs)
                self._dataset = self.file._file.create_dataset(name=self._full_name, **self.kwargs)
                if self.file._file.swmr_mode:
                    if self.file.allow_swmr_create:
                        self.file.close()
                        self.file.open(mode='a')
                        self.file._file.swmr_mode = True
                    else:
                        raise RuntimeError("Creating a new dataset with SWMR mode on causes issues")
                self.attributes.construct_attributes()
                if self._scale_name is not None:
                    self._dataset.make_scale(self._scale_name)
            else:
                self._dataset = self.file._file[self._full_name]
                data = kwargs.get("data", None)
                if data is not None:
                    self.replace_data(data=data)

    def require_axis(self, dim: int, scale_name: str, **kwargs: Any) -> "HDF5Dataset":
        """Creates and fills an axis for this dataset if it does not exist.

        Args:
            dim: The dimension of the axis.
            scale_name: The name of the axis to create.
            **kwargs: The keyword arguments for creating the axis.
        """
        if len(self.axes) < dim + 1:
            axis = None
        else:
            axis = self.axes[dim].get(scale_name, None)

        if axis is None:
            if len(self.axes) < dim + 1:
                self.axes.extend([{}] * (dim + 1 - len(self.axes)))

            old_kwargs = self.axes_kwargs[dim].get(scale_name, {}) if dim < len(self.axes_kwargs) else {}
            temp_kwargs = {
                "name": f"{self._full_name}_{scale_name}",
                "scale_name": name, "require": True,
                "file": self.file,
            }
            if "data" not in kwargs and "data" not in old_kwargs and self.exists:
                temp_kwargs["size"] = self.shape[dim]
            new_kwargs = temp_kwargs | old_kwargs | kwargs
            self.axes[dim][name] = axis = self.map.axis_maps[dim][scale_name].get_object(**new_kwargs)
            self._dataset.dims[dim].attach_scale(axis._dataset)

        return axis

    def require_axes(self, axes_kwargs: Iterable[dict[str, dict[str, Any]]] = ()) -> None:
        """Creates and fills the axes for this dataset if any do not exists.

        Args:
            axes_kwargs: The keyword arguments for creating the axes objects.
        """
        if len(self.axes) < len(self.map.axis_maps):
            self.axes.extend([{}] * (len(self.map.axis_maps) - len(self.axes)))

        temp_kwargs = {"require": True, "file": self.file}
        new_kwargs_len = len(axes_kwargs)
        old_kwargs_len = len(self.axes_kwargs)
        for i, dim in enumerate(self.map.axis_maps):
            for name, axis_map in dim.items():
                new_kwargs = axes_kwargs[i].get(name, {}) if i < new_kwargs_len else {}
                old_kwargs = self.axes_kwargs[i].get(name, {}) if i < old_kwargs_len else {}
                temp_kwargs["name"] = f"{self._full_name}_{name}"
                temp_kwargs["scale_name"] = name
                if "data" not in new_kwargs and "data" not in old_kwargs and self.exists:
                    temp_kwargs["component_kwargs"] = {"axis": {"size": self.shape[i]}}
                kwargs = temp_kwargs | old_kwargs | new_kwargs
                self.axes[i][name] = axis = axis_map.get_object(**kwargs)
                self._dataset.dims[i].attach_scale(axis._dataset)

    def require(
        self,
        name: str | None = None,
        axes_kwargs: Iterable[dict[str, dict[str, Any]]] = (),
        component_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> "HDF5Dataset":
        """Creates and fills the data and axes if they does not exist.

        Args:
            name: The name of the dataset.
            axes_kwargs: The keyword arguments for creating the axes objects.
            component_kwargs: The keyword arguments for the components' create methods.
            **kwargs: The keyword arguments for constructing a HDF5 Dataset.

        Returns:
            This object.
        """
        self.require_data(name=name, **kwargs)
        self.require_axes(axes_kwargs=axes_kwargs)
        self.require_components(**component_kwargs)
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
            self.clear_all_caches()

    def set_data_compontents(self, **component_kwargs: dict[str, Any]) -> None:
        """Sets the data of the components of this dataset.

        Args:
           **component_kwargs: The keyword arguments for the components' set data methods as keywords.
        """
        for name, component in self.components.items():
            kwargs = component_kwargs.get(name, {})
            component.set_data_component(**kwargs)

    def set_data_exclusively(self, data: np.ndarray, **kwargs: Any) -> None:
        """Sets the data by either creating it or replacing it.

        Args:
            data: The data to fill the dataset with.
            **kwargs: The keyword arguments for creating the dataset.
        """
        if self.exists:
            self.replace_data(data=data)
        else:
            self.require_data(data=data, **kwargs)

    def set_data(self, data: np.ndarray, component_kwargs: dict[str, Any] = {}, **kwargs: Any) -> None:
        """Sets the data by either creating it or replacing it.

        Args:
            data: The data to fill the dataset with.
            component_kwargs: The keyword arguments for the components' create methods.
            **kwargs: The keyword arguments for creating the dataset.
        """
        if self.exists:
            self.replace_data(data=data)
            self.set_data_compontents(**component_kwargs)
        else:
            self.require(data=data, **kwargs)

    def append_data(self, data: np.ndarray, axis: int = 0) -> None:
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
            self.clear_all_caches()

    def append_data_item_dict(self, dict_: dict, axis: int = 0) -> None:
        """Appends a dictionary which would represent a single item to the dataset.

        Args:
            dict_: The dictionary to add as an item to the dataset.
            axis: The axis to add the dictionary along.
        """
        self.append_data(np.array(self.dict_to_item(dict_), dtype=self.dtype), axis=axis)

    def append_components(self, **component_kwargs: dict[str, Any]) -> None:
        """Appends data to the components of this dataset.

        Args:
           **component_kwargs: The keyword arguments for the components' append methods as keywords.
        """
        for name, component in self.components.items():
            kwargs = component_kwargs.get(name, {})
            component.append_component(**kwargs)

    def append(self, data: np.ndarray, axis: int = 0, component_kwargs: dict[str, Any] = {}) -> None:
        """Append data to the dataset along a specified axis.

        Args:
            data: The data to append.
            axis: The axis to append the data along.
            component_kwargs: The keyword arguments for the components' append methods as keywords.
        """
        self.append_data(data=data, axis=axis)
        self.append_components(**component_kwargs)

    def append_item_dict(self, dict_: dict, axis: int = 0) -> None:
        """Appends a dictionary which would represent a single item to the dataset.

        Args:
            dict_: The dictionary to add as an item to the dataset.
            axis: The axis to add the dictionary along.
        """
        self.append(np.array(self.dict_to_item(dict_), dtype=self.dtype), axis=axis)

    def extend_data_item_dicts(self, iter_: Iterable[dict], axis: int = 0) -> None:
        """Extends the dataset with an iterable of dictionaries which would represent single items.

        Args:
            iter_: An iterable of dictionaries to append to the dataset.
            axis: The axis to extend the dictionaries to.
        """
        self.append_data(np.fromiter((self.dict_to_item(item) for item in iter_), dtype=list(self._dtype)), axis=axis)

    def insert_data(self, data: np.ndarray, index: int | slice | Iterable[int], axis: int = 0) -> None:
        """Insert data to the dataset along a specified axis.

        Args:
            data: The data to append.
            index: The index or slice to insert the data into.
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
                raise ValueError("Cannot insert with two different rank shapes.")

            # Determine the new shape of the dataset
            new_shape = list(s_shape) if s_shape.any() else d_shape.copy()
            new_shape[axis] = s_extension = s_shape[axis] + d_extension
            # Determine the location where the new data should be assigned
            slicing = (slice(None),) * axis + (slice(s_shape[axis], s_extension),)
            old_data = self._dataset[...]

            # Assign Data
            self._dataset.resize(new_shape)  # resize for new data
            self._dataset[...] = np.insert(old_data, index, data, axis)    # Assign data to the new location
            self.clear_all_caches()

    def insert_data_item_dict(self, dict_: dict, index: int | slice | Iterable[int], axis: int = 0) -> None:
        """Inserts a dictionary which would represent a single item to the dataset.

        Args:
            dict_: The dictionary to add as an item to the dataset.
            index: The index or slice to insert the data into.
            axis: The axis to add the dictionary along.
        """
        self.insert_data(np.array(self.dict_to_item(dict_), dtype=self.dtype), axis=axis, index=index)

    def insert_components(self, index: int | slice | Iterable[int], **component_kwargs: dict[str, Any]) -> None:
        """Appends data to the components of this dataset.

        Args:
           **component_kwargs: The keyword arguments for the components' append methods as keywords.
        """
        for name, component in self.components.items():
            kwargs = component_kwargs.get(name, {})
            component.append_component(index=index, **kwargs)

    def insert(
        self,
        data: np.ndarray,
        index: int | slice | Iterable[int],
        axis: int = 0,
        component_kwargs: dict[str, Any] = {},
    ) -> None:
        """Append data to the dataset along a specified axis.

        Args:
            data: The data to append.
            index: The index or slice to insert the data into.
            axis: The axis to append the data along.
            component_kwargs: The keyword arguments for the components' append methods as keywords.
        """
        self.insert_data(data=data, index=index, axis=axis)
        self.insert_components(index=index, **component_kwargs)

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

    def attach_axis(self, dataset: "HDF5Dataset", axis: int = 0, scale_name: str | None = None) -> None:
        """Attaches an axis (scale) to this dataset.

        Args:
            dataset: The dataset to attach as an axis (scale).
            axis: The axis to attach the axis (scale) to.
            scale_name: Set or override the scale name of the axis.
        """
        if len(self.axes) < axis + 1:
            self.axes.extend([{}] * (axis + 1 - len(self.axes)))

        if scale_name is None:
            scale_name = dataset.scale_name
        else:
            dataset.scale_name = scale_name

        self.axes[axis][scale_name] = dataset

        with self:
            self._dataset.dims[axis].attach_scale(dataset._dataset)

    def detach_axis(self, dataset: "HDF5Dataset", axis: int = 0) -> None:
        """Detaches an axis (scale) from this dataset.

        Args:
            dataset: The dataset to detach as an axis (scale).
            axis: The axis to detach the axis (scale) from.
        """
        del self.axes[axis][dataset.scale_name]

        with self:
            self._dataset.dims[axis].detach_scale(dataset._dataset)


# Assign Cyclic Definitions
DatasetMap.default_type = HDF5Dataset
