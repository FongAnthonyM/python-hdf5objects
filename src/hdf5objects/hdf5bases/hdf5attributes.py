"""hdf5attributes.py
An object for handling HDF5 attributes.
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
from collections.abc import Iterable, Iterator, ValuesView, ItemsView, Mapping
import pathlib
from typing import Any

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch
from baseobjects.cachingtools import timed_keyless_cache
from baseobjects.collections import TimedDict
import h5py
import numpy as np

# Local Packages #
from .hdf5map import HDF5Map
from .hdf5baseobject import HDF5BaseObject


# Definitions #
# Classes #
class HDF5Attributes(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 attribute manager and gives more functionality.

    Class Attributes:
        _wrapped_types: A list of either types or objects to set up wrapping for.
        _wrap_attributes: Attribute names that will contain the objects to wrap where the resolution order is descending
            inheritance.

    Attributes:
        _attribute_manager: The HDF5 attribute_manager to wrap.
        _attributes_dict: A cache to hold the attributes in.

    Args:
        attributes: The HDF5 attribute_manager to build this attribute_manager around.
        name: The HDF5 name of this object.
        map_: The map for this HDF5 object.
        file: The file object that this attribute object originates from.
        load: Determines if this object will load the attribute values from the file on construction.
        require: Determines if this object will create and fill the attributes in the file on construction.
        parent: The HDF5 name of the parent of this HDF5 object.
        component_kwargs: The keyword arguments for the components.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    _wrapped_types: list[type | object] = [h5py.AttributeManager]
    _wrap_attributes: list[str] = ["attribute_manager"]

    # Magic Methods #
    # Constructors/Destructors
    def __init__(
        self,
        attributes: h5py.AttributeManager | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        require: bool = False,
        parent: str | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._attribute_manager: h5py.AttributeManager | None = None
        self._attributes_dict: TimedDict = TimedDict()

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(
                attributes=attributes,
                name=name,
                map_=map_,
                file=file,
                load=load,
                require=require,
                parent=parent,
                component_kwargs=component_kwargs,
                **kwargs,
            )

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            dict: A dictionary of this object's attributes.
        """
        state = super().__getstate__()
        state["_attribute_manager"] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Builds this object based on a dictionary of corresponding attributes.

        Args:
            state: The attributes to build this object from.
        """
        super().__setstate__(state=state)
        with self.file.temp_open:
            self._attribute_manager = self.file._file[self._full_name]

    # Container Methods
    def __getitem__(self, key: str) -> Any:
        """Gets an attribute from this object."""
        return self.get_attributes()[self._parse_name(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets an attribute in this object and the within the file."""
        self.set_attribute(key, value)

    def __delitem__(self, key: str) -> None:
        """Deletes an attribute in this object and the within the file."""
        self.del_attribute(key)

    def __iter__(self) -> Iterator[Any]:
        """Returns an iterator of the attributes."""
        return self.get_attributes().__iter__()

    def __contains__(self, item: str) -> bool:
        """Checks if an attribute exists."""
        item = self._parse_name(item)
        with self:
            return item in self._attribute_manager

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        attributes: h5py.AttributeManager | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        require: bool = False,
        parent: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            attributes: The HDF5 attribute_manager to build this attribute_manager around.
            name: The HDF5 name of this object.
            map_: The map for this HDF5 object.
            file: The file object that this attribute object originates from.
            load: Determines if this object will load the attribute values from the file on construction.
            require: Determines if this object will create and fill the attributes in the file on construction.
            parent: The HDF5 name of the parent of this HDF5 object.
            **kwargs: Keyword arguments for inheritance.
        """
        super().construct(name=name, map_=map_, file=file, parent=parent, **kwargs)

        if attributes is not None:
            self.set_attribute_manager(attributes)

        if load and self.exists:
            self.load()

        if require:
            self.construct_attributes()

    def construct_attributes(self, map_: HDF5Map | None = None, override: bool = False) -> None:
        """Creates the attributes in the HDF5.

        Args:
            map_: The map to use to create the attributes.
            override: Determines if a value will be overridden if exists already.
        """
        if map_ is not None:
            self.map = map_

        with self:
            for name, value in self.map.attributes.items():
                name = self._parse_name(name)
                if name not in self._attribute_manager:
                    self._attribute_manager.create(name, value)
                elif override:
                    self._attribute_manager[name] = value

    def construct_components(
        self,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
    ) -> None:
        """Constructs or adds components.

        Args:
            component_kwargs: The keyword arguments for creating the components.
            component_types: Component class and their keyword arguments to instantiate.
            components: Components to add.
        """
        component_types = {} if component_types is None else component_types
        temp_types = self.default_component_types | self.map.attribute_component_types | component_types
        new_kwargs = {} if component_kwargs is None else component_kwargs
        default_components = {n: c(composite=self, **(k | new_kwargs.get(n, {}))) for n, (c, k) in temp_types.items()}
        self.components.update(default_components | self.components | {} if components is None else components)

    # Parsers
    def _parse_name(self, name: str) -> str:
        """Returns the hdf5 name of an attribute.

        Args:
            name: Either the python name of the attribute or the hdf5 name.

        Returns:
            The hdf5 name of the attribute.
        """
        new_name = self.map.attribute_names.get(name, self.sentinel)
        if new_name is not self.sentinel:
            name = new_name
        return name

    # Getters/Setters
    @singlekwargdispatch("attributes")
    def set_attribute_manager(self, attributes: h5py.AttributeManager | HDF5BaseObject) -> None:
        """Sets the wrapped attribute_manager.

        Args:
            attributes: The attribute_manager this object will wrap.
        """
        raise TypeError(f"{type(attributes)} is not a valid type for set_attribute_manager.")

    @set_attribute_manager.register
    def _(self, attributes: h5py.AttributeManager) -> None:
        """Sets the wrapped attribute_manager.

        Args:
            attributes: The attribute_manager this object will wrap.
        """
        if not attributes:
            raise ValueError("Attributes needs to be open")
        if self.file is None:
            self.set_file(attributes.file)
        self._name = attributes.name
        self._attribute_manager = attributes

    @set_attribute_manager.register
    def _(self, attributes: HDF5BaseObject) -> None:
        """Sets the wrapped attribute_manager.

        Args:
            attributes: An object that holds the attribute_manager this object will wrap.
        """
        self.set_file(attributes.file)
        self._name = attributes._name
        if isinstance(attributes, HDF5Attributes):
            self._attribute_manager = attributes._attribute_manager

    def get_attributes(self) -> dict[str, Any]:
        """Gets all file attributes from the HDF5 file.

        Returns:
            The file attributes.
        """
        a_names = set(self._attributes_dict.keys())
        missing = self.keys() - a_names
        if missing:
            with self._attributes_dict.pause_reset_timer():
                with self:
                    for name in missing:
                        self._attributes_dict[name] = self._attribute_manager[name]
        return dict(self._attributes_dict)

    def get_attribute(self, name: str, *args: Any) -> Any:
        """Gets an attribute from the HDF5 file.

        Args:
            name: The name of the file attribute to get.
            *args: An optional sentinel to return if a value is not present.

        Returns:
            The attribute requested.
        """
        name = self._parse_name(name)
        value = self._attributes_dict.get(name, self.sentinel)
        if value is self.sentinel:
            with self:
                if name in self._attribute_manager:
                    self._attributes_dict[name] = value = self._attribute_manager[name]
                else:
                    value = self.map.get_attribute(name, self.sentinel)
                    if value is self.sentinel:
                        if args:
                            value = args[0]
                        else:
                            return self._attribute_manager[name]
        return value

    def set_attribute(self, name: str, value: Any) -> None:
        """Sets a file attribute for the HDF5 file.

        Args:
            name: The name of the file attribute to set.
            value: The object to set the file attribute to.
        """
        name = self._parse_name(name)
        if self.exists:
            with self:
                if name in self._attribute_manager:
                    self._attribute_manager[name] = value
                else:
                    self._attribute_manager.create(name, value)
                self._attributes_dict[name] = self._attribute_manager[name]
        else:
            self.map.attributes[name] = value

    def del_attribute(self, name: str) -> None:
        """Deletes an attribute from the HDF5 file.

        Args:
            name: The name of the file attribute to delete.
        """
        name = self._parse_name(name)
        del self._attributes_dict[name]
        with self:
            del self._attribute_manager[name]

    # Attribute Modification
    def create(
        self,
        name: str,
        data: Any,
        shape: Iterable[int] | None = None,
        dtype: np.dtype | None = None,
    ) -> None:
        """Creates an attribute in the HDF5 file.

        Args:
            name: The name of the attribute.
            data: The value to fill the attribute.
            shape: The shape of the value/data.
            dtype: The type of the value/data.
        """
        name = self._parse_name(name)
        with self:
            self._attribute_manager.create(name, data, shape=shape, dtype=dtype)

    def modify(self, name: str, value: Any) -> None:
        """Modifies an attribute in the HDF5 file.

        Args:
            name: The name of the attribute.
            value: The value to fill the attribute.
        """
        name = self._parse_name(name)
        with self:
            self._attribute_manager.modify(name, value)

    # Mapping
    def get(self, key: str, *args: Any) -> Any:
        """Get an attribute within this object.

        Args:
            key: The key name to use to get the attribute.
            *args: An optional sentinel to return if a value is not present.

        Returns:
            The value of the attribute or the sentinel.
        """
        return self.get_attribute(key, *args)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def _keys(self) -> set[str]:
        """Get the names of all the attributes as a set.

        Returns:
            The name of all the attributes.
        """
        with self:
            return set(self._attribute_manager.keys())

    def keys(self) -> set[str]:
        """Get the names of all the attributes as a set, using caching.

        Returns:
            The name of all the attributes.
        """
        try:
            return self._keys.caching_call()
        except AttributeError:
            return self._keys()

    def values(self) -> ValuesView:
        """Returns the values of the attributes.

        Returns:
            All values of the attributes.
        """
        return self.get_attributes().values()

    def items(self) -> ItemsView:
        """Returns the attribute names and their values.

        Returns:
            The attribute names and their values.
        """
        return self.get_attributes().items()

    def update(self, **items) -> None:
        """Updates the file attributes based on the dictionary update scheme.

        Args:
            **items: The keyword arguments which are the attributes and their values.
        """
        with self:
            for name, value in items.items():
                name = self._parse_name(name)
                self._attribute_manager[name] = value

    def pop(self, key: str) -> Any:
        """Gets an attribute and deletes it from this object.

        Args:
            key: The attribute name to pop.

        Returns:
            The value of the attribute.
        """
        key = self._parse_name(key)
        with self:
            self._attribute_manager.pop(key)

    def clear(self) -> None:
        """Deletes all the attributes."""
        with self:
            self._attribute_manager.clear()

    # File
    def open(self, mode: str = "a", **kwargs: Any) -> "HDF5Attributes":
        """Opens the file to make this dataset usable.

        Args:
            mode: The file mode to open the file with.
            **kwargs: The additional keyword arguments to open the file with.

        Returns:
            This object.
        """
        self._file_was_open = self.file.is_open
        if not self._file_was_open:
            self.file.open(mode=mode, **kwargs)

        try:
            if not self._attribute_manager:
                self._attribute_manager = self.file._file[self._full_name].attrs
        except ValueError:
            self._attribute_manager = self.file._file[self._full_name].attrs

        return self

    def close(self) -> None:
        """Closes the file of this object."""
        if not self._file_was_open:
            self.file.close()
        elif self.file.mode not in {"w", "a"} and self.file._reopen and self.file.swmr_mode:
            self.file.close()
            self.file.open(**self.file.open_kwargs)

    def load(self) -> None:
        """Loads the attributes from the file."""
        self.get_attributes()

    def refresh(self) -> None:
        """Reloads the attributes from the file."""
        self.get_attributes()
