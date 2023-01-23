""" hdf5map.py
A map of the data within an HDF5 object/file. It outlines the different data objects and attributes within the 
object/file which can have different names in python than what is stored in the file.
"""
# Package Header #
from hdf5objects.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from collections.abc import Mapping, Iterator, ItemsView, KeysView, ValuesView
import copy
from typing import Any
import weakref

# Third-Party Packages #
from baseobjects import BaseObject, search_sentinel
from bidict import bidict
import numpy as np

# Local Packages #
from .hdf5caster import HDF5Caster


# Definitions #
# Classes #
class HDF5Map(BaseObject):
    """A nesting map representing data within an HDF5 object/file.
    
    This object represents one object within a HDF5 hierarchy, but other map objects can be nested within this one to 
    outline the full hierarchy.
    
    Class Attributes:
        sentinel: An object that helps with mapping searches.
        default_name: The default name of the map which represents its relation in the hdf5 file.
        default_parent: The default parent of the map.
        default_attributes_type: The default type for attribute objects in this map.
        default_attribute_names: The default name map of python name vs hdf5 name of the attribute.
        default_attributes: The default values of the attributes of the represented hdf5 object. 
        default_type: The default type of the hdf5 object this map represents.
        default_kwargs: The default keyword arguments for the object this map represents.
        default_map_names: The default name map of python name vs hdf5 name of the maps contained within this map.
        default_maps: The default nested maps within this map.
        default_attribute_component_types: The defualt attributes' components and their keyword arguments to add to when
            created.
        default_component_types: The components and their keyword arguments to add to the HDF5Object when created.
        caster: The caster object used to cast types from an HDF5 object to a python type.
        map_registry: A registry of map subclasses to be used when determining a map type to use.

    Attributes:
        _name: The name of this map.
        parents: The parents of this map as a list.
        attributes_type: The type for attribute objects in this map.
        attribute_names: The name map of python name vs hdf5 name of the attribute.
        attributes: The default values of the attributes of the represented hdf5 object. 
        type: The type of the hdf5 object this map represents.
        kwargs: The keyword arguments for the object this map represents.
        weak_object: A weak reference to the object that this map represents.
        map_names: The name map of python name vs hdf5 name of the maps contained within this map.
        maps: The nested maps.
        attribute_component_types: The attributes' components and their keyword arguments to add to when created.
        component_types: The components and their keyword arguments to add to the HDF5Object when created.
        
    Args:
        name: The name of this map.
        type_:  The type of the hdf5 object this map represents.
        attribute_names: The name map of python name vs hdf5 name of the attribute.
        attributes: The default values of the attributes of the represented hdf5 object. 
        map_names: The name map of python name vs hdf5 name of the maps contained within this map.
        maps: The nested maps.
        parent: The parent of this map.
        component_types: The components to add to the HDF5Object when created.
        component_kwargs: The keyword arguments for the components.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the object this map represents.
    """
    __slots__ = {
        "_name",
        "parents",
        "attributes_type",
        "attribute_names",
        "attributes",
        "type",
        "kwargs",
        "weak_object",
        "map_names",
        "maps",
    }
    sentinel: Any = search_sentinel
    default_name: str | None = None
    default_parent: str | None = None
    default_attributes_type: type | None = None
    default_attribute_names: Mapping[str, str] = {}
    default_attributes: Mapping[str, Any] = {}
    default_type: type | None = None
    default_kwargs: dict[str, Any] = {}
    default_map_names: Mapping[str, str] = {}
    default_maps: Mapping[str, "HDF5Map"] = {}
    default_attribute_component_types: dict[str, list[Any, dict[str, Any]]] = {}
    default_component_types: dict[str, list[Any, dict[str, Any]]] = {}
    caster: type = HDF5Caster
    map_registry: dict[str, type] = {}

    # Class Methods
    # Construction/Destruction
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """The init when creating a subclass.

        Args:
            **kwargs: The keyword arguments for creating a subclass.
        """
        super().__init_subclass__(**kwargs)
        # Add subclass (maps) to the registry.
        cls.map_registry[cls.__name__] = cls

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self, 
        name: str | None = None, 
        type_: type | None = None, 
        attribute_names: Mapping[str, str] | None = None, 
        attributes: Mapping[str, Any] | None = None,
        map_names: Mapping[str, str] | None = None, 
        maps: Mapping[str, "HDF5Map"] | None = None, 
        parent: str | None = None,
        component_types: dict[str, type] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._name: str | None = None
        self.parents: list[str] | None = None

        self.attributes_type: type = self.default_attributes_type
        self.attribute_names: bidict = bidict(self.default_attribute_names)
        self.attributes: Mapping[str, Any] = self.default_attributes

        self.type: type = self.default_type
        self.kwargs: dict[str, Any] = self.default_kwargs.copy()
        self.weak_object: weakref.ReferenceType | None = None

        self.map_names: bidict = bidict(self.default_map_names)
        self.maps: Mapping[str, "HDF5Map"] = copy.deepcopy(self.default_maps)

        self.attribute_component_types: dict[str, list[Any, dict[str, Any]]] = self.default_attribute_component_types.copy()
        self.component_types: dict[str, list[Any, dict[str, Any]]] = self.default_component_types.copy()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

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
                **kwargs,
            )

    @property
    def name(self) -> str | None:
        """The name of this map. The setter supports parsing a full hdf5 name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self.set_name(name=value)

    @property
    def parent(self) -> str:
        """Concatenates the parents into one str. The setter supports parsing a full hdf5 name."""
        if self.parents is None:
            return "/"
        else:
            return "".join(f"/{p}" for p in self.parents)

    @parent.setter
    def parent(self, value: str | None) -> None:
        if value is None:
            self.parents = None
        else:
            self.set_parent(parent=value)

    @property
    def full_name(self) -> str | None:
        """Returns the full hdf5 name of this map."""
        if self.parents is None:
            return f"/{'' if self._name is None else self._name}"
        else:
            return f"{''.join(f'/{p}' for p in self.parents)}/{'' if self._name is None else self._name}"

    @property
    def object(self) -> Any:
        """The object that this map represents"""
        return self.weak_object()

    @object.setter
    def object(self, value: Any) -> None:
        self.weak_object = weakref.ref(value)

    # Container Methods
    def __getitem__(self, key: str) -> "HDF5Map":
        """Gets a map within this object."""
        return self.get_item(key)

    def __setitem__(self, key: str, value) -> None:
        """Sets a map within this object."""
        self.set_item(key, value)

    def __delitem__(self, key: str) -> None:
        """Deletes a map within this object."""
        self.del_item(key)

    def __iter__(self) -> Iterator["HDF5Map"]:
        """Iterates over the maps within this object."""
        return iter(self.maps.values())

    def __contains__(self, item: str) -> bool:
        """Determines if a map is within this object."""
        return item in self.map_names or item in self.map_names.inverse

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self, 
        name: str | None = None, 
        type_: type | None = None, 
        attribute_names: Mapping[str, str] | None = None, 
        attributes: Mapping[str, Any] | None = None,
        map_names: Mapping[str, str] | None = None, 
        maps: Mapping[str, "HDF5Map"] | None = None,
        parent: str | None = None,
        component_types: dict[str, type] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object, setting attributes and sets nested maps' parents.
        
        Args:
            name: The name of this map.
            type_:  The type of the hdf5 object this map represents.
            attribute_names: The name map of python name vs hdf5 name of the attribute.
            attributes: The default values of the attributes of the represented hdf5 object. 
            map_names: The name map of python name vs hdf5 name of the maps contained within this map.
            maps: The nested maps.
            parent: The parent of this map.
            component_types: The components to add to the HDF5Object when created.
            component_kwargs: The keyword arguments for the components.
            **kwargs: The keyword arguments for the object this map represents.
        """
        if parent is not None:
            self.set_parent(parent=parent)

        if name is not None:
            self.set_name(name=name)

        if type_ is not None:
            self.type = type_

        if attribute_names is not None:
            self.attribute_names = attribute_names

        if attributes is not None:
            self.attributes = attributes

        if map_names is not None:
            self.map_names = map_names

        if maps is not None:
            self.maps = maps

        if component_types is not None:
            for name, type_ in component_types.items():
                self.component_types[name] = (type_, self.component_types.get(name, (None, {}))[1])

        if component_kwargs is not None:
            for name, c_kwargs in component_kwargs.items():
                type_, old_kwargs = self.component_types.get(name, (None, {}))
                self.component_types[name] = (type_, old_kwargs | c_kwargs)

        if kwargs is not None:
            self.kwargs.update(kwargs)

        self.set_children()

    def create_object(self, **kwargs: Any) -> Any:
        """Creates the object that this map is for.

        Args:
            **kwargs: The keyword arguments for the object.

        Returns:
            The HDF5Object that this map is for.
        """
        if "map_" not in kwargs:
            kwargs["map_"] = self

        object_ = self.type(**kwargs)
        self.weak_object = weakref.ref(object_)

        return object_

    def get_object(self, **kwargs: Any) -> Any:
        """Get the object that this map is for or constructs it if it has not been created.

        Args:
            **kwargs: The keyword arguments for the object.

        Returns:
            The HDF5Object that this map is for.
        """
        if self.weak_object is None or self.weak_object() is None:
            return self.create_object(**kwargs)
        else:
            return self.weak_object()

    # Parsers
    def _parse_attribute_name(self, name: str) -> str:
        """Returns the hdf5 name of an attribute.
        
        Args:
            name: Either the python name of the attribute or the hdf5 name.

        Returns:
            The hdf5 name of the attribute.
        """
        if name in self.attributes:
            return name
        else:
            new_name = self.attribute_names.inverse.get(name, self.sentinel)
            if new_name in self.attributes:
                return new_name
            else:
                return name

    def _parse_map_name(self, name: str) -> str:
        """Returns the hdf5 name of a map.

        Args:
            name: Either the python name of the map or the hdf5 name.

        Returns:
            The hdf5 name of the map.
        """
        if name in self.maps:
            return name
        else:
            new_name = self.map_names.inverse.get(name, self.sentinel)
            if new_name in self.maps:
                return new_name
            else:
                return name

    # Getters/Setters
    def get_attribute(self, key: str, *args: Any) -> Any:
        """Get an attribute in this map.

        Args:
            key: The key name to get the attribute with.
            *args: An optional sentinel to return if a value is not present.

        Returns:
            The value of the attribute or the sentinel.
        """
        attribute = self.attributes.get(key, self.sentinel)
        if attribute is self.sentinel:
            key = self.attribute_names.inverse.get(key, self.sentinel)
            attribute = self.attributes.get(key, self.sentinel)
            if attribute is self.sentinel:
                if args:
                    attribute = args[0]
                else:
                    return self.attributes.get(key)
        return attribute

    def set_attribute(self, name: str, value: Any, python_name: str = None) -> None:
        """Sets an attribute in this map.

        Args:
            name: The hdf5 name of the attribute to set.
            value: The value to set the attribute to.
            python_name: The python name of the attribute.
        """
        if python_name is None:
            new_name = self.attribute_names.get(name, self.sentinel)
            python_name = name
            if new_name is not self.sentinel:
                name = new_name

        self.attribute_names[python_name] = name

        if python_name in self.attributes:
            self.attributes[python_name] = value
        else:
            self.attributes[name] = value

    def get_item(self, key: str, *args) -> Any:
        """Get a nested map in this map.

        Args:
            key: The key name to get the map with.
            *args: An optional sentinel to return if a map is not present.

        Returns:
            The value of the map or the sentinel.
        """
        map_ = self.maps.get(key, self.sentinel)
        if map_ is self.sentinel:
            key = self.map_names.inverse.get(key, self.sentinel)
            map_ = self.maps.get(key, self.sentinel)
            if map_ is self.sentinel:
                if args:
                    map_ = args[0]
                else:
                    return self.maps.get(key)
        return map_

    def set_item(self, map_: "HDF5Map", name: str | None = None, python_name: str | None = None) -> None:
        """Sets a nested map in this map.

        Args:
            map_: The value to set the map to.
            name: The hdf5 name of the map to set.
            python_name: The python name of the map.
        """
        if name is None:
            name = map_.name

        if python_name is None:
            new_name = self.map_names.get(name, self.sentinel)
            python_name = name
            if new_name is not self.sentinel:
                name = new_name

        self.map_names[python_name] = name

        if python_name in self.maps:
            self.maps[python_name] = map_
        else:
            self.maps[name] = map_

    def del_item(self, key: str) -> None:
        """Deletes a nested map within this map.

        Args:
            key: The key name to delete the map with.
        """
        key = self._parse_map_name(key)
        del self.maps[key]
        del self.map_names.inverse[key]

    def set_parent(self, parent: str | None) -> None:
        """Sets the parent of this map to the str

        Args:
            parent: The str to parse and set as the parent of this map.
        """
        if parent is None:
            self.parents = None
        else:
            parent = parent.lstrip('/')
            parts = parent.split('/')
            self.parents = parts

    def set_name(self, name: str) -> None:
        """Sets the name of this map, can be a full hdf5 name.

        Args:
            name: The name of this map, can be a full hdf5 name.
        """
        name = name.lstrip('/')
        parts = name.split('/')
        self._name = parts.pop(-1)
        if parts:
            self.parents = parts

    def set_children(self) -> None:
        """Sets the nested maps parents to the correct hierarchy."""
        for name, child in self.maps.items():
            child.set_parent(parent=self.full_name)
            child_name = child.name
            if child_name is None:
                child_name = self.map_names.get(name, self.sentinel)
                if child_name is self.sentinel:
                    child_name = name
                child.name = child_name
            child.set_children()

    # Container
    def items(self) -> ItemsView:
        """Returns the names and nested maps of this map.

        Returns:
            The names and nested maps of this map.
        """
        return self.maps.items()

    def keys(self) -> KeysView:
        """Returns the names of the nested maps.

        Returns:
            The names of the nested maps.
        """
        return self.maps.keys()

    def values(self) -> ValuesView:
        """The nested maps of this map.

        Returns:
            The nested maps of this map.
        """
        return self.maps.values()

    def print_tree(self, indent: int = 0) -> None:
        """Prints the entire map.

        Args:
            indent: The number of space to print between each layer.
        """
        if self.attribute_names:
            print(f"{' ' * indent}  Attributes:")
            for name in self.attribute_names.values():
                print(f"{' ' * indent}      {name}")
        if self.maps:
            print(f"{' ' * indent}  Contents: ({''.join(f'{name}, ' for name in self.map_names.values())})")
            for name, map_ in self.maps.items():
                print(f"{' ' * indent}  +  {name}: {map_.full_name} {map_.type}")
                map_.print_tree(indent=indent+5)
