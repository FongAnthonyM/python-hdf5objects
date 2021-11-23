#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5objects.py
Description:
"""
# Package Header #
from .__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from abc import abstractmethod
from contextlib import contextmanager
from functools import singledispatchmethod
import pathlib
from typing import Any, Union
from warnings import warn

# Third-Party Packages #
from baseobjects import BaseObject, StaticWrapper, TimedDict, search_sentinel
from baseobjects.cachingtools import CachingInitMeta, CachingObject, timed_keyless_cache_method
from bidict import bidict
import h5py

# Local Packages #


# Definitions #
# Classes #
class HDF5Map(BaseObject):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    __slots__ = ["_name", "parents", "attributes_type", "attribute_names", "attributes", "type", "map_names", "maps"]
    sentinel = search_sentinel
    default_name = None
    default_parent = None
    default_attributes_type = None
    default_attribute_names = {}
    default_attributes = {}
    default_type = None
    default_map_names = {}
    default_maps = {}

    # Magic Methods
    # Construction/Destruction
    def __init__(self, name: str = None, type_=None, attribute_names: dict = None, attributes: dict = None,
                 map_names: dict = None, maps: dict = None, parent: str = None, init: bool = True):
        self._name = None
        self.parents = None

        self.attributes_type = self.default_attributes_type
        self.attribute_names = bidict(self.default_attribute_names)
        self.attributes = self.default_attributes

        self.type = self.default_type
        self.map_names = bidict(self.default_map_names)
        self.maps = self.default_maps.copy()

        if init:
            name = name if name is not None else self.default_name
            parent = parent if parent is not None else self.default_parent
            self.construct(name=name, type_=type_, attribute_names=attribute_names, attributes=attributes,
                           map_names=map_names, maps=maps, parent=parent)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self.set_name(name=value)

    @property
    def parent(self):
        if self.parents is None:
            return "/"
        else:
            return "".join(f"/{p}" for p in self.parents)

    @parent.setter
    def parent(self, value):
        if value is None:
            self.parents = None
        else:
            self.set_parent(parent=value)

    @property
    def full_name(self):
        if self._name is None:
            return None
        else:
            return self.parent + "/" + self._name

    # Container Methods
    def __getitem__(self, key: str):
        """Gets a map within this object."""
        return self.get_item(key)

    def __setitem__(self, key: str, value):
        """Sets a map within this object."""
        self.set_item(key, value)

    def __delitem__(self, key: str):
        """Deletes a map within this object."""
        self.del_item(key)

    def __iter__(self):
        """Iterates over the maps within this object."""
        return self.maps.__iter__()

    def __contains__(self, item: str):
        """Determines if a map is within this object."""
        return item in self.map_names or item in self.map_names.inverse

    # Instance Methods
    # Constructors/Destructors
    def construct(self, name: str = None, type_=None, attribute_names: dict = None, attributes: dict = None,
                  map_names: dict = None, maps: dict = None, parent: str = None):
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

        self.set_children()

    # Parsers
    def _parse_attribute_name(self, name: str):
        if name in self.attributes:
            return name
        else: 
            new_name = self.attribute_names.inverse.get(name, self.sentinel)
            if new_name in self.attributes:
                return new_name
            else:
                return name

    def _parse_map_name(self, name: str):
        if name in self.maps:
            return name
        else:
            new_name = self.map_names.inverse.get(name, self.sentinel)
            if new_name in self.maps:
                return new_name
            else:
                return name

    # Getters/Setters
    def get_attribute(self, key: str, *args):
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
    
    def set_attribute(self, name: str, value, python_name: str = None):
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
    
    def get_item(self, key: str, *args):
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

    def set_item(self, name: str, map_, python_name: str = None):
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

    def del_item(self, key: str):
        key = self._parse_map_name(key)
        del self.maps[key]
        del self.map_names.inverse[key]

    def set_parent(self, parent: str):
        if parent is None:
            self.parents = None
        else:
            parent = parent.lstrip('/')
            parts = parent.split('/')
            self.parents = parts

    def set_name(self, name: str):
        name = name.lstrip('/')
        parts = name.split('/')
        self._name = parts.pop(-1)
        if parts:
            self.parents = parts

    def set_children(self):
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
    def items(self):
        return self.maps.items()

    def keys(self):
        return self.maps.keys()

    def values(self):
        return self.maps.values()


class HDF5BaseObject(StaticWrapper, CachingObject, metaclass=CachingInitMeta):
    """An abstract wrapper which wraps object from an HDF5 file and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this dataset was accessed.
        _file: The file this dataset is from.
        _name (str): The name of the object from the HDF5 file.

    Args:
        file: The file which the dataset originates from.
        init (bool): Determines if this object should initialize.
    """
    sentinel = search_sentinel
    file_type = None
    default_map = None

    # Class Methods
    # Wrapped Attribute Callback Functions
    @classmethod
    def _get_attribute(cls, obj: Any, wrap_name: str, attr_name: str):
        """Gets an attribute from a wrapped HDF5 object.

        Args:
            obj (Any): The target object to get the wrapped object from.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to get from the wrapped object.

        Returns:
            (Any): The wrapped object.
        """
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
            return super()._get_attribute(obj, wrap_name, attr_name)

    @classmethod
    def _set_attribute(cls, obj: Any, wrap_name: str, attr_name: str, value: Any):
        """Sets an attribute in a wrapped HDF5 object.

        Args:
            obj (Any): The target object to set.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to set from the wrapped object.
            value (Any): The object to set the wrapped fileobjects attribute to.
        """
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
            super()._set_attribute(obj, wrap_name, attr_name, value)

    @classmethod
    def _del_attribute(cls, obj: Any, wrap_name: str, attr_name: str):
        """Deletes an attribute in a wrapped HDF5 object.

        Args:
            obj (Any): The target object to set.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to delete from the wrapped object.
        """
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
            super()._del_attribute(obj, wrap_name, attr_name)

    # Magic Methods
    # Constructors/Destructors
    def __init__(self, name: str = None, map_: HDF5Map = None, file=None, parent: str = None, init: bool = True):
        super().__init__()

        self._file_was_open = None
        self._file = None

        self._name_ = None
        self._parents = None

        self.map = self.default_map.copy()

        if init:
            self.construct(name=name, map_=map_, file=file, parent=parent)

    @property
    def _name(self):
        return self._name_

    @_name.setter
    def _name(self, value: str):
        self.set_name(name=value)

    @property
    def _parent(self):
        if self._parents is None:
            return "/"
        else:
            return "".join(f"/{p}" for p in self._parents)

    @_parent.setter
    def _parent(self, value):
        if value is None:
            self._parents = None
        else:
            self.set_parent(parent=value)

    @property
    def _full_name(self):
        if self._parents is None:
            return f"/{self._name_}"
        else:
            return "".join(f"/{p}" for p in self._parents) + self._name_
        
    @property
    def exists(self):
        return self.is_exist()

    # Container Methods
    def __getitem__(self, key: Any):
        """Ensures HDF5 object is open for getitem"""
        with self:
            return getattr(self, self._wrap_attributes[0])[key]

    def __setitem__(self, key: Any, value: Any):
        """Ensures HDF5 object is open for setitem"""
        with self:
            getattr(self, self._wrap_attributes[0])[key] = value

    def __delitem__(self, key: Any):
        """Ensures HDF5 object is open for delitem"""
        with self:
            del getattr(self, self._wrap_attributes[0])[key]

    # Context Managers
    def __enter__(self):
        """The enter context which opens the file to make this dataset usable"""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """The exit context which close the file."""
        self.close()

    # Type Conversion
    def __bool__(self):
        """When cast as a bool, this object True if valid and False if not.

        Returns:
            bool: If this object is open or not.
        """
        return bool(getattr(self, self._wrap_attributes[0]))

    # Instance Methods
    # Constructors/Destructors
    def construct(self, name: str = None, map_: HDF5Map = None, file=None, parent: str = None):
        """Constructs this object from the provided arguments.

        Args:
            file: The file which the dataset originates from.
        """
        if map_ is not None:
            self.map = map_

        if parent is not None:
            self.set_parent(parent=parent)
        elif map_ is not None:
            self._parents = self.map.parents

        if name is not None:
            self.set_name(name=name)
        elif map_ is not None:
            self._name_ = self.map.name

        if file is not None:
            self.set_file(file)
    
    def is_exist(self):
        with self._file.temp_open():
            try:
                self._file._file[self._full_name]
                return True
            except KeyError:
                return False

    # File
    def open(self, mode: str = 'a', **kwargs):
        """Opens the file to make this dataset usable.

        Args:
            mode (str, optional): The file mode to open the file with.
            **kwargs (dict, optional): The additional keyword arguments to open the file with.

        Returns:
            This object.
        """
        self._file_was_open = self._file.is_open
        if not getattr(self, self._wrap_attributes[0]):
            self._file.open(mode=mode, **kwargs)
            setattr(self, self._wrap_attributes[0], self._file._file[self._full_name])

        return self

    def close(self):
        """Closes the file of this dataset."""
        if not self._file_was_open:
            self._file.close()

    # Getters/Setters
    @singledispatchmethod
    def set_file(self, file):
        """Sets the file for this object to an HDF5File.

        Args:
            file: An object to set the file to.
        """
        if isinstance(file, HDF5File):
            self._file = file
        else:
            raise ValueError("file must be a path, File, or HDF5File")

    @set_file.register(str)
    @set_file.register(pathlib.Path)
    @set_file.register(h5py.File)
    def _(self, file: Union[str, pathlib.Path, h5py.File]):
        """Sets the file for this object to an HDF5File.

        Args:
            file: An object to set the file to.
        """
        self._file = self.file_type(file)

    def set_parent(self, parent: str):
        parent = parent.lstrip('/')
        parts = parent.split('/')
        self._parents = parts

    def set_name(self, name: str):
        if name is None:
            self._name_ = None
        else:
            name = name.lstrip('/')
            parts = name.split('/')
            name = parts.pop(-1)

            if name == "":
                self._name_ = "/"
            else:
                self._name_ = name

            if parts:
                self._parents = parts


class HDF5Attributes(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 attribute manager and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this attribute_manager was accessed.
        _file: The file this attribute_manager is from.
        _name (str): The name of this attribute_manager.
        _attribute_manager: The HDF5 attribute_manager to wrap.
        _attribute_names: The names of the attributes.
        is_updating (bool): Determines if this object should constantly open the file for updating attributes.

    Args:
        attributes: The HDF5 attribute_manager to build this attribute_manager around.
        name (str): The location of the attributes in the HDF5 file.
        file: The file which the attribute_manager originates from.
        update (bool): Determines if this object should constantly open the file for updating attributes.
        load (bool): Determines if the attributes should be loaded immediately.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.AttributeManager]
    _wrap_attributes = ["attribute_manager"]

    # Magic Methods #
    # Constructors/Destructors
    def __init__(self, attributes=None,  name: str = None, map_: HDF5Map = None, file=None,
                 load: bool = False, build: bool = False, parent: str = None, init: bool = True):
        super().__init__(file=file, init=False)
        self._attribute_manager = None
        self._attributes_dict = TimedDict()

        if init:
            self.construct(attributes=attributes, name=name, map_=map_, file=file,
                           load=load, build=build, parent=parent)

    # Container Methods
    def __getitem__(self, name):
        """Ensures HDF5 object is open for getitem"""
        return self.get_attribute(name)

    def __setitem__(self, name, value):
        """Ensures HDF5 object is open for setitem"""
        self.set_attribute(name, value)

    def __delitem__(self, name):
        """Ensures HDF5 object is open for delitem"""
        self.del_attribute(name)

    def __iter__(self):
        """Ensures HDF5 object is open for iter"""
        return self.get_attributes().__iter__()

    def __contains__(self, item):
        """Ensures HDF5 object is open for contains"""
        with self:
            return item in self._attribute_manager

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, attributes=None,  name: str = None, map_: HDF5Map = None, file=None,
                  load: bool = False, build: bool = False, parent: str = None):
        """Constructs this object from the provided arguments.

        Args:
            attributes: The HDF5 attribute_manager to build this attribute_manager around.
            name (str): The location of the attributes in the HDF5 file.
            file: The file which the attribute_manager originates from.
            update (bool): Determines if this object should constantly open the file for updating attributes.
            load (bool): Determines if the attributes should be loaded immediately.
        """
        super().construct(name=name, map_=map_, file=file, parent=parent)

        if attributes is not None:
            self.set_attribute_manager(attributes)

        if load and self.exists:
            self.load()

        if build:
            self.construct_attributes()

    def construct_attributes(self, map_=None, override: bool = False):
        if map_ is not None:
            self.map = map_

        with self:
            for name, value in self.map.attributes.items():
                name = self._parse_name(name)
                if name not in self._attribute_manager:
                    self._attribute_manager.create(name, value)
                elif override:
                    self._attribute_manager[name] = value

    # Parsers
    def _parse_name(self, name: str):
        new_name = self.map.attribute_names.get(name, self.sentinel)
        if new_name is not self.sentinel:
            name = new_name
        return name

    # Getters/Setters
    @singledispatchmethod
    def set_attribute_manager(self, attributes):
        """Sets the wrapped attribute_manager.

        Args:
            attributes: The attribute_manager this object will wrap.
        """
        raise ValueError("attribute_manager must be an AttributeManager or an HDF5Object")

    @set_attribute_manager.register
    def _(self, attributes: HDF5BaseObject):
        self._file = attributes._file
        self._name = attributes._name
        if isinstance(attributes, HDF5Attributes):
            self._attribute_manager = attributes._attribute_manager

    @set_attribute_manager.register
    def _(self, attributes: h5py.AttributeManager):
        if not attributes:
            raise ValueError("Attributes needs to be open")
        if self._file is None:
            self._file = self.file_type(attributes.file)
        self._name = attributes.name
        self._attribute_manager = attributes

    def get_attributes(self):
        """Gets all file attributes from the HDF5 file.

        Returns:
            dict: The file attributes.
        """
        a_names = set(self._attributes_dict.keys())
        missing = self.keys() - a_names
        if missing:
            with self._attributes_dict.pause_reset_timer():
                with self:
                    for name in missing:
                        self._attributes_dict[name] = self._attribute_manager[name]
        return dict(self._attributes_dict)

    def get_attribute(self, name: str, *args):
        """Gets an attribute from the HDF5 file.

        Args:
            name (str): The name of the file attribute to get.

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

    def set_attribute(self, name: str, value: Any):
        """Sets a file attribute for the HDF5 file.

        Args:
            name (str): The name of the file attribute to set.
            value (Any): The object to set the file attribute to.
        """
        name = self._parse_name(name)
        with self:
            if self.exists:
                if name in self._attribute_manager:
                    self._attribute_manager[name] = value
                else:
                    self._attribute_manager.create(name, value)
                self._attributes_dict[name] = self._attribute_manager[name]
            else:
                self.map.attributes[name] = value

    def del_attribute(self, name: str):
        """Deletes an attribute from the HDF5 file.

        Args:
            name (str): The name of the file attribute to delete.

        """
        name = self._parse_name(name)
        del self._attributes_dict[name]
        with self:
            del self._attribute_manager[name]

    # Attribute Modification
    def create(self, name: str, data, shape=None, dtype=None):
        name = self._parse_name(name)
        with self:
            self._attribute_manager.create(name, data, shape=shape, dtype=dtype)

    def modify(self, name: str, value: Any):
        name = self._parse_name(name)
        with self:
            self._attribute_manager.modify(name, value)

    # Mapping
    def get(self, key: str, *args):
        return self.get_attribute(key, *args)

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def _keys(self):
        with self:
            return set(self._attribute_manager.keys())

    def keys(self):
        try:
            return self._keys.caching_call()
        except AttributeError:
            return self._keys()

    def values(self):
        return self.get_attributes().values()

    def items(self):
        return self.get_attributes().items()

    def update(self, **items):
        """Updates the file attributes based on the dictionary update scheme.

        Args:
            **items: The keyword arguments which are the attributes an their values.
        """
        with self:
            for name, value in items.items():
                name = self._parse_name(name)
                self._attribute_manager[name] = value

    def pop(self, key: str):
        key = self._parse_name(key)
        with self:
            self._attribute_manager.pop(key)

    def clear(self):
        with self:
            self._attribute_manager.clear()

    # File
    def open(self, mode: str = 'a', **kwargs):
        """Opens the file to make this dataset usable.

        Args:
            mode (str, optional): The file mode to open the file with.
            **kwargs (dict, optional): The additional keyword arguments to open the file with.

        Returns:
            This object.
        """
        self._file_was_open = self._file.is_open
        try:
            if not self._attribute_manager:
                self._file.open(mode=mode, **kwargs)
                self._attribute_manager = self._file._file[self._name].attrs
        except ValueError:
            self._file.open(mode=mode, **kwargs)
            self._attribute_manager = self._file._file[self._name].attrs

    def load(self):
        self.get_attributes()

    def refresh(self):
        self.get_attributes()


class HDF5Group(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 _group_ and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this dataset was accessed.
        _file: The file this dataset is from.
        _name (str): The name of this dataset.
        _group: The HDF5 _group_ to wrap.
        attributes (:obj:`HDF5Attributes`): The attributes of this _group_.

    Args:
        group: The HDF5 dataset to build this dataset around.
        file: The file which the dataset originates from.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.Group]
    _wrap_attributes = ["group"]
    default_group = None
    default_dataset = None

    # Magic Methods #
    # Constructors/Destructors
    def __init__(self, group=None, name: str = None, map_: HDF5Map = None, file=None,
                 load: bool = False, build: bool = False, parent: str = None, init: bool = True):
        super().__init__(file=file, init=False)
        self._group = None
        self.attributes = None

        self.members = {}

        if init:
            self.construct(group=group, name=name, map_=map_, file=file, load=load, build=build, parent=parent)

    # Container Methods
    def __getitem__(self, key):
        """Ensures HDF5 object is open for getitem"""
        return self.get_item(key)

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, group=None, name: str = None, map_: HDF5Map = None, file=None,
                  load: bool = False, build: bool = False, parent: str = None):
        """Constructs this object from the provided arguments.

        Args:
            group: The HDF5 _group_ to build this _group_ around.
            file: The file which the _group_ originates from.
        """
        super().construct(name=name, map_=map_, file=file, parent=parent)

        if self.map.name is None:
            self.map.name = "/"

        if self.map.type is None:
            self.map.type = self.default_group

        if group is not None:
            self.set_group(group)

        self.construct_attributes(build=build)

        if load and self.exists:
            self.load(load=load)

        if build:
            self.require()
            self.construct_members(load=load, build=build)

    def construct_attributes(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        if map_ is None:
            map_ = self.map
        self.attributes = map_.attributes_type(name=self._full_name, map_=map_, file=self._file, load=load, build=build)

    def construct_members(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        if map_ is not None:
            self.map = map_

        for name, value in self.map.items():
            name = self._parse_name(name)
            if name not in self.members:
                self.members[name] = value.type(map_=value, load=load, build=build, file=self._file)

    # Parsers
    def _parse_name(self, name: str):
        new_name = self.map.map_names.get(name, self.sentinel)
        if new_name is not self.sentinel:
            name = new_name
        return name

    # File
    def load(self, load: bool = False, build: bool = False):
        self.attributes.load()
        self.get_members(load=load, build=build)

    def refresh(self):
        self.attributes.refresh()

    # Getters/Setters
    @singledispatchmethod
    def set_group(self, group):
        """Sets the wrapped _group_.

        Args:
            group: The _group_ this object will wrap.
        """
        if isinstance(group, HDF5Group):
            if self._file is None:
                self._file = group._file
            self._name = group._name
            self._group = group._group
        else:
            raise ValueError("_group_ must be a Dataset or HDF5Dataset")

    @set_group.register
    def _(self, group: h5py.Group):
        """Sets the wrapped _group_.

        Args:
            group (:obj:`Group`): The _group_ this object will wrap.
        """
        if not group:
            raise ValueError("Group needs to be open")
        if self._file is None:
            self._file = self.file_type(group.file)
        self._name = group.name
        self._group = group

    def get_member(self, name: str, load: bool = False, build: bool = False, **kwargs):
        name = self._parse_name(name)
        with self:
            item = self._group[name]
            map_ = self.map.get_item(name, self.sentinel)
            if map_ is not self.sentinel:
                if isinstance(map_.type, HDF5Group):
                    kwargs["group"] = item
                else:
                    kwargs["dataset"] = item
                self.members[name] = map_.type(map_=map_, file=self._file, load=load, build=build, **kwargs)
            else:
                if isinstance(item, h5py.Dataset):
                    self.members[name] = self.default_dataset(dataset=item, file=self._file,
                                                              load=load, build=build, **kwargs)
                elif isinstance(item, h5py.Group):
                    self.members[name] = self.default_group(group=item, file=self._file,
                                                            load=load, build=build, **kwargs)
        return self.members[name]

    def get_members(self,  load: bool = False, build: bool = False):
        with self:
            for name, value in self._group.items():
                map_ = self.map.get_item(name, self.sentinel)
                if map_ is not self.sentinel:
                    if isinstance(map_.type, HDF5Group):
                        kwargs = {"group": value}
                    else:
                        kwargs = {"dataset": value}
                    self.members[name] = map_.type(map_=map_, file=self._file, load=load, build=build, **kwargs)
                else:
                    if isinstance(value, h5py.Dataset):
                        self.members[name] = self.default_dataset(dataset=value, file=self._file,
                                                                  load=load, build=build)
                    elif isinstance(value, h5py.Group):
                        self.members[name] = self.default_group(group=value, file=self._file, load=load, build=build)
        return self.members

    def get_item(self, key: str):
        key = self._parse_name(key)
        item = self.members.get(key, self.sentinel)
        if item is not self.sentinel:
            return item
        else:
            return self.get_member(key)

    # Group Modification
    def create(self, name: str = None, track_order=None):
        if name is not None:
            self._name = name

        with self._file.temp_open():
            self._group = self._file._file.create_group(name=self._full_name, track_order=track_order)
            self.attributes.construct_attributes()

        return self

    def require(self, name: str = None):
        if name is not None:
            self._name = name

        with self._file.temp_open():
            if not self.exists:
                self._group = self._file._file.create_group(name=self._full_name)
                self.attributes.construct_attributes()

        return self


class HDF5Dataset(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 dataset and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this dataset was accessed.
        _file: The file this dataset is from.
        _name (str): The name of this dataset.
        _dataset: The HDF5 dataset to wrap.
        attributes (:obj:`HDF5Attributes`): The attributes of this dataset.

    Args:
        dataset: The HDF5 dataset to build this dataset around.
        file: The file which the dataset originates from.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.Dataset]
    _wrap_attributes = ["dataset"]

    # Magic Methods
    # Constructors/Destructors
    def __init__(self, dataset=None,  name: str = None, map_: HDF5Map = None, file=None,
                 load: bool = False, build: bool = False, parent: str = None, init: bool = True, **kwargs):
        super().__init__(file=file, init=False)
        self._dataset = None
        self._scale_name = None
        self.attributes = None

        if init:
            self.construct(dataset=dataset, name=name, map_=map_, file=file,
                           load=load, build=build, parent=parent, **kwargs)

    @property
    def scale_name(self):
        return self._scale_name

    @scale_name.setter
    def scale_name(self, value: str):
        self.make_scale(value)

    @property
    def shape(self):
        try:
            return self.get_shape.caching_call()
        except AttributeError:
            return self.get_shape()

    @property
    def all_data(self):
        try:
            return self.get_all_data.caching_call()
        except AttributeError:
            return self.get_all_data()

    def __array__(self, dtype=None):
        with self:
            return self._dataset.__array__(dtype=dtype)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, dataset=None,  name: str = None, map_: HDF5Map = None, file=None,
                  load: bool = False, build: bool = False, parent: str = None, **kwargs):
        """Constructs this object from the provided arguments.

        Args:
            dataset: The HDF5 dataset to build this dataset around.
            name: The name of the dataset.
            file: The file which the dataset originates from.
            create: Determines if the dataset will be created if it does not exist.
            kwargs: The key word arguments to construct the base HDF5 dataset.
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

    def construct_attributes(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        if map_ is None:
            map_ = self.map
        self.attributes = map_.attributes_type(name=self._full_name, map_=map_, file=self._file, load=load, build=build)

    def construct_dataset(self, **kwargs):
        self.require(name=self._full_name, **kwargs)

    # File
    def load(self):
        self.attributes.load()

    def refresh(self):
        with self:
            self._dataset.refresh()
        self.attributes.refresh()
        self.get_shape.clear_cahce()
        self.get_all_data.clear_cache()

    # Getters/Setters
    @singledispatchmethod
    def set_dataset(self, dataset):
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
            raise ValueError("dataset must be a Dataset or HDF5Dataset")

    @set_dataset.register
    def _(self, dataset: h5py.Dataset):
        """Sets the wrapped dataset.

        Args:
            dataset (:obj:`Dataset`): The dataset this object will wrap.
        """
        if not dataset:
            raise ValueError("Dataset needs to be open")
        if self._file is None:
            self._file = self.file_type(dataset.file)
        self.set_name(dataset.name)
        self._dataset = dataset

    @set_dataset.register
    def _(self, dataset: str):
        """Sets the wrapped dataset base on a str.

        Args:
            dataset (str): The name of the dataset.
        """
        self.set_name(dataset)

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_shape(self):
        with self:
            return self._dataset.shape

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_all_data(self):
        with self:
            return self._dataset[...]

    # Data Modification
    def create(self, name: str = None, **kwargs):
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

    def require(self, name: str = None, **kwargs):
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

    def replace_data(self, data):
        with self:
            # Assign Data
            self._dataset.resize(data.shape)  # Reshape for new data
            self._dataset[...] = data

    def set_data(self, data, **kwargs):
        if self.exists:
            self.replace_data(data=data)
        else:
            self.require(data=data, **kwargs)

    def append(self, data, axis=0):
        """Append data to the dataset along a specified axis.

        Args:
            data: The data to append.
            axis (int): The axis to append the data along.
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
            self._dataset.resize(new_shape)  # Reshape for new data
            self._dataset[slicing] = data    # Assign data to the new location

    def make_scale(self, name: str = None):
        if name is not None:
            self._scale_name = name
        if self._scale_name is not None and self.exists:
            with self:
                self._dataset.make_scale(self._scale_name)

    def attach_axis(self, dataset, axis=0):
        if isinstance(dataset, HDF5Dataset):
            dataset = dataset._dataset

        with self:
            self._dataset.dims[axis].attach_scale(dataset)

    def detach_axis(self, dataset, axis=0):
        if isinstance(dataset, HDF5Dataset):
            dataset = dataset._dataset

        with self:
            self._dataset.dims[axis].detach_scale(dataset)


# Assign Cyclic Definitions
HDF5Map.default_attributes_type = HDF5Attributes

HDF5BaseObject.default_map = HDF5Map()

HDF5Group.default_group = HDF5Group
HDF5Group.default_dataset = HDF5Dataset


class HDF5File(HDF5BaseObject):
    """A class which wraps a HDF5 File and gives more functionality.

    Class Attributes:
        attribute_type: The class to cast the HDF5 attribute manager as.
        group_type: The class to cast the HDF5 _group_ as.
        dataset_type: The class to cast the HDF5 dataset as.

    Attributes:
        _file_attrs (set): The names of the attributes in the HDF5 file.
        _path (obj:`Path`): The path to were the HDF5 file exists.
        is_updating (bool): Determines if this object should constantly open the file for updating attributes.

        c_kwargs: The keyword arguments for the data compression.
        default_dataset_kwargs: The default keyword arguments for datasets when they are created.
        default_file_attributes (dict): The default file attributes the HDF5 file should have.
        default_datasets (dict): The default datasets the HDF5 file should have.

        hf_fobj: The HDF5 File object this object wraps.

    Args:
        obj: An object to build this object from. It can be the path to the file or a File fileobjects.
        update (bool): Determines if this object should constantly open the file for updating attributes.
        open_ (bool): Determines if this object will remain open after construction.
        init (bool): Determines if this object should initialize.
        **kwargs: The keyword arguments for the open method.
    """
    # Todo: Rethink about how Errors and Warnings are handled in this object.
    _wrapped_types = [HDF5Group, h5py.File]
    _wrap_attributes = ["group", "_file"]
    attribute_type = HDF5Attributes
    group_type = HDF5Group
    dataset_type = HDF5Dataset

    # Class Methods
    # Wrapped Attribute Callback Functions
    @classmethod
    def _get_attribute(cls, obj: Any, wrap_name: str, attr_name: str):
        """Gets an attribute from a wrapped HDF5 object.

        Args:
            obj (Any): The target object to get the wrapped object from.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to get from the wrapped object.

        Returns:
            (Any): The wrapped object.
        """
        with obj.temp_open():  # Ensures the hdf5 dataset is open when accessing attributes
            return super()._get_attribute(obj, wrap_name, attr_name)

    @classmethod
    def _set_attribute(cls, obj: Any, wrap_name: str, attr_name: str, value: Any):
        """Sets an attribute in a wrapped HDF5 object.

        Args:
            obj (Any): The target object to set.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to set from the wrapped object.
            value (Any): The object to set the wrapped fileobjects attribute to.
        """
        with obj.temp_open():  # Ensures the hdf5 dataset is open when accessing attributes
            super()._set_attribute(obj, wrap_name, attr_name, value)

    @classmethod
    def _del_attribute(cls, obj: Any, wrap_name: str, attr_name: str):
        """Deletes an attribute in a wrapped HDF5 object.

        Args:
            obj (Any): The target object to set.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to delete from the wrapped object.
        """
        with obj.temp_open():  # Ensures the hdf5 dataset is open when accessing attributes
            super()._del_attribute(obj, wrap_name, attr_name)

    # Validation #
    @classmethod
    def is_openable(cls, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if path.is_file():
            try:
                h5py.File(path)
                return True
            except OSError:
                return False
        else:
            return False

    # Magic Methods
    # Construction/Destruction
    def __init__(self, file=None, open_: bool = False, map_: HDF5Map = None, load: bool = False,
                 create: bool = False, build: bool = False, init: bool = True, **kwargs):
        super().__init__(init=False)

        self._path = None
        self._name_ = "/"

        self._group = None

        if init:
            self.construct(file=file, open_=open_, map_=map_, load=load, create=create, build=build, **kwargs)

    @property
    def path(self):
        """:obj:`Path`: The path to the file.

        The setter casts fileobjects that are not Path to path before setting
        """
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    @property
    def is_open(self):
        """bool: Determines if the hdf5 file is open."""
        try:
            return bool(self._file)
        except:
            return False

    @property
    def attributes(self):
        return self._group.attributes

    def __del__(self):
        """Closes the file when this object is deleted."""
        self.close()

    # Pickling
    def __getstate__(self):
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            dict: A dictionary of this object's attributes.
        """
        state = self.__dict__.copy()
        state["open_state"] = self.is_open
        del state["_file"]
        return state

    def __setstate__(self, state):
        """Builds this object based on a dictionary of corresponding attributes.

        Args:
            state (dict): The attributes to build this object from.
        """
        state["_file"] = h5py.File(state["path"].as_posix(), "r+")
        if not state.pop("open_state"):
            state["_file"].close()
        self.__dict__.update(state)

    # Container Methods
    def __getitem__(self, item):
        """Gets a container from the HDF5 file based on the arguments.

        Args:
            item (str): The name of the container to get.

        Returns:
            The container requested.
        """
        return self._group[item]

    # Context Managers
    def __enter__(self):
        """The context enter which opens the HDF5 file.

        Returns:
            This object.
        """
        if self._file is None:
            self.construct(open_=True)
        else:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """The context exit which closes the file."""
        return self.close()

    # Type Conversion
    def __bool__(self):
        """When cast as a bool, this object True if open and False if closed.

        Returns:
            bool: If this object is open or not.
        """
        return self.is_open

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, file=None, open_: bool = True, map_: HDF5Map = None, load: bool = False,
                  create: bool = False, build: bool = False, **kwargs):
        """Constructs this object.

        Args:
            obj: An object to build this object from. It can be the path to the file or a File object.
            update (bool): Determines if this object should constantly open the file for updating attributes.
            open_ (bool): Determines if this object will remain open after construction.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if map_ is not None:
            self.map = map_

        if self.map.name is None:
            self.map.name = "/"

        if file is not None:
            self._set_path(file)

        if not self.path.is_file():
            if create:
                self.require_file(open_, **kwargs)
            elif load:
                raise ValueError("A file is required to load this file.")
            elif build:
                raise ValueError("A file is required to build this file.")
        elif open_ or load or build:
            self.open(**kwargs)

        self.construct_group(load=load, build=build)

        if build:
            self.construct_file_attributes()

        if not open_:
            self.close()

        return self

    def construct_file_attributes(self):
        pass

    def construct_group(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        if map_ is not None:
            self.map = map_
        self._group = self.group_type(name=self._name_, map_=self.map, file=self, load=load, build=build)

    # Getters/Setters
    @singledispatchmethod
    def _set_path(self, file):
        if isinstance(file, HDF5File):
            self.path = file.path

    @_set_path.register(str)
    @_set_path.register(pathlib.Path)
    def _(self, file: Union[str, pathlib.Path]):
        """Constructs the path attribute of this object.

        Args:
            file: The path to the file to build this object around.
        """
        self.path = file

    @_set_path.register
    def _(self, file: h5py.File):
        """Constructs the path attribute of this object.

        Args:
            file (obj:`File`): A HDF5 file to build this object around.
        """
        if file:
            self._file = file
            self._path = pathlib.Path(file.filename)
        else:
            raise ValueError("The supplied HDF5 File must be open.")

    # File Creation/Construction
    def create_file(self, name=None, open_=True, map_: HDF5Map = None, build: bool = False, **kwargs):
        """Creates the HDF5 file.

        Args:
            attr (dict, optional): File attributes to set when the file is created.
            data (dict, optional): Datasets to assign when the file is created.
            construct (bool, optional): Determines if the file will be constructed.
            open_ (bool, optional): Determines if this object will remain open after construction.

        Returns:
            This object.
        """
        if name is not None:
            self.path = name

        if map_ is not None:
            self.map = map_

        self.open(**kwargs)
        if build:
            self._group.construct_members(build=True)
        elif not open_:
            self.close()

        return self

    def require_file(self, open_=False, load: bool = False,  map_: HDF5Map = None, build: bool = False,  **kwargs):
        if self.path.is_file():
            self.open(**kwargs)
            if load:
                self._group.load(load=True, build=build)
            if not open_:
                self.close()
        else:
            self.create_file(open_=open_, map_=map_, build=build, **kwargs)

        return self

    # def copy_file(self, path):  # Todo: Implement this.
    #     pass

    # File
    def open(self, mode='a', exc=False, **kwargs):
        """Opens the HDF5 file.

        Args:
            mode (str): The mode which this file should be opened in.
            exc (bool): Determines if an error should be excepted as warning or not.
            kwargs: The keyword arguments for opening the HDF5 file.

        Returns:
            This object.
        """
        if not self.is_open:
            try:
                self._file = h5py.File(self.path.as_posix(), mode=mode, **kwargs)
                return self
            except Exception as error:
                if exc:
                    warn("Could not open" + self.path.as_posix() + "due to error: " + str(error), stacklevel=2)
                    self._file = None
                    return self
                else:
                    raise error

    @contextmanager
    def temp_open(self, **kwargs):
        """Temporarily opens the file if it is not already open.

        Args:
            **kwargs: The keyword arguments for opening the HDF5 file.

        Returns:
            This object.
        """
        was_open = self.is_open
        self.open(**kwargs)
        try:
            yield self
        finally:
            if not was_open:
                self.close()

    def close(self):
        """Closes the HDF5 file.

        Returns:
            bool: If the file was successfully closed.
        """
        if self.is_open:
            self._file.flush()
            self._file.close()
        return not self.is_open


# Assign Cyclic Definitions
HDF5BaseObject.file_type = HDF5File
