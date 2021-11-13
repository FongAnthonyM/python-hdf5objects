#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5objects.py
Description:
"""
__author__ = "Anthony Fong"
__copyright__ = "Copyright 2021, Anthony Fong"
__credits__ = ["Anthony Fong"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Anthony Fong"
__email__ = ""
__status__ = "Prototype"

# Standard Libraries #
from abc import abstractmethod
from contextlib import contextmanager
import pathlib
from typing import Any
from warnings import warn
from functools import singledispatchmethod

# Third-Party Packages #
from baseobjects import BaseObject, StaticWrapper, search_sentinel
from baseobjects.cachingtools import MetaCachingInit, CachingObject, timed_keyless_cache_method
from bidict import bidict
import h5py
from multipledispatch import dispatch

# Local Packages #


# Definitions #
# Classes #
class HDF5Map(BaseObject):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    __slots__ = ["name", "parent", "attributes_type", "attribute_names", "type", "attributes", "map_names", "maps"]
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
        return self.parent + "/" + self.name

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
        new_name = self.attribute_names.get(name, self.sentinel)
        if new_name is not self.sentinel:
            name = new_name
        return name

    def _parse_map_name(self, name: str):
        new_name = self.map_names.get(name, self.sentinel)
        if new_name is not self.sentinel:
            name = new_name
        return name

    # Getters/Setters
    def get_item(self, key: str):
        key = self._parse_map_name(key)
        return self.maps[key]

    def set_item(self, name: str, map_, python_name: str = None):
        self.maps[name] = map_
        if python_name is None:
            self.map_names[name] = name
        else:
            self.map_names[python_name] = name

    def del_item(self, key: str):
        key = self._parse_map_name(key)
        del self.maps[key]
        del self.map_names.inverse[key]

    def set_parent(self, parent: str):
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
        for child in self.maps.values():
            child.set_parent(parent=self.full_name)
            child.set_children()

    # Container
    def items(self):
        return self.maps.items()

    def keys(self):
        return self.maps.keys()

    def values(self):
        return self.maps.values()


class HDF5BaseObject(StaticWrapper, CachingObject, metaclass=MetaCachingInit):
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
    default_map = HDF5Map()

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

    # Container Methods
    def __getitem__(self, key: str):
        """Ensures HDF5 object is open for getitem"""
        with self:
            return getattr(self, self._wrap_attributes[0])[key]

    def __setitem__(self, key: str, value: Any):
        """Ensures HDF5 object is open for setitem"""
        with self:
            getattr(self, self._wrap_attributes[0])[key] = value

    def __delitem__(self, key: str):
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

    # Getters/Setters
    @dispatch(object)
    def set_file(self, file):
        """Sets the file for this object to an HDF5File.

        Args:
            file: An object to set the file to.
        """
        if isinstance(file, HDF5File):
            self._file = file
        else:
            raise ValueError("file must be a path, File, or HDF5File")

    @dispatch((str, pathlib.Path, h5py.File))
    def set_file(self, file):
        """Sets the file for this object to an HDF5File.

        Args:
            file: An object to set the file to.
        """
        self._file = HDF5File(file)

    def set_parent(self, parent: str):
        parent = parent.lstrip('/')
        parts = parent.split('/')
        self._parents = parts

    def set_name(self, name: str):
        name = name.lstrip('/')
        parts = name.split('/')
        self._name = parts.pop(-1)
        if parts:
            self._parents = parts

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
            setattr(self, self._wrap_attributes[0], self._file.h5_fobj[self._full_name])

        return self

    def close(self):
        """Closes the file of this dataset."""
        if not self._file_was_open:
            self._file.close()


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
    _wrap_attributes = ["_attribute_manager"]

    # Magic Methods #
    # Constructors/Destructors
    def __init__(self, attributes=None,  name: str = None, map_: HDF5Map = None, file=None,
                 load: bool = False, build: bool = False, parent: str = None, init: bool = True):
        super().__init__(file=file, init=False)
        self._attribute_manager = None

        if init:
            self.construct(attributes=attributes, name=name, map_=map_, file=file,
                           load=load, build=build, parent=parent)

    @property
    def attributes_dict(self):
        try:
            return self.get_attributes.caching_call()
        except AttributeError:
            return self.get_attributes()

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
        return self.attributes_dict.__iter__()

    def __contains__(self, item):
        """Ensures HDF5 object is open for contains"""
        return self.attributes_dict.__containes__(item)

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
        if file is None and attributes is None:
            raise ValueError("A file or an attribute manager must be given")

        super().construct(name=name, map_=map_, file=file, parent=parent)

        if attributes is not None:
            self.set_attribute_manager(attributes)

        if load:
            self.load()

        if build:
            self.construct_attributes()

    def construct_attributes(self, map_=None):
        if map_ is not None:
            self.map = map_

        with self:
            for name, value in map_.attributes.items():
                if name not in self._attribute_manager:
                    self._attribute_manager.create(name, value)

        self.get_attributes.clear_cache()

    # Parsers
    def _parse_name(self, name: str):
        new_name = self.map.attributes.get(name, self.sentinel)
        if new_name is not self.sentinel:
            name = new_name
        return name

    # Getters/Setters
    @dispatch(object)
    def set_attribute_manager(self, attributes):
        """Sets the wrapped attribute_manager.

        Args:
            attributes: The attribute_manager this object will wrap.
        """
        if isinstance(attributes, HDF5BaseObject):
            self._file = attributes._file
            self._name = attributes._name
            if isinstance(attributes, HDF5Attributes):
                self._attribute_manager = attributes._attribute_manager
        else:
            raise ValueError("attribute_manager must be a Dataset or HDF5Dataset")

    @dispatch(h5py.AttributeManager)
    def set_attribute_manager(self, attributes):
        if not attributes:
            raise ValueError("Attributes needs to be open")
        if self._file is None:
            self._file = HDF5File(attributes.file)
        self._name = attributes.name
        self._attribute_manager = attributes
    
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_attributes(self):
        """Gets all file attributes from the HDF5 file.

        Returns:
            dict: The file attributes.
        """
        with self:
            attributes = dict(self._attribute_manager.items())

        return attributes

    def get_attribute(self, name: str):
        """Gets an attribute from the HDF5 file.

        Args:
            name (str): The name of the file attribute to get.

        Returns:
            The attribute requested.
        """
        name = self._parse_name(name)
        return self.attributes_dict[name]

    def set_attribute(self, name: str, value: Any):
        """Sets a file attribute for the HDF5 file.

        Args:
            name (str): The name of the file attribute to set.
            value (Any): The object to set the file attribute to.
        """
        name = self._parse_name(name)
        with self:
            self._attribute_manager[name] = value
        self.get_attributes.clear_cache()

    def del_attribute(self, name: str):
        """Deletes an attribute from the HDF5 file.

        Args:
            name (str): The name of the file attribute to delete.

        """
        name = self._parse_name(name)
        with self:
            del self._attribute_manager[name]
        self.get_attributes.clear_cache()

    # Attribute Modification
    def create(self, name: str, data, shape=None, dtype=None):
        name = self._parse_name(name)
        with self:
            self._attribute_manager.create(name, data, shape=shape, dtype=dtype)
        self.get_attributes.clear_cache()

    def modify(self, name: str, value: Any):
        name = self._parse_name(name)
        with self:
            self._attribute_manager(name, value)
        self.get_attributes.clear_cache()

    # Mapping
    def get(self, key: str, *args):
        key = self._parse_name(key)
        self.attributes_dict(key, *args)

    def keys(self):
        return self.attributes_dict.keys()

    def values(self):
        return self.attributes_dict.values()

    def items(self):
        return self.attributes_dict.values()

    def update(self, **items):
        """Updates the file attributes based on the dictionary update scheme.

        Args:
            **items: The keyword arguments which are the attributes an their values.
        """
        with self:
            for name, value in items.items():
                name = self._parse_name(name)
                self._attribute_manager[name] = value
        self.get_attributes.clear_cache()

    def pop(self, key: str):
        key = self._parse_name(key)
        with self:
            self._attribute_manager.pop(key)
        self.get_attributes.clear_cache()

    def clear(self):
        with self:
            self._attribute_manager.clear()
        self.get_attributes.clear_cache()

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
                self._attribute_manager = self._file.h5_fobj[self._name].attrs
        except ValueError:
            self._file.open(mode=mode, **kwargs)
            self._attribute_manager = self._file.h5_fobj[self._name].attrs

    def load(self):
        self.get_attributes()


class HDF5Group(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 group and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this dataset was accessed.
        _file: The file this dataset is from.
        _name (str): The name of this dataset.
        _group: The HDF5 group to wrap.
        attributes (:obj:`HDF5Attributes`): The attributes of this group.

    Args:
        group: The HDF5 dataset to build this dataset around.
        file: The file which the dataset originates from.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.Group]
    _wrap_attributes = ["_group"]
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
            group: The HDF5 group to build this group around.
            file: The file which the group originates from.
        """
        if file is None and group is None:
            raise ValueError("A file or group must be given")

        super().construct(name=name, map_=map_, file=file, parent=parent)

        if group is not None:
            self.set_group(group)

        self.construct_attributes(build=build)

        if load:
            self.load(load=load, build=build)

        if build:
            self.construct_members(load=load, build=build)

    def construct_attributes(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        if map_ is None:
            map_ = self.map
        self.attributes = map_.attributes_type(name=self._name, map_=map_, file=self._file, load=load, build=build)

    def construct_members(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        if map_ is not None:
            self.map = map_

        for name, value in self.map.items():
            if name not in self.members:
                self.members[name] = value.type(map_=value, load=load, build=build)

    # Parsers
    def _parse_name(self, name: str):
        new_name = self.map.map_names.get(name, self.sentinel)
        if new_name is not self.sentinel:
            name = new_name
        return name

    # Getters/Setters
    @dispatch(object)
    def set_group(self, group):
        """Sets the wrapped group.

        Args:
            group: The group this object will wrap.
        """
        if isinstance(group, HDF5Group):
            if self._file is None:
                self._file = group._file
            self._name = group._name
            self._group = group._group
        else:
            raise ValueError("group must be a Dataset or HDF5Dataset")

    @dispatch(h5py.Group)
    def set_group(self, group):
        """Sets the wrapped group.

        Args:
            group (:obj:`Group`): The group this object will wrap.
        """
        if not group:
            raise ValueError("Group needs to be open")
        if self._file is None:
            self._file = HDF5File(group.file)
        self._name = group.name
        self._group = group

    def get_member(self, name: str, load: bool = False, build: bool = False):
        with self:
            item = self._group[name]
            map_ = self.map.maps.get(name, self.sentinel)
            if map_ is not self.sentinel:
                self.members[name] = map_.type(item, map_=map_, file=self._file, load=load, build=build)
            else:
                if isinstance(item, h5py.Dataset):
                    self.members[name] = self.default_dataset(item, file=self._file, load=load, build=build)
                elif isinstance(item, h5py.Group):
                    self.members[name] = self.default_group(item, file=self._file, load=load, build=build)
        return self.members[name]

    def get_members(self,  load: bool = False, build: bool = False):
        with self:
            for name, value in self._group.items():
                map_ = self.map.maps.get(name, self.sentinel)
                if map_ is not self.sentinel:
                    self.members[name] = map_.type(value, map_=map_, load=load, build=build)
                else:
                    if isinstance(value, h5py.Dataset):
                        self.members[name] = self.default_dataset(dataset=value, load=load, build=build)
                    elif isinstance(value, h5py.Group):
                        self.members[name] = self.default_group(group=value, load=load, build=build)
        return self.members

    def get_item(self, key: str):
        key = self._parse_name(key)
        item = self.members.get(key, self.sentinel)
        if item is not self.sentinel:
            return item
        else:
            return self.get_member(key)

    # File
    def load(self, load: bool = False, build: bool = False):
        self.get_members(load=load, build=build)


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
    _wrap_attributes = ["_dataset"]
    default_map = HDF5Map()

    # Magic Methods
    # Constructors/Destructors
    def __init__(self, dataset=None,  name: str = None, map_: HDF5Map = None, file=None,
                 load: bool = False, build: bool = False, parent: str = None, init: bool = True, **kwargs):
        super().__init__(file=file, init=False)
        self._dataset = None
        self.attributes = None

        if init:
            self.construct(dataset=dataset, name=name, map_=map_, file=file,
                           load=load, build=build, parent=parent, **kwargs)

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

        if dataset is not None:
            self.set_dataset(dataset)

        self.construct_attributes(load=load, build=build)

        if load:
            self.load()

        if build:
            self.construct_dataset(**kwargs)

    def construct_attributes(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        if map_ is None:
            map_ = self.map
        self.attributes = map_.attributes_type(name=self._name, map_=map_, file=self._file, load=load, build=build)

    def construct_dataset(self, **kwargs):
        self.require(name=self._name, **kwargs)

    # Getters/Setters
    @dispatch(object)
    def set_dataset(self, dataset):
        """Sets the wrapped dataset.

        Args:
            dataset: The dataset this object will wrap.
        """
        if isinstance(dataset, HDF5Dataset):
            if self._file is None:
                self._file = dataset._file
            self._name = dataset._name
            self._dataset = dataset._dataset
        else:
            raise ValueError("dataset must be a Dataset or HDF5Dataset")

    @dispatch(h5py.Dataset)
    def set_dataset(self, dataset):
        """Sets the wrapped dataset.

        Args:
            dataset (:obj:`Dataset`): The dataset this object will wrap.
        """
        if not dataset:
            raise ValueError("Dataset needs to be open")
        if self._file is None:
            self._file = HDF5File(dataset.file)
        self._name = dataset.name
        self._dataset = dataset

    @dispatch(str)
    def set_dataset(self, dataset):
        """Sets the wrapped dataset base on a str.

        Args:
            dataset (str): The name of the dataset.
        """
        self._name = dataset

    # File
    def load(self):
        pass

    # Data Modification
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

    def require(self, name=None, **kwargs):
        if name is not None:
            self._name = name

        if "data" in kwargs:
            if "shape" not in kwargs:
                kwargs["shape"] = kwargs["data"].shape
            if "maxshape" not in kwargs:
                kwargs["maxshape"] = kwargs["data"].shape

        with self._file.temp_open():
            self._dataset = self._file.h5_fobj.require_dataset(name=self._name, **kwargs)

        return self

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

    def replace_data(self, data):
        with self:
            # Assign Data
            self._dataset.resize(data.shape)  # Reshape for new data
            self._dataset[...] = data


class HDF5File(HDF5BaseObject):
    """A class which wraps a HDF5 File and gives more functionality.

    Class Attributes:
        attribute_type: The class to cast the HDF5 attribute manager as.
        group_type: The class to cast the HDF5 group as.
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
    _wrap_attributes = ["_group", "_file"]
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
    def __init__(self, file=None, open_: bool = False, load: bool = False,
                 create: bool = False, build: bool = False, init: bool = True, **kwargs):
        super().__init__(init=False)

        self._path = None
        self._name_ = ""

        self._group = None

        if init:
            self.construct(file=file, open_=open_, load=load, create=create, build=build, **kwargs)

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
        # Todo: Change this to structure?
        return self._file[item]

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

    # Instance Methods
    # Constructors/Destructors
    def construct(self, file=None, open_: bool = False, load: bool = False,
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
        if file is not None:
            self._set_path(file)

        if self.path is None:
            if create:
                self.require_file(open_, **kwargs)
            elif load:
                raise ValueError("A file is required to load this file.")
            elif build:
                raise ValueError("A file is required to build this file.")
        elif open_:
            self.open(**kwargs)

        self.construct_group(load=load, build=build)

        return self

    def construct_group(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        if map_ is not None:
            self.map = map_
        self._group = self.group_type(name=self._name_, map_=self.map, file=self, load=load, build=build)

    # Getters/Setters
    @dispatch(object)
    def _set_path(self, file):
        if isinstance(file, HDF5File):
            self.path = file.path

    @dispatch((str, pathlib.Path))
    def _set_path(self, file):
        """Constructs the path attribute of this object.

        Args:
            file: The path to the file to build this object around.
        """
        self.path = file

    @dispatch(h5py.File)
    def _set_path(self, file):
        """Constructs the path attribute of this object.

        Args:
            file (obj:`File`): A HDF5 file to build this object around.
        """
        if file:
            self._file = file
            self.path = file.filename
        else:
            raise ValueError("The supplied HDF5 File must be open.")

    # File Creation/Construction
    def create_file(self, open_=True, map_: HDF5Map = None, build: bool = False, **kwargs):
        """Creates the HDF5 file.

        Args:
            attr (dict, optional): File attributes to set when the file is created.
            data (dict, optional): Datasets to assign when the file is created.
            construct (bool, optional): Determines if the file will be constructed.
            open_ (bool, optional): Determines if this object will remain open after construction.

        Returns:
            This object.
        """
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
HDF5Map.attributes_type = HDF5Attributes

HDF5Group.default_group = HDF5Group
HDF5Group.default_dataset = HDF5Dataset
