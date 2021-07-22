#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5structure.py
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

# Default Libraries  #

# Downloaded Libraries #
from baseobjects import BaseObject
import h5py

# Local Libraries #
from .hdf5map import HDF5Map


# Definitions #
# Classes #
class HDF5Structure(BaseObject):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    __slots__ = ["name", "parent", "map", "object", "containers"]
    attribute_type = HDF5Attributes
    group_type = HDF5Group
    dataset_type = HDF5Dataset
    default_name = ""
    default_parent = ""
    default_map = HDF5Map
    default_containers = {}

    # Class Methods
    @classmethod
    def from_map(cls, map_, name=None):
        return cls(name=name, map_=map_)

    # Magic Methods
    # Construction/Destruction
    def __init__(self, name=None, containers=None, map_=None, parent=None, file=None, init=True):
        self.name = self.default_name
        self.parent = self.default_parent
        self.map = None
        self.object = None
        self.containers = {}

        if init:
            self.construct(name, containers, map_, parent, file)

    @property
    def full_name(self):
        return self.parent + "/" + self.name

    # Container Methods
    def __getitem__(self, name):
        """Gets a container within this object."""
        return self.containers[name]

    def __setitem__(self, name, value):
        """Sets a container within this object."""
        self.containers[name] = value

    def __delitem__(self, name):
        """Deletes a container within this object."""
        del self.containers[name]

    def __iter__(self):
        """Iterates over the containers within this object."""
        return self.containers.__iter__()

    def __contains__(self, item):
        """Determines if a container is within this object."""
        return item in self.containers

    # Instance Methods
    # Constructors/Destructors
    def construct(self, name=None, containers=None, map_=None, parent=None, file=None):
        if name is not None:
            self.name = name

        if parent is not None:
            self.parent = parent

        if containers is not None:
            self.containers = containers

        if map_ is not None:
            self.map = map_
            self.construct_from_map(map_=map_)

        if file is not None:
            self.construct_object(file)
            self.construct_containers(file)

    def construct_from_map(self, map_):
        self.name = map_.name
        for name, inner_map_ in map_.items():
            self.containers[name] = type(self)(name=name, map_=inner_map_, parent=self.full_name)

    def construct_object(self, file):
        obj = file[self.full_name]
        if self.map is None:
            self.object = self.assign_type(obj, file)
        else:
            self.object = self.map.type(obj, file=file)

    def construct_containers(self, file):
        if isinstance(file[self.full_name], h5py.Group):
            for name, obj in file[self.full_name].items():
                if self.map and name in self.map:
                    map_ = self.map[name]
                else:
                    map_ = None

                if name not in self.containers:
                    self.containers[name] = type(self)(name=name, map_=map_, parent=self.full_name, file=file)
                elif self.containers[name].object is None:
                    self.containers[name].construct_object(file)
                    self.containers[name].construct_containers(file)

    def assign_type(self, obj, file=None):
        if isinstance(obj, h5py.AttributeManager):
            return self.attribute_type(attributes=obj, file=file)
        elif isinstance(obj, h5py.Group):
            return self.group_type(group=obj, file=file)
        elif isinstance(obj, h5py.Dataset):
            return self.dataset_type(dataset=obj, file=file)



