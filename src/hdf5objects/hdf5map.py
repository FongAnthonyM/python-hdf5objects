#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5map.py
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

# Default Libraries #

# Downloaded Libraries #
import h5py
from baseobjects import BaseObject
from bidict import bidict

# Local Libraries #


# Definitions #
# Classes #
class HDF5Map(BaseObject):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    __slots__ = ["name", "parent", "type", "attribute_type", "attributes", "containers", "maps"]
    default_name = ""
    default_parent = ""
    default_type = None
    default_attributes_type = None
    default_attributes = bidict()
    default_containers = bidict()
    default_maps = {}

    # Magic Methods
    # Construction/Destruction
    def __init__(self, name=None, type_=None, attributes=None, containers=None, maps=None, parent=None, init=True):
        self.name = self.default_name
        self.parent = self.default_parent
        self.type = self.default_type
        self.attributes_type = self.default_attributes_type
        self.attributes = self.default_attributes.copy()
        self.containers = self.default_containers.copy()
        self.maps = self.default_maps.copy()

        if init:
            self.construct(name, type_, attributes, containers, maps, parent)

    @property
    def full_name(self):
        return self.parent + "/" + self.name

    # Container Methods
    def __getitem__(self, name):
        """Gets a map within this object."""
        return self.maps[name]

    def __setitem__(self, name, value):
        """Sets a map within this object."""
        self.maps[name] = value

    def __delitem__(self, name):
        """Deletes a map within this object."""
        del self.maps[name]

    def __iter__(self):
        """Iterates over the maps within this object."""
        return self.maps.__iter__()

    def __contains__(self, item):
        """Determines if a map is within this object."""
        return item in self.maps

    # Instance Methods
    # Constructors/Destructors
    def construct(self, name=None, type_=None, attributes=None, containers=None, maps=None, parent=None):
        if name is not None:
            self.name = name

        if parent is not None:
            self.parent = parent

        if type_ is not None:
            self.type = type_

        if attributes is not None:
            self.attributes = attributes

        if containers is not None:
            self.containers = containers

        if maps is not None:
            self.maps = maps
            self.construct_maps()

    def construct_maps(self):
        pass

    def create_object(self, obj=None, file=None, **kwargs):
        if obj is None:
            new_obj = self.type(a_type=self.attributes_type, file=file, **kwargs)
        elif isinstance(obj, h5py.AttributeManager):
            new_obj = self.type(attributes=obj, a_type=self.attributes_type, file=file, **kwargs)
        elif isinstance(obj, h5py.Group):
            new_obj = self.type(group=obj, a_type=self.attributes_type, file=file, **kwargs)
        elif isinstance(obj, h5py.Dataset):
            new_obj = self.type(dataset=obj, a_type=self.attributes_type, file=file, **kwargs)
        else:
            new_obj = self.type(obj, a_type=self.attributes_type, file=file, **kwargs)

        new_obj.map = self
        return new_obj

    # Container
    def items(self):
        return self.maps.items()

    def keys(self):
        return self.maps.keys()

    def values(self):
        return self.maps.values()
