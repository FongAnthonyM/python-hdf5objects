#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" basehdf5.py
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
import pathlib
from warnings import warn

# Downloaded Libraries #
from baseobjects import AutomaticProperties
from bidict import bidict
from classversioning import VersionedInitMeta, VersionedClass, VersionType, TriNumberVersion
import h5py

# Local Libraries #
from ..hdf5object import HDF5Map, HDF5Group, HDF5Object


# Definitions #
# Classes #
class BaseHDF5Map(HDF5Map):
    default_type = HDF5Group
    default_attributes = bidict({"file_type": "FileType", "file_version": "FileVersion"})


class BaseHDF5(HDF5Object, AutomaticProperties, VersionedClass, metaclass=VersionedInitMeta):
    _registration = False
    _VERSION_TYPE = VersionType(name="BaseHDF5", class_=TriNumberVersion)
    FILE_TYPE = "Abstract"
    VERSION = TriNumberVersion(0, 0, 0)
    default_map = BaseHDF5Map()

    # Class Methods
    # Property Creation
    # Callbacks
    @classmethod
    def _get_attribute(cls, obj, name):
        try:
            setattr(obj, "_" + name, obj.attributes[obj.map.attributes[name]])
        finally:
            return getattr(obj, "_" + name)

    @classmethod
    def _set_attribute(cls, obj, name, value):
        try:
            obj.attributes[obj.map.attributes[name]] = value
        finally:
            setattr(obj, "_" + name, value)

    @classmethod
    def _del_attribute(cls, obj, name):
        try:
            del obj.attrs[obj.map.attributes[name]]
        finally:
            delattr(obj, "_" + name)

    # Callback Factories
    @classmethod
    def _attribute_callback_factory(cls, name):
        """A factory for creating property modification functions.

        Args:
            name (str): An object that can be use to create the get, set, and delete functions

        Returns:
            get_: The get function for a property object.
            set_: The wet function for a property object.
            del_: The del function for a property object.
        """

        def get_(obj):
            """Gets the object."""
            return cls._get_attribute(obj, name)

        def set_(obj, value):
            """Sets the wrapped object."""
            cls._set_attribute(obj, name, value)

        def del_(obj):
            """Deletes the wrapped object."""
            cls._del_attribute(obj, name)

        return get_, set_, del_

    # Property Constructors
    @classmethod
    def _hdf5map_to_properties(cls, map_, callback_factory):
        for name in map_.attributes.keys():
            if not hasattr(cls, name):
                get_, set_, del_ = callback_factory(name)
                setattr(cls, name, property(get_, set_, del_))

    # Property Mapping
    @classmethod
    def _construct_properties_map(cls):
        file_attributes = ["default_map", cls._hdf5map_to_properties, cls._attribute_callback_factory]
        cls._properties_map.append(file_attributes)

    # File Validation
    @classmethod
    def validate_file_type(cls, obj):
        t_name = cls.default_map.attributes["file_type"]

        if isinstance(obj, (str, pathlib.Path)):
            if not isinstance(obj, pathlib.Path):
                obj = pathlib.Path(obj)

            if obj.is_file():
                try:
                    with h5py.File(obj) as obj:
                        return t_name in obj.attrs and cls.FILE_TYPE == obj.attrs[t_name]
                except OSError:
                    return False
            else:
                return False
        elif isinstance(obj, HDF5Object):
            obj = obj.h5_fobj
            return t_name in obj.attrs and cls.FILE_TYPE == obj.attrs[t_name]

    @classmethod
    def validate_file(cls, obj):
        raise NotImplementedError

    @classmethod
    def new_validated(cls, obj, **kwargs):
        t_name = cls.default_map.attributes["file_type"]

        if isinstance(obj, (str, pathlib.Path)):
            if not isinstance(obj, pathlib.Path):
                obj = pathlib.Path(obj)

            if obj.is_file():
                try:
                    obj = h5py.File(obj)
                    if t_name in obj.attrs and cls.FILE_TYPE == obj.attrs[t_name]:
                        return cls(obj=obj, **kwargs)
                except OSError:
                    return None
            else:
                return None
        elif isinstance(obj, HDF5Object):
            obj = obj.h5_fobj
            if t_name in obj.attrs and cls.FILE_TYPE == obj.attrs[t_name]:
                return cls(obj=obj, **kwargs)

    @classmethod
    def get_version_from_object(cls, obj):
        """An optional abstract method that must return a version from an object."""
        v_name = cls.default_map.attributes["file_version"]

        if isinstance(obj, pathlib.Path):
            obj = obj.as_posix()

        if isinstance(obj, str):
            obj = h5py.File(obj)

        return TriNumberVersion(obj.attrs[v_name])

    # Magic Methods
    # Construction/Destruction
    def __new__(cls, *args, **kwargs):
        """With given input, will return the correct subclass."""
        if id(cls) == id(BaseHDF5) and (kwargs or args):
            if args:
                obj = args[0]
            else:
                obj = kwargs["obj"]
            class_ = cls.get_version_class(obj)
            return class_(*args, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self, init=True, **kwargs):
        self._h5_fobj = None
        super().__init__(init=False)
        self._file_type = ""
        self._file_version = ""

        if init:
            self.construct(**kwargs)

    @property
    def h5_fobj(self):
        return self._h5_fobj

    @h5_fobj.setter
    def h5_fobj(self, value):
        for attribute, name in self.map.attributes.items():
            try:
                new = getattr(self, "_" + attribute)
                value.attr[name] = new
            except AttributeError:
                pass
        self._h5_fobj = value

    # Instance Methods
    # File
    def open(self, mode="a", exc=False, validate=False, **kwargs):
        if not self.is_open:
            try:
                self.h5_fobj = h5py.File(self.path.as_posix(), mode=mode)
                self.construct_structure()
                if validate:
                    self.validate_file_structure(**kwargs)
                return self.h5_fobj
            except Exception as e:
                if exc:
                    warn("Could not open" + self.path.as_posix() + "due to error: " + str(e), stacklevel=2)
                    self.h5_fobj = None
                    return None
                else:
                    raise e

    def construct_file_attributes(self, **kwargs):
        self.attributes[self.map.attributes["file_type"]] = self.FILE_TYPE
        self.attributes[self.map.attributes["file_version"]] = self.VERSION.str()
        super().construct_file_attributes(**kwargs)

    # General Methods # Todo: Maybe implement this.
    # def report_file_structure(self):
    #     op = self.is_open
    #     self.open()
    #
    #     # Construct Structure Report Dictionary
    #     report = {"file_type": {"valid": False, "differences": {"object": self.FILE_TYPE, "file": None}},
    #               "attrs": {"valid": False, "differences": {"object": None, "file": None}},
    #               "datasets": {"valid": False, "differences": {"object": None, "file": None}}}
    #
    #     # Check H5 File Type
    #     if "FileType" in self.h5_fobj.attrs:
    #         if self.h5_fobj.attrs["FileType"] == self.FILE_TYPE:
    #             report["file_type"]["valid"] = True
    #             report["file_type"]["differences"]["object"] = None
    #         else:
    #             report["file_type"]["differences"]["file"] = self.h5_fobj.attrs["FileType"]
    #
    #     # Check File Attributes
    #     if self.h5_fobj.attrs.keys() == self.attributes:
    #         report["attrs"]["valid"] = True
    #     else:
    #         f_attr_set = set(self.h5_fobj.attrs.keys())
    #         o_attr_set = self.attributes
    #         report["attrs"]["differences"]["object"] = o_attr_set - f_attr_set
    #         report["attrs"]["differences"]["file"] = f_attr_set - o_attr_set
    #
    #     # Check File Datasets
    #     if self.h5_fobj.keys() == self._datasets:
    #         report["attrs"]["valid"] = True
    #     else:
    #         f_attr_set = set(self.h5_fobj.keys())
    #         o_attr_set = self._datasets
    #         report["datasets"]["differences"]["object"] = o_attr_set - f_attr_set
    #         report["datasets"]["differences"]["file"] = f_attr_set - o_attr_set
    #
    #     if not op:
    #         self.close()
    #     return report
    #
    # def validate_file_structure(self, file_type=True, o_attrs=True, f_attrs=False, o_datasets=True, f_datasets=False):
    #     report = self.report_file_structure()
    #     # Validate File Type
    #     if file_type and not report["file_type"]["valid"]:
    #         warn(self.path.as_posix() + " file type is not a " + self.FILE_TYPE, stacklevel=2)
    #     # Validate Attributes
    #     if not report["attrs"]["valid"]:
    #         if o_attrs and report["attrs"]["differences"]["object"] is not None:
    #             warn(self.path.as_posix() + " is missing attributes", stacklevel=2)
    #         if f_attrs and report["attrs"]["differences"]["file"] is not None:
    #             warn(self.path.as_posix() + " has extra attributes", stacklevel=2)
    #     # Validate Datasets
    #     if not report["datasets"]["valid"]:
    #         if o_datasets and report["datasets"]["differences"]["object"] is not None:
    #             warn(self.path.as_posix() + " is missing datasets", stacklevel=2)
    #         if f_datasets and report["datasets"]["differences"]["file"] is not None:
    #             warn(self.path.as_posix() + " has extra datasets", stacklevel=2)
