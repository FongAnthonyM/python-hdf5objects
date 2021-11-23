#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" basehdf5.py
Description:
"""
# Package Header #
from ..__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
import pathlib
from warnings import warn

# Third-Party Packages #
from classversioning import CachingVersionedInitMeta, VersionedClass, VersionType, TriNumberVersion
import h5py

# Local Packages #
from ..hdf5objects import HDF5Map, HDF5Group, HDF5File


# Definitions #
# Classes #
class BaseHDF5Map(HDF5Map):
    default_type = HDF5Group
    default_attribute_names = {"file_type": "FileType", "file_version": "FileVersion"}


class BaseHDF5(HDF5File, VersionedClass, metaclass=CachingVersionedInitMeta):
    _registration = False
    _VERSION_TYPE = VersionType(name="BaseHDF5", class_=TriNumberVersion)
    FILE_TYPE = "Abstract"
    VERSION = TriNumberVersion(0, 0, 0)
    default_map = BaseHDF5Map()

    # Class Methods #
    # File Validation
    @classmethod
    def validate_file_type(cls, obj):
        t_name = cls.default_map.attribute_names["file_type"]

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
        elif isinstance(obj, HDF5File):
            obj = obj._file
            return t_name in obj.attrs and cls.FILE_TYPE == obj.attrs[t_name]

    @classmethod
    def validate_file(cls, obj):
        raise NotImplementedError

    @classmethod
    def new_validated(cls, obj, **kwargs):
        t_name = cls.default_map.attribute_names["file_type"]

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
        elif isinstance(obj, HDF5File):
            obj = obj._file
            if t_name in obj.attrs and cls.FILE_TYPE == obj.attrs[t_name]:
                return cls(obj=obj, **kwargs)

    @classmethod
    def get_version_from_object(cls, obj):
        """An optional abstract method that must return a version from an object."""
        v_name = cls.default_map.attribute_names["file_version"]

        if isinstance(obj, pathlib.Path):
            obj = obj.as_posix()

        if isinstance(obj, str):
            obj = h5py.File(obj)

        return TriNumberVersion(obj.attrs[v_name])

    # Magic Methods #
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

    def __init__(self, file=None, open_: bool = True, map_: HDF5Map = None, load: bool = False,
                 create: bool = False, build: bool = False, init: bool = True, **kwargs):
        super().__init__(init=False)
        self._file_type = ""
        self._file_version = ""

        if init:
            self.construct(file=file, open_=open_, map_=map_, load=load, create=create, build=build, **kwargs)
    
    @property
    def file_type(self):
        return self.attributes["file_type"]

    @file_type.setter
    def file_type(self, value):
        self.attributes.set_attribute("file_type", value)
        
    @property
    def file_version(self):
        return self.attributes["file_version"]

    @file_version.setter
    def file_version(self, value):
        self.attributes.set_attribute("file_version", value)
    
    # Instance Methods #
    # Constructors/Destructors
    def construct_file_attributes(self):
        self.attributes["file_type"] = self.FILE_TYPE
        self.attributes["file_version"] = self.VERSION.str()

    # File
    def open(self, mode="a", exc=False, validate=False, **kwargs):
        if not self.is_open:
            try:
                self._file = h5py.File(self.path.as_posix(), mode=mode, **kwargs)
                if validate:
                    self.validate_file_structure(**kwargs)
                return self
            except Exception as e:
                if exc:
                    warn("Could not open" + self.path.as_posix() + "due to error: " + str(e), stacklevel=2)
                    self._file = None
                    return None
                else:
                    raise e

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
    #     if "FileType" in self._file.attrs:
    #         if self._file.attrs["FileType"] == self.FILE_TYPE:
    #             report["file_type"]["valid"] = True
    #             report["file_type"]["differences"]["object"] = None
    #         else:
    #             report["file_type"]["differences"]["file"] = self._file.attrs["FileType"]
    #
    #     # Check File Attributes
    #     if self._file.attrs.keys() == self.attributes:
    #         report["attrs"]["valid"] = True
    #     else:
    #         f_attr_set = set(self._file.attrs.keys())
    #         o_attr_set = self.attributes
    #         report["attrs"]["differences"]["object"] = o_attr_set - f_attr_set
    #         report["attrs"]["differences"]["file"] = f_attr_set - o_attr_set
    #
    #     # Check File Datasets
    #     if self._file.keys() == self._datasets:
    #         report["attrs"]["valid"] = True
    #     else:
    #         f_attr_set = set(self._file.keys())
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
