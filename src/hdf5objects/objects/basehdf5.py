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
from abc import ABC
import pathlib
from warnings import warn

# Downloaded Libraries #
from classversioning import VersionedClass, VersionType, TriNumberVersion
import h5py

# Local Libraries #
from ..hdf5object import HDF5Object


# Definitions #
# Classes #
class BaseHDF5(HDF5Object, VersionedClass):
    _registration = False
    _VERSION_TYPE = VersionType(name="BaseHDF5", class_=TriNumberVersion)
    VERSION = TriNumberVersion(0, 0, 0)
    file_attribute_map = {"file_type": "FileType", "version": "Version"}
    dataset_map = {}
    FILE_TYPE = "Abstract"

    # Class Methods
    @classmethod
    def validate_file_type(cls, obj):
        t_name = cls.file_attribute_map["file_type"]

        if isinstance(obj, pathlib.Path):
            obj = obj.as_posix()

        if isinstance(obj, str):
            obj = HDF5Object(obj)

        if isinstance(obj, h5py.File):
            obj = HDF5Object(obj)

        return cls.FILE_TYPE == obj[t_name]

    @classmethod
    def validate_file(cls, obj):
        raise NotImplementedError

    @classmethod
    def get_version_from_object(cls, obj):
        """An optional abstract method that must return a version from an object."""
        v_name = cls.file_attribute_map["version"]

        if isinstance(obj, pathlib.Path):
            obj = obj.as_posix()

        if isinstance(obj, str):
            obj = h5py.File(obj)

        return TriNumberVersion(obj.attrs[v_name])

    # Magic Methods
    # Construction/Destruction
    def __new__(cls, *args, **kwargs):
        """With given input, will return the correct subclass."""
        if cls == BaseHDF5 and (kwargs or args):
            if args:
                obj = args[0]
            else:
                obj = kwargs["obj"]
            class_ = cls.get_version_class(obj)
            return class_(*args, **kwargs)
        else:
            return super(BaseHDF5, cls).__new__(cls)

    # Instance Methods
    # File
    def open(self, mode="a", exc=False, validate=False, **kwargs):
        if not self.is_open:
            try:
                self.h5_fobj = h5py.File(self.path.as_posix(), mode=mode)
            except Exception as e:
                if exc:
                    warn("Could not open" + self.path.as_posix() + "due to error: " + str(e), stacklevel=2)
                    self.h5_fobj = None
                    return None
                else:
                    raise e
            else:
                if validate:
                    self.validate_file_structure(**kwargs)
                self.load_attributes()
                self.load_datasets()
                return self.h5_fobj

    # General Methods
    def report_file_structure(self):
        op = self.is_open
        self.open()

        # Construct Structure Report Dictionary
        report = {"file_type": {"valid": False, "differences": {"object": self.FILE_TYPE, "file": None}},
                  "attrs": {"valid": False, "differences": {"object": None, "file": None}},
                  "datasets": {"valid": False, "differences": {"object": None, "file": None}}}

        # Check H5 File Type
        if "FileType" in self.h5_fobj.attrs:
            if self.h5_fobj.attrs["FileType"] == self.FILE_TYPE:
                report["file_type"]["valid"] = True
                report["file_type"]["differences"]["object"] = None
            else:
                report["file_type"]["differences"]["file"] = self.h5_fobj.attrs["FileType"]

        # Check File Attributes
        if self.h5_fobj.attrs.keys() == self._file_attrs:
            report["attrs"]["valid"] = True
        else:
            f_attr_set = set(self.h5_fobj.attrs.keys())
            o_attr_set = self._file_attrs
            report["attrs"]["differences"]["object"] = o_attr_set - f_attr_set
            report["attrs"]["differences"]["file"] = f_attr_set - o_attr_set

        # Check File Datasets
        if self.h5_fobj.keys() == self._datasets:
            report["attrs"]["valid"] = True
        else:
            f_attr_set = set(self.h5_fobj.keys())
            o_attr_set = self._datasets
            report["datasets"]["differences"]["object"] = o_attr_set - f_attr_set
            report["datasets"]["differences"]["file"] = f_attr_set - o_attr_set

        if not op:
            self.close()
        return report

    def validate_file_structure(self, file_type=True, o_attrs=True, f_attrs=False, o_datasets=True, f_datasets=False):
        report = self.report_file_structure()
        # Validate File Type
        if file_type and not report["file_type"]["valid"]:
            warn(self.path.as_posix() + " file type is not a " + self.FILE_TYPE, stacklevel=2)
        # Validate Attributes
        if not report["attrs"]["valid"]:
            if o_attrs and report["attrs"]["differences"]["object"] is not None:
                warn(self.path.as_posix() + " is missing attributes", stacklevel=2)
            if f_attrs and report["attrs"]["differences"]["file"] is not None:
                warn(self.path.as_posix() + " has extra attributes", stacklevel=2)
        # Validate Datasets
        if not report["datasets"]["valid"]:
            if o_datasets and report["datasets"]["differences"]["object"] is not None:
                warn(self.path.as_posix() + " is missing datasets", stacklevel=2)
            if f_datasets and report["datasets"]["differences"]["file"] is not None:
                warn(self.path.as_posix() + " has extra datasets", stacklevel=2)
