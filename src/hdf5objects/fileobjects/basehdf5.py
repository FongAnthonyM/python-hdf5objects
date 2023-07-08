"""basehdf5.py
A more specific HDF5File which implements versioning and validation.
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
import pathlib
from typing import Any, Union

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch
from classversioning import (
    CachingVersionedInitMeta,
    VersionedClass,
    VersionType,
    TriNumberVersion,
    Version,
)
import h5py

# Local Packages #
from ..hdf5bases import HDF5Map, FileMap, HDF5File


# Definitions #
# Classes #
class BaseHDF5Map(FileMap):
    """A map for BaseHDF5 files."""

    default_attribute_names = {"file_type": "FileType", "file_version": "FileVersion"}
    default_attributes = {"file_type": "", "file_version": ""}


class BaseHDF5(HDF5File, VersionedClass, metaclass=CachingVersionedInitMeta):
    """A more specific HDF5File which implements versioning and validation.

    Class Attributes:
        _registration: Determines if this class will be included in class registry.
        _VERSION_TYPE: The type of versioning to use.
        FILE_TYPE: The file type name of this class.
        VERSION: The version of this class.
        default_map: The HDF5 map of this object.

    Attributes:
        _file_type: The type of file this object is.
        _file_version: The version of this file.

    Args:
        file: Either the file object or the path to the file.
        open_: Determines if this object will remain open after construction.
        map_: The map for this HDF5 object.
        load: Determines if this object will load the file on construction.
        create: Determines if this object will create an empty file on construction.
        require: Determines if this object will create and fill the file on construction.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the open method.
    """

    _dispatch_kwarg: str = "file"
    _registration: bool = False
    _VERSION_TYPE: VersionType = VersionType(name="BaseHDF5", class_=TriNumberVersion)
    FILE_TYPE: str = "Abstract"
    VERSION: Version = TriNumberVersion(0, 0, 0)
    default_map: HDF5Map = BaseHDF5Map()

    # Class Methods #
    # File Validation
    @singlekwargdispatch("file")
    @classmethod
    def validate_file_type(cls, file: pathlib.Path | str | HDF5File | h5py.File) -> bool:
        """Checks if the given file or path is a valid type.

        Args:
            file: The path or file object.

        Returns:
            If this is a valid file type.
        """
        raise TypeError(f"{type(file)} is not a valid type for validate_file_type.")

    @validate_file_type.register
    @classmethod
    def _validate_file_type(cls, file: pathlib.Path) -> bool:
        """Checks if the given path is a valid type.

        Args:
            file: The path.

        Returns:
            If this is a valid file type.
        """
        t_name = cls.default_map.attribute_names["file_type"]

        if file.is_file():
            try:
                with h5py.File(file) as obj:
                    return t_name in obj.attrs and cls.FILE_TYPE == obj.attrs[t_name]
            except OSError:
                return False
        else:
            return False

    @validate_file_type.register
    @classmethod
    def _validate_file_type(cls, file: str) -> bool:
        """Checks if the given path is a valid type.

        Args:
            file: The path.

        Returns:
            If this is a valid file type.
        """
        t_name = cls.default_map.attribute_names["file_type"]

        file = pathlib.Path(file)

        if file.is_file():
            try:
                with h5py.File(file) as obj:
                    return t_name in obj.attrs and cls.FILE_TYPE == obj.attrs[t_name]
            except OSError:
                return False
        else:
            return False

    @validate_file_type.register
    @classmethod
    def _validate_file_type(cls, file: HDF5File) -> bool:
        """Checks if the given file is a valid type.

        Args:
            file: The file object.

        Returns:
            If this is a valid file type.
        """
        t_name = cls.default_map.attribute_names["file_type"]
        file = file._file
        return t_name in file.attrs and cls.FILE_TYPE == file.attrs[t_name]

    @validate_file_type.register
    @classmethod
    def _validate_file_type(cls, file: h5py.File) -> bool:
        """Checks if the given file is a valid type.

        Args:
            file: The file object.

        Returns:
            If this is a valid file type.
        """
        t_name = cls.default_map.attribute_names["file_type"]
        return t_name in file.attrs and cls.FILE_TYPE == file.attrs[t_name]

    @classmethod
    def validate_file(cls, file: pathlib.Path | str | HDF5File | h5py.File) -> bool:
        """A method for checking if a file valid.

        Args:
            file: The path or file object.

        Returns:
            If this is a valid file.
        """
        raise NotImplementedError

    @singlekwargdispatch("file")
    @classmethod
    def new_validated(cls, file: pathlib.Path | str | HDF5File | h5py.File, **kwargs: Any) -> Union["BaseHDF5", None]:
        """Checks if the given file or path is a valid type and returns the file if valid.

        Args:
            file: The path or file object.

        Returns:
            The file or None.
        """
        raise TypeError(f"{type(file)} is not a valid type for new_validate.")

    @new_validated.register
    @classmethod
    def _new_validated(cls, file: pathlib.Path, **kwargs: Any) -> Any:
        """Checks if the given path is a valid type and returns the file if valid.

        Args:
            file: The path.

        Returns:
            The file or None.
        """
        t_name = cls.default_map.attribute_names["file_type"]

        if file.is_file():
            try:
                file = h5py.File(file)
                if t_name in file.attrs and cls.FILE_TYPE == file.attrs[t_name]:
                    return cls(file=file, **kwargs)
            except OSError:
                return None
        else:
            return None

    @new_validated.register
    @classmethod
    def _new_validated(cls, file: str, **kwargs: Any) -> Any:
        """Checks if the given path is a valid type and returns the file if valid.

        Args:
            file: The path.

        Returns:
            The file or None.
        """
        t_name = cls.default_map.attribute_names["file_type"]
        file = pathlib.Path(file)

        if file.is_file():
            try:
                file = h5py.File(file)
                if t_name in file.attrs and cls.FILE_TYPE == file.attrs[t_name]:
                    return cls(file=file, **kwargs)
            except OSError:
                return None
        else:
            return None

    @new_validated.register
    @classmethod
    def _new_validated(cls, file: HDF5File, **kwargs: Any) -> Any:
        """Checks if the given file is a valid type and returns the file if valid.

        Args:
            file: The file.

        Returns:
            The file or None.
        """
        t_name = cls.default_map.attribute_names["file_type"]
        file = file._file
        if t_name in file.attrs and cls.FILE_TYPE == file.attrs[t_name]:
            return cls(file=file, **kwargs)
        else:
            return None

    @new_validated.register
    @classmethod
    def _new_validated(cls, file: h5py.File, **kwargs: Any) -> Any:
        """Checks if the given file is a valid type and returns the file if valid.

        Args:
            file: The file.

        Returns:
            The file or None.
        """
        t_name = cls.default_map.attribute_names["file_type"]
        if t_name in file.attrs and cls.FILE_TYPE == file.attrs[t_name]:
            return cls(file=file, **kwargs)
        else:
            return None

    @classmethod
    def get_version_from_file(cls, file: pathlib.Path | str | h5py.File) -> Version:
        """Return a version from a file.

        Args:
            file: The path to file to get the version from.

        Returns:
            The version from the file.
        """
        v_name = cls.default_map.attribute_names["file_version"]

        if isinstance(file, pathlib.Path):
            file = file.as_posix()

        if isinstance(file, str):
            file = h5py.File(file)

        return TriNumberVersion(file.attrs[v_name])

    @classmethod
    def get_version_from_object(cls, obj: Any) -> Version:
        """Returns a version from an object.

        Args:
            obj: The path to file to get the version from.

        Returns:
            The version from the file.
        """
        return cls.get_version_from_file(obj)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        file: str | pathlib.Path | h5py.File | None = None,
        open_: bool = True,
        map_: HDF5Map | None = None,
        load: bool = False,
        create: bool = False,
        require: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._file_type: str = ""
        self._file_version: str = ""

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(
                file=file,
                open_=open_,
                map_=map_,
                load=load,
                create=create,
                require=require,
                **kwargs,
            )

    @property
    def file_type(self):
        """Gets the file type from the attributes."""
        return self.attributes["file_type"]

    @file_type.setter
    def file_type(self, value):
        self.attributes.set_attribute("file_type", value)

    @property
    def file_version(self):
        """Gets the file version from the file."""
        return self.attributes["file_version"]

    @file_version.setter
    def file_version(self, value):
        self.attributes.set_attribute("file_version", value)

    # Instance Methods #
    # Constructors/Destructors
    def construct_file_attributes(self, map_: HDF5Map = None, load: bool = False, require: bool = False) -> None:
        """Creates the attributes for this group.

        Args:
            map_: The map to use to create the attributes.
            load: Determines if this object will load the attribute values from the file on construction.
            require: Determines if this object will create and fill the attributes in the file on construction.
        """
        super().construct_file_attributes(map_=map_, load=load, require=require)
        self._group.attributes["file_type"] = self.FILE_TYPE
        self._group.attributes["file_version"] = self.VERSION.str()

    # File  # Todo: Maybe implement this.
    # def open(self, mode="a", exc=False, validate=False, **kwargs):
    #     if not self.is_open:
    #         try:
    #             self._file = h5py.File(self.path.as_posix(), mode=mode, **kwargs)
    #             if validate:
    #                 self.validate_file_structure(**kwargs)
    #             return self
    #         except Exception as e:
    #             if exc:
    #                 warn("Could not open" + self.path.as_posix() + "due to error: " + str(e), stacklevel=2)
    #                 self._file = None
    #                 return None
    #             else:
    #                 raise e
    #
    # General Methods
    # def report_file_structure(self):
    #     op = self.is_open
    #     self.open()
    #
    #     # Construct Structure Report Dictionary
    #     report = {"file_type": {"valid": False, "differences": {"object": self.FILE_TYPE, "file": None}},
    #               "attrs": {"valid": False, "differences": {"object": None, "file": None}},
    #               "dataset": {"valid": False, "differences": {"object": None, "file": None}}}
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
    #         report["dataset"]["differences"]["object"] = o_attr_set - f_attr_set
    #         report["dataset"]["differences"]["file"] = f_attr_set - o_attr_set
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
    #     if not report["dataset"]["valid"]:
    #         if o_datasets and report["dataset"]["differences"]["object"] is not None:
    #             warn(self.path.as_posix() + " is missing dataset", stacklevel=2)
    #         if f_datasets and report["dataset"]["differences"]["file"] is not None:
    #             warn(self.path.as_posix() + " has extra dataset", stacklevel=2)
