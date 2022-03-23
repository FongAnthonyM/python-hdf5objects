""" hdf5baseobject.py
The base object for hdf5 objects.
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
from typing import Any

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod, search_sentinel
from baseobjects.cachingtools import CachingInitMeta, CachingObject
from baseobjects.objects import StaticWrapper
import h5py

# Local Packages #
from .hdf5map import HDF5Map


# Definitions #
# Classes #
class HDF5BaseObject(StaticWrapper, CachingObject, metaclass=CachingInitMeta):
    """An abstract wrapper which wraps object from an HDF5 file and gives more functionality.

    Class Attributes:
        sentinel: An object that helps with mapping searches.
        file_type: The type of the file to use when creating a file object.
        default_map: The map of this HDF5 object.

    Attributes:
        _file_was_open: Determines if the file object was open when this dataset was accessed.
        _file: The file object that this HDF5 object originates from.
        _name_: The HDF5 name of this object.
        _parents: The parents of this object as a list.
        map: The map of this HDF5 object.

    Args:
        name: The HDF5 name of this object.
        map_: The map for this HDF5 object.
        file: The file object that this HDF5 object originates from.
        parent: The HDF5 name of the parent of this HDF5 object.
        init: Determines if this object will construct.
    """
    sentinel: Any = search_sentinel
    file_type: type | None = None
    default_map: HDF5Map | None = None

    # Class Methods
    # Wrapped Attribute Callback Functions
    @classmethod
    def _get_attribute(cls, obj: Any, wrap_name: str, attr_name: str) -> Any:
        """Gets an attribute from a wrapped HDF5 object.

        Args:
            obj: The target object to get the wrapped object from.
            wrap_name: The attribute name of the wrapped object.
            attr_name: The attribute name of the attribute to get from the wrapped object.

        Returns:
            The wrapped object.
        """
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
            return super()._get_attribute(obj, wrap_name, attr_name)

    @classmethod
    def _set_attribute(cls, obj: Any, wrap_name: str, attr_name: str, value: Any) -> None:
        """Sets an attribute in a wrapped HDF5 object.

        Args:
            obj: The target object to set.
            wrap_name: The attribute name of the wrapped object.
            attr_name: The attribute name of the attribute to set from the wrapped object.
            value: The object to set the wrapped fileobjects attribute to.
        """
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
            super()._set_attribute(obj, wrap_name, attr_name, value)

    @classmethod
    def _del_attribute(cls, obj: Any, wrap_name: str, attr_name: str) -> None:
        """Deletes an attribute in a wrapped HDF5 object.

        Args:
            obj: The target object to set.
            wrap_name: The attribute name of the wrapped object.
            attr_name: The attribute name of the attribute to delete from the wrapped object.
        """
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
            super()._del_attribute(obj, wrap_name, attr_name)

    # Magic Methods
    # Constructors/Destructors
    def __init__(
        self,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        parent: str | None = None,
        init: bool = True,
    ) -> None:
        # Parent Attributes #
        super().__init__()

        # New Attributes #
        self._file_was_open: bool | None = None
        self._file: h5py.File | HDF5File | None = None

        self._name_: str | None = None
        self._parents: list[str] | None = None

        self.map: HDF5Map = self.default_map.copy()

        # Object Construction #
        if init:
            self.construct(name=name, map_=map_, file=file, parent=parent)

    @property
    def _name(self) -> str:
        """The name of this map. The setter supports parsing a full hdf5 name."""
        return self._name_

    @_name.setter
    def _name(self, value: str) -> None:
        self.set_name(name=value)

    @property
    def _parent(self) -> str:
        """Concatenates the parents into one str. The setter supports parsing a full hdf5 name."""
        if self._parents is None:
            return "/"
        else:
            return "".join(f"/{p}" for p in self._parents)

    @_parent.setter
    def _parent(self, value: str | None):
        if value is None:
            self._parents = None
        else:
            self.set_parent(parent=value)

    @property
    def _full_name(self) -> str:
        """Returns the full hdf5 name of this map."""
        if self._parents is None:
            return f"/{self._name_}"
        else:
            return f"{''.join(f'/{p}' for p in self._parents)}{self._name_}"

    @property
    def exists(self) -> bool:
        """Checks if this object exists in the hdf5 file."""
        return self.is_exist()

    # Container Methods
    def __getitem__(self, key: str) -> Any:
        """Ensures HDF5 object is open for getitem"""
        with self:
            return getattr(self, self._wrap_attributes[0])[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Ensures HDF5 object is open for setitem"""
        with self:
            getattr(self, self._wrap_attributes[0])[key] = value

    def __delitem__(self, key: str) -> None:
        """Ensures HDF5 object is open for delitem"""
        with self:
            del getattr(self, self._wrap_attributes[0])[key]

    # Context Managers
    def __enter__(self) -> "HDF5BaseObject":
        """The enter context which opens the file to make this dataset usable"""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """The exit context which close the file."""
        self.close()

    # Type Conversion
    def __bool__(self) -> bool:
        """When cast as a bool, this object True if valid and False if not.

        Returns:
            bool: If this object is open or not.
        """
        return bool(getattr(self, self._wrap_attributes[0]))

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        parent: str | None = None,
    ) -> None:
        """Constructs this object.

        Args:
            name: The HDF5 name of this object.
            map_: The map for this HDF5 object.
            file: The file object that this HDF5 object originates from.
            parent: The HDF5 name of the parent of this HDF5 object.
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

    def is_exist(self) -> bool:
        """Determine if this object exists in the HDF5 file."""
        with self._file.temp_open():
            try:
                self._file._file[self._full_name]
                return True
            except KeyError:
                return False

    # File
    def open(self, mode: str = 'a', **kwargs: Any) -> "HDF5BaseObject":
        """Opens the file to make this dataset usable.

        Args:
            mode: The file mode to open the file with.
            **kwargs: The additional keyword arguments to open the file with.

        Returns:
            This object.
        """
        self._file_was_open = self._file.is_open
        if not getattr(self, self._wrap_attributes[0]):
            self._file.open(mode=mode, **kwargs)
            setattr(self, self._wrap_attributes[0], self._file._file[self._full_name])

        return self

    def close(self) -> None:
        """Closes the file of this dataset."""
        if not self._file_was_open:
            self._file.close()

    # Getters/Setters
    @singlekwargdispatchmethod("file")
    def set_file(self, file: str | pathlib.Path | h5py.File) -> None:
        """Sets the file for this object to an HDF5File.

        Args:
            file: An object to set the file to.
        """
        if isinstance(file, self.file_type):
            self._file = file
        else:
            raise TypeError("file must be a path, File, or HDF5File")

    @set_file.register(str)
    @set_file.register(pathlib.Path)
    @set_file.register(h5py.File)
    def _(self, file: str | pathlib.Path | h5py.File) -> None:
        """Sets the file for this object to an HDF5File.

        Args:
            file: An object to set the file to.
        """
        self._file = self.file_type(file)

    def set_parent(self, parent: str) -> None:
        """Sets the parent of this object to the str

        Args:
            parent: The str to parse and set as the parent of this map.
        """
        parent = parent.lstrip('/')
        parts = parent.split('/')
        self._parents = parts

    def set_name(self, name: str | None) -> None:
        """Sets the name of this map, can be a full hdf5 name.

        Args:
            name: The name of this map, can be a full hdf5 name.
        """
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
