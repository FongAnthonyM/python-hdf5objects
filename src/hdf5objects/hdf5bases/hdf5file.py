""" hdf5file.py
Description:
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
from collections.abc import Mapping
from contextlib import contextmanager
import pathlib
from typing import Any
from warnings import warn

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod
import h5py

# Local Packages #
from .hdf5map import HDF5Map
from .hdf5baseobject import HDF5BaseObject
from .hdf5attributes import HDF5Attributes
from .hdf5group import HDF5Group
from .hdf5dataset import HDF5Dataset


# Definitions #
# Classes #
class HDF5File(HDF5BaseObject):
    """A class which wraps a HDF5 File and gives more functionality, but retains its generalization.

    Class Attributes:
        _wrapped_types: A list of either types or objects to set up wrapping for.
        _wrap_attributes: Attribute names that will contain the objects to wrap where the resolution order is descending
            inheritance.
        attribute_type: The class to cast the HDF5 attribute manager as.
        group_type: The class to cast the HDF5 group as.
        dataset_type: The class to cast the HDF5 dataset as.

    Attributes:
        _path: The path to the file.
        _name_: The name of the first layer in the file.
        _group: The first layer group this object will wrap.

    Args:
        file: Either the file object or the path to the file.
        open_: Determines if this object will remain open after construction.
        map_: The map for this HDF5 object.
        load: Determines if this object will load the file on construction.
        create: Determines if this object will create an empty file on construction.
        build: Determines if this object will create and fill the file on construction.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the open method.
    """
    # Todo: Rethink about how Errors and Warnings are handled in this object.
    _wrapped_types: list[type | object] = [HDF5Group, h5py.File]
    _wrap_attributes: list[str] = ["group", "_file"]
    attribute_type: type = HDF5Attributes
    group_type: type = HDF5Group
    dataset_type: type = HDF5Dataset

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
        with obj.temp_open():  # Ensures the hdf5 dataset is open when accessing attributes
            return super()._get_attribute(obj, wrap_name, attr_name)

    @classmethod
    def _set_attribute(cls, obj: Any, wrap_name: str, attr_name: str, value: Any) -> None:
        """Sets an attribute in a wrapped HDF5 object.

        Args:
            obj: The target object to set.
            wrap_name: The attribute name of the wrapped object.
            attr_name: The attribute name of the attribute to set from the wrapped object.
            value: The object to set the wrapped file objects attribute to.
        """
        with obj.temp_open():  # Ensures the hdf5 dataset is open when accessing attributes
            super()._set_attribute(obj, wrap_name, attr_name, value)

    @classmethod
    def _del_attribute(cls, obj: Any, wrap_name: str, attr_name: str) -> None:
        """Deletes an attribute in a wrapped HDF5 object.

        Args:
            obj: The target object to set.
            wrap_name: The attribute name of the wrapped object.
            attr_name: The attribute name of the attribute to delete from the wrapped object.
        """
        with obj.temp_open():  # Ensures the hdf5 dataset is open when accessing attributes
            super()._del_attribute(obj, wrap_name, attr_name)

    # Validation #
    @classmethod
    def is_openable(cls, path: str | pathlib.Path) -> bool:
        """Checks if a path can be opened as an HDF5 file.

        Args:
            path: The path of the file to validate.
        """
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
    def __init__(
        self,
        file: str | pathlib.Path | h5py.File | None = None,
        open_: bool = False,
        map_: HDF5Map | None = None,
        load: bool = False,
        create: bool = False,
        build: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._path: pathlib.Path | None = None
        self._name_: str = "/"

        self._group: HDF5Group | None = None

        # Object Construction #
        if init:
            self.construct(file=file, open_=open_, map_=map_, load=load, create=create, build=build, **kwargs)

    @property
    def path(self) -> pathlib.Path:
        """The path to the file. The setter casts file objects that are not Path to path before setting"""
        return self._path

    @path.setter
    def path(self, value: str | pathlib) -> None:
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    @property
    def is_open(self) -> bool:
        """Determines if the hdf5 file is open."""
        try:
            return bool(self._file)
        except:
            return False

    @property
    def attributes(self) -> HDF5Attributes:
        """Gets the attributes of the file."""
        return self._group.attributes

    def __del__(self) -> None:
        """Closes the file when this object is deleted."""
        self.close()

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            dict: A dictionary of this object's attributes.
        """
        state = self.__dict__.copy()
        state["open_state"] = self.is_open
        del state["_file"]
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Builds this object based on a dictionary of corresponding attributes.

        Args:
            state: The attributes to build this object from.
        """
        state["_file"] = h5py.File(state["path"].as_posix(), "r+")
        if not state.pop("open_state"):
            state["_file"].close()
        self.__dict__.update(state)

    # Container Methods
    def __getitem__(self, key: str) -> HDF5BaseObject:
        """Gets a HDF5 object from the HDF5 file.

        Args:
           key: The name of the HDF5 object to get.

        Returns:
            The HDF5 object requested.
        """
        return self._group[key]

    # Context Managers
    def __enter__(self) -> "HDF5File":
        """The context enter which opens the HDF5 file.

        Returns:
            This object.
        """
        if self._file is None:
            self.construct(open_=True)
        else:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """The context exit which closes the file."""
        return self.close()

    # Type Conversion
    def __bool__(self) -> bool:
        """When cast as a bool, this object True if open and False if closed.

        Returns:
            If this object is open or not.
        """
        return self.is_open

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        file: str | pathlib.Path | h5py.File | None = None,
        open_: bool = False,
        map_: HDF5Map | None = None,
        load: bool = False,
        create: bool = False,
        build: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> "HDF5File":
        """Constructs this object.

        Args:
            file: Either the file object or the path to the file.
            open_: Determines if this object will remain open after construction.
            map_: The map for this HDF5 object.
            load: Determines if this object will load the file on construction.
            create: Determines if this object will create an empty file on construction.
            build: Determines if this object will create and fill the file on construction.
            init: Determines if this object will construct.
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
                self.require_file(open_=open_, **kwargs)
            elif load:
                raise ValueError("A file is required to load this file.")
            elif build:
                raise ValueError("A file is required to build this file.")
        elif open_ or load or build:
            self.open(**kwargs)

        self.construct_group(load=load, build=build)

        if not open_:
            self.close()

        return self

    def construct_file_attributes(self, map_: HDF5Map = None, load: bool = False, build: bool = False) -> None:
        """Creates the attributes for this group.

        Args:
            map_: The map to use to create the attributes.
            load: Determines if this object will load the attribute values from the file on construction.
            build: Determines if this object will create and fill the attributes in the file on construction.
        """
        self._group.construct_attributes(map_=map_, load=load, build=build)

    def construct_group(self, map_: HDF5Map = None, load: bool = False, build: bool = False):
        """Creates the group object for this file.

        Args:
            map_: The map to use to create the attributes.
            load: Determines if this object will recursively load the members from the file on construction.
            build: Determines if this object will recursively create and fill the members in the file on construction.
        """
        if map_ is not None:
            self.map = map_
        self._group = self.group_type(name=self._name_, map_=self.map, file=self, load=load, build=build)

    # Getters/Setters
    @singlekwargdispatchmethod("file")
    def _set_path(self, file: str | pathlib.Path | h5py.File | "HDF5File") -> None:
        """Sets the path for the file.

        Args:
            file: The path or the file object to set the path to.
        """
        if isinstance(file, HDF5File):
            self.path = file.path
        else:
            raise ValueError(f"{type(file)} is not a valid type for _set_path.")

    @_set_path.register(str)
    @_set_path.register(pathlib.Path)
    def _(self, file: str | pathlib.Path) -> None:
        """Sets the path for the file.

        Args:
            file: The path to the file to build this object around.
        """
        self.path = file

    @_set_path.register
    def _(self, file: h5py.File) -> None:
        """Sets the path for the file.

        Args:
            file: A HDF5 file to build this object around.
        """
        if file:
            self._file = file
            self._path = pathlib.Path(file.filename)
        else:
            raise ValueError("The supplied HDF5 File must be open.")

    # File Creation/Construction
    def create_file(
        self,
        name: str | pathlib.Path = None,
        open_: bool = True,
        map_: HDF5Map = None,
        build: bool = False,
        **kwargs: Any
    ) -> "HDF5File":
        """Creates the HDF5 file.

        Args:
            name: The file name as path.
            open_: Determines if this object will remain open after creation.
            map_: The map for this HDF5 object.
            build: Determines if the values of this file will be filled.
            **kwargs: The keyword arguments for the open method.

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

    def require_file(
        self,
        name: str | pathlib.Path = None,
        open_: bool = True,
        map_: HDF5Map = None,
        load: bool = False,
        build: bool = False,
        **kwargs: Any
    ) -> "HDF5File":
        """Creates the HDF5 file or loads it if it exists.

        Args:
            name: The file name as path.
            open_: Determines if this object will remain open after creation.
            map_: The map for this HDF5 object.
            load: Determines if the values of this file will be loaded.
            build: Determines if the values of this file will be filled.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if name is not None:
            self.path = name

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
    def open(self, mode: str = 'a', exc: bool = False, **kwargs: Any) -> "HDF5File":
        """Opens the HDF5 file.

        Args:
            mode: The mode which this file should be opened in.
            exc: Determines if an error should be excepted as warning or not.
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
    def temp_open(self, **kwargs: Any) -> "HDF5File":
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

    def close(self) -> bool:
        """Closes the HDF5 file.

        Returns:
            If the file was successfully closed.
        """
        if self.is_open:
            self._file.flush()
            self._file.close()
        return not self.is_open
