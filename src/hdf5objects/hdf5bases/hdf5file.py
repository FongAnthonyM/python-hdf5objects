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
        _is_open:
        _path: The path to the file.
        _name_: The name of the first layer in the file.
        allow_swmr_create: Determines if creating a dataset during swmr is allowed, forces close open if allowed.
        _group: The first layer group this object will wrap.

    Args:
        file: Either the file object or the path to the file.
        mode: The edit mode of this object.
        open_: Determines if this object will remain open after construction.
        map_: The map for this HDF5 object.
        load: Determines if this object will load the file on construction.
        create: Determines if this object will create an empty file on construction.
        construct: Determines if this object will create members int the file on construction.
        require: Determines if this object will create and fill the file on construction.
        path: The path to the file.
        component_kwargs: The keyword arguments for the components.
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
            return getattr(getattr(obj, wrap_name), attr_name)

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
            setattr(getattr(obj, wrap_name), attr_name, value)

    @classmethod
    def _del_attribute(cls, obj: Any, wrap_name: str, attr_name: str) -> None:
        """Deletes an attribute in a wrapped HDF5 object.

        Args:
            obj: The target object to set.
            wrap_name: The attribute name of the wrapped object.
            attr_name: The attribute name of the attribute to delete from the wrapped object.
        """
        with obj.temp_open():  # Ensures the hdf5 dataset is open when accessing attributes
            delattr(getattr(obj, wrap_name), attr_name)

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
        mode: str = 'r',
        open_: bool = True,
        map_: HDF5Map | None = None,
        load: bool = False,
        create: bool = False,
        construct: bool = False,
        require: bool = False,
        path: str | pathlib.Path | h5py.File | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._is_open: bool = False

        self._path: pathlib.Path | None = None
        self._name_: str = "/"
        self.allow_swmr_create: bool = False

        self._group: HDF5Group | None = None

        # Parent Attributes #
        super().__init__(init=False)

        # Override Attributes #
        self._mode_: str = 'r'

        # Object Construction #
        if init:
            self.construct(
                file=file,
                mode=mode,
                open_=open_,
                map_=map_,
                load=load,
                create=create,
                construct=construct,
                require=require,
                path=path,
                component_kwargs=component_kwargs,
                **kwargs,
            )

    @property
    def path(self) -> pathlib.Path:
        """The path to the file. The setter casts file objects that are not Path to path before setting"""
        return self._path

    @path.setter
    def path(self, value: str | pathlib.Path) -> None:
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

    @property
    def swmr_mode(self) -> bool:
        """The Single Write Multiple Read of this file. If set to True it also disables caching."""
        return self._file.swmr_mode

    @swmr_mode.setter
    def swmr_mode(self, value: bool) -> None:
        self._file.swmr_mode = value
        if value:
            self._group.disable_all_caching()

    def __del__(self) -> None:
        """Closes the file when this object is deleted."""
        self.close()

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            dict: A dictionary of this object's attributes.
        """
        state = super().__getstate__()
        state["open_state"] = self.is_open
        del state["_file"]
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Builds this object based on a dictionary of corresponding attributes.

        Args:
            state: The attributes to build this object from.
        """
        was_open = state.pop("open_state")
        super().__setstate__(state=state)
        if was_open:
            self.open()

    # Container Methods
    def __getitem__(self, key: str | h5py.Reference) -> HDF5BaseObject:
        """Gets a HDF5 object from the HDF5 file.

        Args:
           key: The name of the HDF5 object to get.

        Returns:
            The HDF5 object requested.
        """
        if isinstance(key, h5py.Reference):
            with self.temp_open():
                key = self._file[key].name
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
        mode: str | None = None,
        open_: bool = True,
        map_: HDF5Map | None = None,
        load: bool = False,
        create: bool = False,
        construct: bool = False,
        require: bool = False,
        path: str | pathlib.Path | h5py.File | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "HDF5File":
        """Constructs this object.

        Args:
            file: Either the file object or the path to the file.
            mode: The edit mode of this object.
            open_: Determines if this object will remain open after construction.
            map_: The map for this HDF5 object.
            load: Determines if this object will load the file on construction.
            create: Determines if this object will create an empty file on construction.
            construct: Determines if this object will create members int the file on construction.
            require: Determines if this object fill the file on construction.
            path: The path to the file.
            component_kwargs: The keyword arguments for the components.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if map_ is not None:
            self.map = map_

        if self.map.name is None:
            self.map.name = "/"

        if mode is not None:
            self.set_mode(mode, timed=False)

        if path is not None:
            self._set_path(path)

        if file is not None:
            self._set_path(file)

        if self.path is not None and not self.path.is_file():
            if create:
                self.require_file(open_=True, **kwargs)
            elif load:
                raise ValueError("A file is required to load this file.")
            elif require:
                raise ValueError("A file is required to fill this file.")
        elif open_ or load or require:
            self.open(**kwargs)

        if component_kwargs is None:
            self.construct_components()
        else:
            self.construct_components(**component_kwargs)

        self.construct_group(load=load, construct=construct, require=require)

        if construct or require:
            self.construct_file_attributes(require=require)

        if not open_:
            self.close()
        elif self._file.swmr_mode:
            self._group.enable_caching()

        return self

    def construct_file_attributes(self, map_: HDF5Map = None, load: bool = False, require: bool = False) -> None:
        """Creates the attributes for this group.

        Args:
            map_: The map to use to create the attributes.
            load: Determines if this object will load the attribute values from the file on construction.
            require: Determines if this object will create and fill the attributes in the file on construction.
        """
        self._group.construct_attributes(map_=map_, load=load, require=require)

    def construct_group(
        self,
        map_: HDF5Map = None,
        load: bool = False,
        construct: bool = False,
        require: bool = False,
    ) -> None:
        """Creates the group object for this file.

        Args:
            map_: The map to use to create the attributes.
            load: Determines if this object will recursively load the members from the file on construction.
            construct: Determines if this object will create members int the file on construction.
            require: Determines if this object will recursively create and fill the members in the file on construction.
        """
        if map_ is not None:
            self.map = map_
        self._group = self.group_type(
            name=self._name_,
            map_=self.map,
            file=self,
            load=load,
            construct=construct,
            require=require,
        )

    # Getters/Setters
    def set_map(self, map_: HDF5Map) -> None:
        """Changes the current map with a different one.

        Args:
            map_: The map to replace the current map.
        """
        self.map_ = map_
        if self._group is not None:
            self._group.set_map(map_)

    @singlekwargdispatchmethod("file")
    def _set_path(self, file: str | pathlib.Path | h5py.File) -> None:
        """Sets the path for the file.

        Args:
            file: The path or the file object to set the path to.
        """
        if isinstance(file, HDF5File):
            self.path = file.path
        else:
            raise TypeError(f"{type(file)} is not a valid type for _set_path.")

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
            self._path = pathlib.Path(file.filename)
            self._file = file
            self._get_file = self._get_file_direct.__func__
        else:
            raise ValueError("The supplied HDF5 File must be open.")

    # File Creation/Construction
    def create_file(
        self,
        name: str | pathlib.Path = None,
        open_: bool = True,
        map_: HDF5Map = None,
        require: bool = False,
        **kwargs: Any
    ) -> "HDF5File":
        """Creates the HDF5 file.

        Args:
            name: The file name as path.
            open_: Determines if this object will remain open after creation.
            map_: The map for this HDF5 object.
            require: Determines if the values of this file will be filled.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if name is not None:
            self.path = name

        if map_ is not None:
            self.map = map_

        self.open(**kwargs)
        if require:
            self._group.construct_members(require=True)
        elif not open_:
            self.close()

        return self

    def create(
        self,
        name: str | pathlib.Path = None,
        open_: bool = True,
        map_: HDF5Map = None,
        load: bool = False,
        require: bool = False,
        component_kwargs: dict[str, Any] = {},
        **kwargs: Any
    ) -> "HDF5File":
        """Creates the HDF5 file.

        Args:
            name: The file name as path.
            open_: Determines if this object will remain open after creation.
            map_: The map for this HDF5 object.
            load: Determines if the values of this file will be loaded.
            require: Determines if the values of this file will be filled.
            component_kwargs: The keyword arguments for the components' create methods.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        self.create_file(name=name, open_=open_, map_=map_, load=load, require=require, **kwargs)
        self.create_components(**component_kwargs)
        return self

    def require_file(
        self,
        name: str | pathlib.Path = None,
        open_: bool = True,
        map_: HDF5Map = None,
        load: bool = False,
        require: bool = False,
        **kwargs: Any
    ) -> "HDF5File":
        """Creates the HDF5 file or loads it if it exists.

        Args:
            name: The file name as path.
            open_: Determines if this object will remain open after creation.
            map_: The map for this HDF5 object.
            load: Determines if the values of this file will be loaded.
            require: Determines if the values of this file will be filled.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if name is not None:
            self.path = name

        if self.path.is_file():
            self.open(**kwargs)
            if load:
                self._group.load(load=True, require=require)
            if not open_:
                self.close()
        else:
            self.create_file(open_=open_, map_=map_, require=require, **kwargs)

        return self

    def require(
        self,
        name: str | pathlib.Path = None,
        open_: bool = True,
        map_: HDF5Map = None,
        load: bool = False,
        require: bool = False,
        component_kwargs: dict[str, Any] = {},
        **kwargs: Any
    ) -> "HDF5File":
        """Creates the HDF5 file or loads it if it exists.

        Args:
            name: The file name as path.
            open_: Determines if this object will remain open after creation.
            map_: The map for this HDF5 object.
            load: Determines if the values of this file will be loaded.
            require: Determines if the values of this file will be filled.
            component_kwargs: The keyword arguments for the components' create methods.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        self.require_file(name=name, open_=open_, map_=map_, load=load, require=require, **kwargs)
        self.require_components(**component_kwargs)
        return self

    # def copy_file(self, path):  # Todo: Implement this.
    #     pass

    # File
    def open(self, mode: str | None = None, exc: bool = False, **kwargs: Any) -> "HDF5File":
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
                if mode is not None:
                    self._mode = mode
                if "libver" not in kwargs:
                    kwargs["libver"] = "latest"
                self._file = h5py.File(self.path.as_posix(), mode=self._mode_, **kwargs)
                self._get_file = self._get_file_direct.__func__
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
        was_open = self.is_open or self._is_open
        if not was_open:
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

    # Caching
    def clear_all_caches(self, **kwargs: Any) -> None:
        """Clears all caches in this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the clear caches method.
        """
        self.clear_caches(**kwargs)
        self._group.clear_all_caches(**kwargs)

    def enable_all_caching(self, **kwargs: Any) -> None:
        """Enables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the enable caching method.
        """
        self.enable_caching(**kwargs)
        self._group.enable_caching(**kwargs)

    def disable_all_caching(self, **kwargs: Any) -> None:
        """Disables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the disable caching method.
        """
        self.disable_caching(**kwargs)
        self._group.disable_caching(**kwargs)

    def timeless_all_caching(self, **kwargs: Any) -> None:
        """Allows timeless caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timeless caching method.
        """
        self.timeless_caching(**kwargs)
        self._group.timeless_caching(**kwargs)

    def timed_all_caching(self, **kwargs: Any) -> None:
        """Allows timed caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timed caching method.
        """
        self.timed_caching(**kwargs)
        self._group.timed_caching(**kwargs)

    def set_all_lifetimes(self, lifetime: int | float | None, **kwargs: Any) -> None:
        """Sets the lifetimes on this object and all contained objects.

        Args:
            lifetime: The lifetime to set all the caches to.
            **kwargs: The keyword arguments for the lifetime caching method.
        """
        self.set_lifetimes(lifetime=lifetime, **kwargs)
        self._group.set_lifetimes(lifetime=lifetime, **kwargs)
