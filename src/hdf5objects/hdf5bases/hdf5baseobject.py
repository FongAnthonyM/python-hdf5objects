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
from collections.abc import Mapping
import pathlib
from types import MethodType
from typing import Any
import weakref

# Third-Party Packages #
from baseobjects import BaseComposite, singlekwargdispatchmethod, search_sentinel
from baseobjects.cachingtools import CachingInitMeta, CachingObject
from baseobjects.wrappers import StaticWrapper
from baseobjects.typing import AnyCallable
import h5py

# Local Packages #
from .hdf5caster import HDF5Caster
from .hdf5basecomponent import HDF5BaseComponent
from .hdf5map import HDF5Map


# Definitions #
# Classes #
class HDF5BaseObject(StaticWrapper, CachingObject, BaseComposite, metaclass=CachingInitMeta):
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
        _mode_: The edit mode of this object.
        map: The map of this HDF5 object.
        components: The components of this composite object.

    Args:
        name: The HDF5 name of this object.
        map_: The map for this HDF5 object.
        mode: The edit mode of this object.
        file: The file object that this HDF5 object originates from.
        parent: The HDF5 name of the parent of this HDF5 object.
        init: Determines if this object will construct.
    """
    sentinel: Any = search_sentinel
    file_type: type | None = None
    default_map: HDF5Map | None = None
    write_modes: set[str] = {"a", "r+"}

    # Class Methods #
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

    # Magic Methods #
    # Constructors/Destructors
    def __init__(
        self,
        name: str | None = None,
        map_: HDF5Map | None = None,
        mode: str | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        parent: str | None = None,
        init: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._file_was_open: bool | None = None
        self._weak_signal: weakref.ref | None = None
        self._weak_file: weakref.ref | None = None
        self._file: h5py.File | "HDF5File" | None = None
        self._get_file: AnyCallable = self._get_weak_file.__func__

        self._name_: str | None = None
        self._parents: list[str] | None = None

        self._mode_: str | None = None

        self.map: HDF5Map = self.default_map.deepcopy()

        # Parent Attributes #
        super().__init__(init=init)

        # Object Construction #
        if init:
            self.construct(name=name, map_=map_, mode=mode, file=file, parent=parent, **kwargs)

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
            return f"{''.join(f'/{p}' for p in self._parents)}/{'' if self._name is None else self._name_}"

    @property
    def _mode(self) -> str:
        """The edit mode of this object"""
        return self._mode_

    @_mode.setter
    def _mode(self, value: str) -> None:
        self.set_mode(mode=value)

    @property
    def exists(self) -> bool:
        """Checks if this object exists in the hdf5 file."""
        return self.is_exist()

    @property
    def file(self) -> Any:
        """Returns the owning file of this HDF5 Object"""
        return self.get_file()

    @property
    def get_file(self) -> AnyCallable:
        """A descriptor to create the bound get file method."""
        return self._get_file.__get__(self, self.__class__)

    @get_file.setter
    def get_file(self, value: AnyCallable) -> None:
        self._get_file = value

    @property
    def caster(self) -> HDF5Caster:
        """An object that can cast python objects to and from HDF5 data types."""
        return self.map.caster

    @caster.setter
    def caster(self, value: Any) -> None:
        self.map.caster = value

    @property
    def kwargs(self) -> dict:
        """The kwargs to use when creating this object in the HDF5 file."""
        return self.map.kwargs

    @kwargs.setter
    def kwargs(self, value: dict) -> None:
        self.map.kwargs = value

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            dict: A dictionary of this object's attributes.
        """
        state = self.__dict__.copy()
        
        weak_file = state.pop("_weak_file")
        if weak_file is not None:
            state["file"] = weak_file()
        else:
            state["file"] = None

        weak_signal = state.pop("_weak_signal")
        if weak_signal is not None:
            state["signal"] = weak_signal()
        else:
            state["signal"] = None
        
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Builds this object based on a dictionary of corresponding attributes.

        Args:
            state: The attributes to build this object from.
        """
        file = state.pop("file")
        if file is not None:
            state["_weak_file"] = weakref.ref(file)
        else:
            state["_weak_file"] = None

        signal = state.pop("signal")
        if signal is not None:
            state["_weak_signal"] = weakref.ref(signal)
        else:
            state["_weak_signal"] = None
        
        self.__dict__.update(state)
    
    # Container Methods
    def __getitem__(self, key: Any) -> Any:
        """Ensures HDF5 object is open for getitem"""
        with self:
            return getattr(self, self._wrap_attributes[0])[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Ensures HDF5 object is open for setitem"""
        with self:
            getattr(self, self._wrap_attributes[0])[key] = value

    def __delitem__(self, key: Any) -> None:
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

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        name: str | None = None,
        map_: HDF5Map | None = None,
        mode: str | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        parent: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            name: The HDF5 name of this object.
            map_: The map for this HDF5 object.
            mode: The edit mode of this object.
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

        if mode is not None:
            self.set_mode(mode, timed=False)

        if file is not None:
            self.set_file(file)
            if mode is None and self._mode_ is None:
                self.set_mode(self.file._mode, timed=False)
                
        super().construct(**kwargs)

    def construct_components(
        self,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
    ) -> None:
        """Constructs or adds components.

        Args:
            component_kwargs: The keyword arguments for creating the components.
            component_types: Component class and their keyword arguments to instantiate.
            components: Components to add.
        """
        temp_types = self.default_component_types | self.map.component_types | component_types
        new_kwargs = {} if component_kwargs is None else component_kwargs
        default_components = {n: c(composite=self, **(k | new_kwargs.get(n, {}))) for n, (c, k) in temp_types.items()}
        self.components.update(default_components | self.components | {} if components is None else components)

    def is_exist(self) -> bool:
        """Determine if this object exists in the HDF5 file."""
        with self.file.temp_open():
            try:
                self.file._file[self._full_name]
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
        self._file_was_open = self.file.is_open
        if not getattr(self, self._wrap_attributes[0]):
            self.file.open(mode=mode, **kwargs)
            setattr(self, self._wrap_attributes[0], self.file._file[self._full_name])

        return self

    def close(self) -> None:
        """Closes the file of this dataset."""
        if not self._file_was_open:
            self.file.close()

    # Getters/Setters
    @singlekwargdispatchmethod("file")
    def set_file(self, file: str | pathlib.Path | h5py.File) -> None:
        """Sets the file for this object to an HDF5File.

        Args:
            file: An object to set the file to.
        """
        if isinstance(file, self.file_type):
            self._weak_file = weakref.ref(file)
            self.get_file = self._get_weak_file.__func__
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
        self._weak_signal = weakref.ref(file)
        self._file = self.file_type(file)
        self.get_file = self._get_weak_file_indirect.__func__

    def _get_weak_file(self):
        """Returns the owning file of this HDF5 Object using a weak reference."""
        try:
            return self._weak_file()
        except TypeError:
            return None

    def _get_weak_file_indirect(self):
        """Returns the owning file of this HDF5 Object using a weak reference as signal."""
        try:
            if self._weak_signal() is None:
                self._file = None
            return self._file
        except TypeError:
            return None

    def _get_file_direct(self):
        """Returns the owning file of this HDF5 Object."""
        return self._file

    def set_parent(self, parent: str) -> None:
        """Sets the parent of this object to the str

        Args:
            parent: The str to parse and set as the parent of this map.
        """
        parent = parent.lstrip('/')
        parts = parent.split('/')
        self._parents = parts

    def set_name(self, name: str | None) -> None:
        """Sets the name of this object, can be a full hdf5 name.

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

    def set_mode(self, mode: str, timed: bool = False, **kwargs: Any) -> None:
        """Sets the edit mode of this object.

        Args:
            mode: The string edit mode to set this object to.
            timed: Determines if the caches will have an expiration or not.
            **kwargs: The keyword arguments for the enable/disable caching methods.
        """
        self._mode_ = mode
        if mode in self.write_modes:
            self.disable_caching(**kwargs)
        else:
            self.enable_caching(**kwargs)

        if timed:
            self.timed_caching()
        else:
            self.timeless_caching()

    def set_map(self, map_: HDF5Map) -> None:
        """Changes the current map with a different one.

        Args:
            map_: The map to replace the current map.
        """
        self.map_ = map_

    def create_components(self, **component_kwargs: dict[str, Any]) -> None:
        """Creates the components of this HDF5 object.

        Args:
            **component_kwargs: The keyword arguments for the components' create methods as keywords.
        """
        for name, component in self.components.items():
            kwargs = component_kwargs.get(name, {})
            component.create_component(**kwargs)

    def require_components(self, **component_kwargs: dict[str, Any]) -> None:
        """Requires the components of this HDF5 object.

        Args:
            **component_kwargs: The keyword arguments for the components' require methods as keywords.
        """
        for name, component in self.components.items():
            kwargs = component_kwargs.get(name, {})
            component.require_component(**kwargs)

    def standardize_attributes(self) -> None:
        """Sets attributes that correspond to values somewhere else to current value."""
        pass
