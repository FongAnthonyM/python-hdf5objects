"""hdf5group.py
An object that represents an HDF5 Group.
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
from collections.abc import Iterable, Mapping, KeysView, ItemsView, ValuesView
import pathlib
from typing import Any

# Third-Party Packages #
from baseobjects import search_sentinel
from baseobjects.functions import singlekwargdispatch
import h5py
import numpy as np

# Local Packages #
from .hdf5map import HDF5Map
from .hdf5baseobject import HDF5BaseObject
from .hdf5attributes import HDF5Attributes
from .hdf5dataset import DatasetMap, HDF5Dataset

# Definitions #
# Names #
search_sentinel_2 = object()


# Classes #
class GroupMap(HDF5Map):
    """A general map for HDF5 Groups.

    Class Attributes:
        default_attributes_type: The default type for attribute objects in this map.
    """

    default_attributes_type: type = HDF5Attributes
    # default_type: type = HDF5Group  # This will be assigned after HDF5Group is defined


class HDF5Group(HDF5BaseObject):
    """A wrapper object which wraps a HDF5 group and gives more functionality.

    Class Attributes:
        _wrapped_types: A list of either types or objects to set up wrapping for.
        _wrap_attributes: Attribute names that will contain the objects to wrap where the resolution order is descending
            inheritance.
        default_group_map: The default group type to use when creating groups.
        default_dataset_map: The default dataset type to use when creating dataset.

    Attributes:
        _group: The HDF5 group to wrap.
        attributes: The attributes of this group.
        members: The HDF5 objects within this group.

    Args:
        group: The HDF5 group to build this dataset around.
        name: The HDF5 name of this object.
        map_: The map for this HDF5 object.
        mode: The edit mode of this object.
        file: The file object that this group object originates from.
        load: Determines if this object will load the group from the file on construction.
        require: Determines if this object will be created on construction.
        construct: Determines if this object will create its members recursively on construction.
        parent: The HDF5 name of the parent of this HDF5 object.
        component_kwargs: The keyword arguments for creating the components.
        component_types: Component class and their keyword arguments to instantiate.
        components: Components to add.
        track_order: Track dataset/group/attribute creation order under this group if True. If None use global
                default h5.get_config().track_order.
        init: Determines if this object will construct.
    """

    _wrapped_types: list[type | object] = [h5py.Group]
    _wrap_attributes: list[str] = ["group"]
    # default_map: HDF5Map = GroupMap()  # This will be assigned after HDF5Group is defined
    default_group_map: type = None  # This will be assigned after HDF5Group is defined
    default_dataset_map: type = DatasetMap

    # Magic Methods #
    # Constructors/Destructors
    def __init__(
        self,
        group: h5py.Group | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        mode: str | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        require: bool = False,
        construct: bool = False,
        parent: str | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        track_order: bool | None = None,
        init: bool = True,
    ) -> None:
        # New Attributes #
        self._group: h5py.Group | None = None
        self.attributes: HDF5Attributes | None = None

        self.members: dict[str, HDF5BaseObject] = {}

        # Parent Attributes #
        super().__init__(file=file, init=False)

        # Object Construction #
        if init:
            self.construct(
                group=group,
                name=name,
                map_=map_,
                mode=mode,
                file=file,
                load=load,
                require=require,
                construct=construct,
                parent=parent,
                component_kwargs=component_kwargs,
                component_types=component_types,
                components=components,
                track_order=track_order
            )

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            dict: A dictionary of this object's attributes.
        """
        state = super().__getstate__()
        state["_group"] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Builds this object based on a dictionary of corresponding attributes.

        Args:
            state: The attributes to build this object from.
        """
        super().__setstate__(state=state)
        with self.file.temp_open:
            self._group = self.file._file[self._full_name]

    # Container Methods
    def __getitem__(self, key: str) -> HDF5BaseObject:
        """Gets an item within this group."""
        return self.get(key)

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        group: h5py.Group | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        mode: str | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        require: bool = False,
        construct: bool = False,
        parent: str | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        track_order: bool | None = None,
    ) -> None:
        """Constructs this object from the provided arguments.

        Args:
            group: The HDF5 group to build this dataset around.
            name: The HDF5 name of this object.
            map_: The map for this HDF5 object.
            mode: The edit mode of this object.
            file: The file object that this group object originates from.
            load: Determines if this object will load the group from the file on construction.
            require: Determines if this object will be created on construction.
            construct: Determines if this object will create its members in the file on construction.
            parent: The HDF5 name of the parent of this HDF5 object.
            component_kwargs: The keyword arguments for creating the components.
            component_types: Component class and their keyword arguments to instantiate.
            components: Components to add.
            track_order: Track dataset/group/attribute creation order under this group if True. If None use global
                default h5.get_config().track_order.
        """
        if map_ is not None:
            self.map = map_

        if self.map.name is None:
            self.map.name = "/"

        if self.map.type is None:
            self.map.type = self.default_group_map

        if parent is not None:
            self.set_parent(parent=parent)
        elif map_ is not None:
            self._parents = self.map.parents

        if name is not None:
            self.set_name(name=name)
        elif map_ is not None:
            self._name_ = self.map.name

        if mode is not None:
            self.set_mode(mode)

        if file is not None:
            self.set_file(file)
            if mode is None and self._mode_ is None:
                self.set_mode(self.file._mode)

        if self.map.name is None:
            self.map.name = "/"

        if self.map.type is None:
            self.map.type = self.default_group_map

        if group is not None:
            self.set_group(group)

        self.construct_attributes()

        super().construct(
            component_kwargs=component_kwargs,
            component_types=component_types,
            components=components,
        )

        if load and self.exists:
            self.load(load=load)

        if require or construct:
            self.require(load=load, construct=construct, track_order=track_order)

    def construct_attributes(self, map_: HDF5Map = None, load: bool = False, require: bool = False) -> None:
        """Creates the attributes for this group.

        Args:
            map_: The map to use to create the attributes.
            load: Determines if this object will load the attribute values from the file on construction.
            require: Determines if this object will create and fill the attributes in the file on construction.
        """
        if map_ is None:
            map_ = self.map
        self.attributes = map_.attributes_type(
            name=self._full_name, map_=map_, file=self.file, load=load, require=require
        )

    def construct_member(
        self,
        name: str,
        map_: HDF5Map = None,
        load: bool = False,
        require: bool = False,
        construct: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Constructs a member of this group.

        Args:
            name: The name of the member to construct.
            map_: The map to use to create the members.
            load: Determines if this object will recursively load the members from the file on construction.
            require: Determines if this object will be created on construction.
            construct: Determines if this object will create its members recursively on construction.
            **kwargs: The keyword arguments to construct the member.
        """
        if map_ is not None:
            self.map = map_

        name = self._parse_name(name)
        member = self.members.get(name, search_sentinel)
        if member is search_sentinel:
            value = self.map[name]
            member = value.get_object(load=load, require=require, construct=construct, file=self.file, **kwargs)
            if member is None:
                self.members[name] = member = value.get_object(load=load, file=self.file, **kwargs)
            else:
                self.members[name] = member

        return member

    def construct_members(
        self,
        map_: HDF5Map = None,
        load: bool = False,
        require: bool = False,
        construct: bool = False,
    ) -> None:
        """Creates the members of this group.

        Args:
            map_: The map to use to create the members.
            load: Determines if this object will recursively load the members from the file on construction.
            require: Determines if this object will be created on construction.
            construct: Determines if this object will create its members recursively on construction.
        """
        if map_ is not None:
            self.map = map_

        for name, value in self.map.items():
            name = self._parse_name(name)
            if name not in self.members:
                obj = value.get_object(load=load, require=require, construct=construct, file=self.file)
                if obj is None:
                    self.members[name] = value.get_object(load=load, file=self.file)
                else:
                    self.members[name] = obj
    
    def require_composite(
        self,
        name: str | None = None,
        load: bool = False,
        construct: bool = False,
        track_order: bool | None = None,
    ) -> "HDF5Group":
        """Creates this group if it does not exist.

        Args:
            name: The name of this group.
            load: Determines if this object will load the contents of the group.
            construct: Determines if this object will create its members recursively on construction.
            track_order: Track dataset/group/attribute creation order under this group if True. If None use global
                default h5.get_config().track_order.

        Returns:
            This object.
        """
        if name is not None:
            self._name = name

        with self.file.temp_open():
            if not self.exists:
                self._group = self.file._file.create_group(name=self._full_name, track_order=track_order)
                self.attributes.construct_attributes()

        if construct:
            self.construct_members(load=load, require=construct, construct=construct)

        return self   
                    
    def require(
        self,
        name: str | None = None,
        load: bool = False,
        construct: bool = False,
        track_order: bool | None = None,
        component_kwargs: dict[str, Any] = {},
    ) -> "HDF5Group":
        """Creates this group if it does not exist.

        Args:
            name: The name of this group.
            load: Determines if this object will load the contents of the group.
            construct: Determines if this object will create its members recursively on construction.
            track_order: Track dataset/group/attribute creation order under this group if True. If None use global
                default h5.get_config().track_order.
            component_kwargs: The keyword arguments for the components' create methods.

        Returns:
            This group.
        """
        self.require_composite(name=name, load=load, construct=construct, track_order=track_order)
        self.require_components(**component_kwargs)

        return self

    # Parsers
    def _parse_name(self, name: str) -> str:
        """Returns the hdf5 name of a member of this group.

        Args:
            name: Either the python name of the member or the hdf5 name.

        Returns:
            The hdf5 name of the member.
        """
        new_name = self.map.map_names.get(name, search_sentinel)
        if new_name is not search_sentinel:
            name = new_name
        return name

    # File
    def load(self, load: bool = False) -> None:
        """Loads this group from file with the option to create and fill it.

        Args:
            load: Determines if this object will recursively load the members from the file on construction.
        """
        self.attributes.load()
        self.get_members(load=load)

    def refresh(self) -> None:
        """Reloads the attributes and members from the file."""
        self.attributes.refresh()
        self.get_members()

    # Caching
    def clear_all_caches(self, **kwargs: Any) -> None:
        """Clears all caches in this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the clear caches method.
        """
        self.attributes.clear_caches(**kwargs)
        self.clear_caches(**kwargs)
        for member in self.members.values():
            member.clear_all_caches(**kwargs)

    def enable_all_caching(self, **kwargs: Any) -> None:
        """Enables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the enable caching method.
        """
        self.attributes.enable_caching(**kwargs)
        self.enable_caching(**kwargs)
        for member in self.members.values():
            member.enable_all_caching(**kwargs)

    def disable_all_caching(self, **kwargs: Any) -> None:
        """Disables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the disable caching method.
        """
        self.attributes.disable_caching(**kwargs)
        self.disable_caching(**kwargs)
        for member in self.members.values():
            member.disable_all_caching(**kwargs)

    def timeless_all_caching(self, **kwargs: Any) -> None:
        """Allows timeless caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timeless caching method.
        """
        self.attributes.timeless_caching(**kwargs)
        self.timeless_caching(**kwargs)
        for member in self.members.values():
            member.timeless_all_caching(**kwargs)

    def timed_all_caching(self, **kwargs: Any) -> None:
        """Allows timed caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timed caching method.
        """
        self.attributes.timed_caching(**kwargs)
        self.timed_caching(**kwargs)
        for member in self.members.values():
            member.timed_all_caching(**kwargs)

    def set_all_lifetimes(self, lifetime: int | float | None, **kwargs: Any) -> None:
        """Sets the lifetimes on this object and all contained objects.

        Args:
            lifetime: The lifetime to set all the caches to.
            **kwargs: The keyword arguments for the lifetime caching method.
        """
        self.attributes.set_lifetimes(lifetime=lifetime, **kwargs)
        self.set_lifetimes(lifetime=lifetime, **kwargs)
        for member in self.members.values():
            member.set_lifetimes(lifetime=lifetime, **kwargs)

    # Getters/Setters
    def set_map(self, map_: HDF5Map) -> None:
        """Changes the current map with a different one.

        Args:
            map_: The map to replace the current map.
        """
        super().set_map(map_=map_)
        map_.object = self
        if map_.name is None:
            map_.set_name(self._full_name)
        if self._name != "/":
            self.file[self._parent].map[self._name] = map_
        self.attributes.set_map(map_)
        self.members.clear()

    @singlekwargdispatch("group")
    def set_group(self, group: "HDF5Group") -> None:
        """Sets the wrapped group.

        Args:
            group: The group this object will wrap.
        """
        if isinstance(group, HDF5Group):
            if self.file is None:
                self.set_file(group.file)
            self.set_name(group._name)
            self._group = group._group
        else:
            raise TypeError(f"{type(group)} is not a valid type for set_group.")

    @set_group.register
    def _(self, group: h5py.Group) -> None:
        """Sets the wrapped group.

        Args:
            group: The group this object will wrap.
        """
        if not group:
            raise ValueError("Group needs to be open")
        if self.file is None:
            self.set_file(group.file)
        self.set_name(group.name)
        self._group = group

    def get_member(
        self,
        name: str,
        default: Any = search_sentinel,
        load: bool = False,
        mapped: bool = True,
        **kwargs: Any,
    ) -> HDF5BaseObject:
        """Get a member of the group.

        Args:
            name: The name of the member to get.
            default: Returns this value if the member is not in this group.
            load: Determines if the member will recursively load the members from the file on construction.
            mapped: Determines if the member will have a default assigned to it if no map is found.
            **kwargs: Extra kwargs to use to create the member.

        Returns:
            The requested member.
        """
        name = self._parse_name(name)
        
        member = self.members.get(name, None)
        if member is None:
            with self:
                if default is not search_sentinel and name not in self._group:
                    return default
                item = self._group[name]
                map_ = self.map.get_item(name, search_sentinel)
                if map_ is search_sentinel:
                    map_namespace = item.attrs.get("map_namespace", "")
                    map_name = item.attrs.get("map_type", "")
                    map_type = self.map.map_registry.get(map_namespace, {}).get(map_name, None)
    
                    if map_type is not None:
                        map_ = map_type(name=name)
                        self.map.set_item(map_)
                    elif not mapped:
                        if isinstance(item, h5py.Dataset):
                            map_ = self.default_dataset_map()
                        elif isinstance(item, h5py.Group):
                            map_ = self.default_group_map()
    
                if map_ is not search_sentinel:
                    if isinstance(item, h5py.Group):
                        kwargs["group"] = item
                    else:
                        kwargs["dataset"] = item
                    self.members[name] = member = map_.get_object(
                        map_=map_,
                        file=self.file,
                        load=load,
                        **kwargs,
                    )
                else:
                    member = item
        return member

    def get_members(self, load: bool = False, mapped: bool = True) -> dict[str, HDF5BaseObject]:
        """Get all the members in this group.

        Args:
            load: Determines if this object will recursively load the members from the file on construction.
            mapped: Determines if members will have a default assigned to it if no map is found.

        Returns:
            The names and members in this group.
        """
        with self:
            for name, item in self._group.items():
                map_ = self.map.get_item(name, search_sentinel)
                if map_ is search_sentinel:
                    map_namespace = item.attrs.get("map_namespace", "")
                    map_name = item.attrs.get("map_type", "")
                    map_type = self.map.map_registry.get(map_namespace, {}).get(map_name, None)

                    if map_type is not None:
                        map_ = map_type(name=name)
                        self.map.set_item(map_)
                    elif not mapped:
                        if isinstance(item, h5py.Dataset):
                            map_ = self.default_dataset_map()
                        elif isinstance(item, h5py.Group):
                            map_ = self.default_group_map()

                if map_ is not search_sentinel:
                    if isinstance(item, h5py.Group):
                        kwargs = {"group": item}
                    else:
                        kwargs = {"dataset": item}
                    self.members[name] = map_.get_object(
                        map_=map_,
                        file=self.file,
                        load=load,
                        **kwargs,
                    )

        return self.members.copy()

    def get(self, key: str | Iterable[str], default: Any = search_sentinel) -> HDF5BaseObject:
        """Get a member of this group.

        Args:
            key: The key name of the member to get.
            default: An object to return if the key cannot be found.

        Returns:
            The requested member.
        """
        keys = key.strip("/").split("/") if isinstance(key, str) else list(key)
        key = keys.pop(0)
        key = self._parse_name(key)

        item = self.get_member(name=key, default=default)

        if item is not default and keys:
            if isinstance(item, HDF5Group):
                return item.get(key=keys, default=default)
            else:
                return item.get(keys)
        else:
            return item

    def keys(
        self,
        as_file_keys: bool = False,
        load: bool = False,
        mapped: bool = False,
    ) -> KeysView:
        """Get all members in this group as an KeysView.

        Args:
            as_file_keys: Determines if the returned keys will be in the file or python form.
            load: Determines if this object will recursively load the members from the file on construction.
            mapped: Determines if this object will only add object that are mapped.

        Returns:
            The names and members in this group.
        """
        self.get_members(load=load, mapped=mapped)
        keys = self.members.keys()
        if as_file_keys:
            return keys
        else:
            return {self.map.map_names.inverse.get(key, key): None for key in keys}.keys()

    def items(
        self,
        as_file_keys: bool = False,
        load: bool = False,
        mapped: bool = False,
    ) -> ItemsView:
        """Get all members in this group as an ItemView.

        Args:
            as_file_keys: Determines if the returned keys will be in the file or python form.
            load: Determines if this object will recursively load the members from the file on construction.
            mapped: Determines if this object will only add object that are mapped.

        Returns:
            The names and members in this group.
        """
        self.get_members(load=load, mapped=mapped)
        items = self.members.items()
        if as_file_keys:
            return items
        else:
            return {self.map.map_names.inverse.get(k, k): i for k, i in items}.items()

    def values(self, load: bool = False, mapped: bool = False) -> ValuesView:
        """Get all members in this group as an ValuesView.

        Args:
            load: Determines if this object will recursively load the members from the file on construction.
            mapped: Determines if this object will only add object that are mapped.

        Returns:
            The names and members in this group.
        """
        self.get_members(load=load, mapped=mapped)
        return self.members.values()

    # Group Modification
    def create_group(
        self, 
        name: str, 
        map_: HDF5Map | None = None, 
        mode: str | None = None,
        require: bool = True,
        construct: bool = False,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        track_order: bool | None = None,
    ) -> "HDF5Group":
        """Creates a group in this group.

        Args:
            name: The HDF5 name of the group to create.
            map_: The map of the group to create.
            mode: The edit mode of the group to create.
            require: Determines if the group will be created on construction.
            construct: Determines if the group will create its members in the file on construction.
            component_kwargs: The keyword arguments for creating the components.
            component_types: Component class and their keyword arguments to instantiate.
            components: Components to add.
            track_order: Track dataset/group/attribute creation order under the group if True. If None use global
                default h5.get_config().track_order.

        Returns:
            The new group.
        """
        if map_ is None:
            map_ = self.default_group_map

        with self.file.temp_open():
            self.members[name] = group = HDF5Group(
                name=f"{self._full_name}/{name}",
                map_=map_,
                mode=self._mode if mode is None else mode,
                file=self.file,
                require=require,
                construct=construct,
                parent=self._full_name,
                component_kwargs=component_kwargs,
                component_types=component_types,
                components=components,
                track_order=track_order,
            )

        return group

    def require_group(
        self, 
        name: str, 
        map_: HDF5Map | None = None, 
        mode: str | None = None,
        load: bool = False,
        require: bool = True,
        construct: bool = False,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        mapped: bool = True,
        track_order: bool | None = None,
    ) -> "HDF5Group":
        """Creates a group if it does not exist.

        Args:
            name: The HDF5 name of the group to create.
            map_: The map of the group to create.
            mode: The edit mode of the group to create.
            load: Determines if the group will load on construction.
            require: Determines if this object will be created on construction.
            construct: Determines if this object will create its members in the file on construction.
            component_kwargs: The keyword arguments for creating the components.
            component_types: Component class and their keyword arguments to instantiate.
            components: Components to add.
            mapped: Determines if the member will have a default assigned to it if no map is found.
            track_order: Track dataset/group/attribute creation order under the group if True. If None use global
                default h5.get_config().track_order.

        Returns:
            The requested group.
        """
        group = self.get_member(name, default=None, load=load, mapped=mapped)
        if group is None:
            group = self.create_group(
                name=name, 
                map_=map_, 
                mode=mode, 
                construct=construct, 
                require=require, 
                component_kwargs=component_kwargs,
                component_types=component_types,
                components=components,
                track_order=track_order,
            )
            
        return group

    def create_dataset(
        self,
        name: str,
        data: np.ndarray | None = None,
        map_: HDF5Map | None = None,
        mode: str | None = None,
        require: bool = False,
        construct: bool = False,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        scale_name: str | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HDF5Dataset:
        """Creates a dataset in this group.

        Args:
            name: The HDF5 name of the dataset to create.
            data: The data to fill in the dataset.
            map_: The map of the dataset to create.
            mode: The edit mode of the dataset to create.
            require: Determines if the group will be created on construction.
            construct: Determines if the group will create its members in the file on construction.
            dtype: The dtype of the dataset.
            scale_name: Makes the data an axis with this name.
            casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.
            component_kwargs: The keyword arguments for creating the components.
            component_types: Component class and their keyword arguments to instantiate.
            components: Components to add.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.

        Returns:
            The new dataset.
        """
        if map_ is None:
            map_ = self.default_dataset_map

        with self.file.temp_open():
            self.members[name] = dataset = HDF5Dataset(
                data=data,
                name=f"{self._full_name}/{name}",
                map_=map_,
                mode=self._mode if mode is None else mode,
                file=self.file,
                require=require,
                construct=construct,
                parent=self._full_name,
                dtype=dtype,
                scale_name=scale_name,
                casting_kwargs=casting_kwargs,
                component_kwargs=component_kwargs,
                component_types=component_types,
                components=components,
                **kwargs,
            )

        return dataset

    def require_dataset(
        self,
        name: str,
        data: np.ndarray | None = None,
        map_: HDF5Map | None = None,
        mode: str | None = None,
        load: bool = False,
        require: bool = False,
        construct: bool = False,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        scale_name: str | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        mapped: bool = True,
        **kwargs: Any,
    ) -> HDF5Dataset:
        """Creates a dataset if it does not exist.

        Args:
            name: The HDF5 name of the dataset to create.
            data: The data to fill in the dataset.
            map_: The map of the dataset to create.
            mode: The edit mode of the dataset to create.
            load: Determines if the dataset will load from the file on construction.
            require: Determines if the group will be created on construction.
            construct: Determines if the group will create its members in the file on construction.
            dtype: The dtype of the dataset.
            scale_name: Makes the data an axis with this name.
            casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.
            component_kwargs: The keyword arguments for creating the components.
            component_types: Component class and their keyword arguments to instantiate.
            components: Components to add.
            mapped: Determines if the member will have a default assigned to it if no map is found.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.

        Returns:
            The requested dataset.
        """
        dataset = self.get_member(name, default=None, load=load, require=require, mapped=mapped)
        if dataset is None:
            dataset = self.create_dataset(
                name=name,
                data=data,
                map_=map_,
                mode=mode,
                require=require,
                construct=construct,
                dtype=dtype,
                scale_name=scale_name,
                casting_kwargs=casting_kwargs,
                component_kwargs=component_kwargs,
                component_types=component_types,
                components=components,
                **kwargs,
            )

        return dataset

    def print_contents(self, indent: int = 0) -> None:
        """Prints the entire contents.

        Args:
            indent: The number of space to print between each layer.
        """
        if self.attributes:
            print(f"{' ' * indent}  Attributes:")
            for name in self.attributes.keys():
                print(f"{' ' * indent}      {name}")
        if self.members:
            print(f"{' ' * indent}  Contents: ({''.join(f'{name}, ' for name in self.keys())})")
            for name, value in self.items():
                print(f"{' ' * indent}  +  {name}: {value._full_name} {value.map}")
                value.print_contents(indent=indent + 5)


# Assign Cyclic Definitions
GroupMap.default_type = HDF5Group
HDF5Group.default_map = GroupMap()
HDF5Group.default_group_map = GroupMap
