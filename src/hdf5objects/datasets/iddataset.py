""" iddataset.py
A Dataset which contains IDs in each datum.
"""
# Package Header #
from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from collections.abc import Mapping, Iterable
from typing import Any
from uuid import UUID

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod, search_sentinel
from bidict import bidict
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Map, HDF5Dataset, DatasetMap


# Definitions #
# Classes #
class IDDatasetMap(DatasetMap):
    """The map for a dataset which stores IDs.

    Class Attributes:
        default_id_fields: The default fields of the dtype that store string IDs.
        default_uuid_fields: The default fields of the dtype that store UUIDs.

    Attributes:
        id_fields: The fields of the dtype that store string IDs.
        uuid_fields: The fields of the dtype that store UUIDs.
        
    Args:
        name: The name of this map.
        type_: The type of the hdf5 object this map represents.
        attribute_names: The name map of python name vs hdf5 name of the attribute.
        attributes: The default values of the attributes of the represented hdf5 object.
        map_names: The name map of python name vs hdf5 name of the maps contained within this map.
        maps: The nested maps.
        parent: The parent of this map.
        dtype: The dtype of the dataset this object will map.
        casting_kwargs: The keyword arguments for the dtype casting.
        id_fields: The fields of the dtype that store string IDs.
        uuid_fields: The fields of the dtype that store UUIDs.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the object this map represents.
    """
    default_id_fields: set[str] = set()
    default_uuid_fields: set[str] = set()

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        name: str | None = None,
        type_: type | None = None,
        attribute_names: Mapping[str, str] | None = None,
        attributes: Mapping[str, Any] | None = None,
        map_names: Mapping[str, str] | None = None,
        maps: Mapping[str, HDF5Map] | None = None,
        parent: str | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        id_fields: set[str] | None = None,
        uuid_fields: set[str] | None = None,
        init: bool = True,
        **kwargs: Any
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self.id_fields: set[str] = self.default_id_fields.copy()
        self.uuid_fields: set[str] = self.default_uuid_fields.copy()

        # Object Construction #
        if init:
            name = name if name is not None else self.default_name
            parent = parent if parent is not None else self.default_parent
            self.construct(
                name=name,
                type_=type_,
                attribute_names=attribute_names,
                attributes=attributes,
                map_names=map_names,
                maps=maps,
                parent=parent,
                id_fields=id_fields,
                uuid_fields=uuid_fields,
                dtype=dtype,
                casting_kwargs=casting_kwargs,
                **kwargs,
            )

    @property
    def all_id_fields(self) -> set[str]:
        """All fields of the dtype that store IDs."""
        return self.id_fields | self.uuid_fields

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        name: str | None = None,
        type_: type | None = None,
        attribute_names: Mapping[str, str] | None = None,
        attributes: Mapping[str, Any] | None = None,
        map_names: Mapping[str, str] | None = None,
        maps: Mapping[str, HDF5Map] | None = None,
        parent: str | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        id_fields: set[str] | None = None,
        uuid_fields: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object, setting attributes and sets nested maps' parents.

        Args:
            name: The name of this map.
            type_: The type of the hdf5 object this map represents.
            attribute_names: The name map of python name vs hdf5 name of the attribute.
            attributes: The default values of the attributes of the represented hdf5 object.
            map_names: The name map of python name vs hdf5 name of the maps contained within this map.
            maps: The nested maps.
            parent: The parent of this map.
            dtype: The dtype of the dataset this object will map.
            casting_kwargs: The keyword arguments for the dtype casting.
            id_fields: The fields of the dtype that store string IDs.
            uuid_fields: The fields of the dtype that store UUIDs.
            **kwargs: The keyword arguments for the object this map represents.
        """
        if id_fields is not None:
            self.id_fields.clear()
            self.id_fields.update(id_fields)

        if uuid_fields is not None:
            self.uuid_fields.clear()
            self.uuid_fields.update(uuid_fields)

        super().construct(
            name=name,
            type_=type_,
            attribute_names=attribute_names,
            attributes=attributes,
            map_names=map_names,
            maps=maps,
            parent=parent,
            dtype=dtype,
            casting_kwargs=casting_kwargs,
            **kwargs,
        )


class IDDataset(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        data: np.ndarray | None = None,
        types: tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        id_fields: Iterable[str] | None = None,
        load: bool = False,
        require: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._id_arrays: dict = {}
        self.ids: dict[str: bidict] = {}

        # Object Construction #
        if init:
            self.construct(
                data=data,
                types=types,
                casting_kwargs=casting_kwargs,
                id_fields=id_fields,
                load=load,
                require=require,
                **kwargs,
            )
    
    @property
    def id_fields(self) -> set[str]:
        """The fields of the dtype that store string IDs."""
        return self.map.id_fields
    
    @id_fields.setter
    def id_fields(self, value: set[str]) -> None:
        self.map.id_fields = value

    @property
    def uuid_fields(self) -> set[str]:
        """The fields of the dtype that store UUIDs."""
        return self.map.uuid_fields

    @uuid_fields.setter
    def uuid_fields(self, value: set[str]) -> None:
        self.map.uuid_fields = value
    
    @property
    def all_id_fields(self) -> set[str]:
        """All fields of the dtype that store IDs."""
        return self.map.all_fields

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        id_fields: set[str] | None = None,
        uuid_fields: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The data to fill in this dataset.
            dtype: The dtype of this dataset.
            id_fields: The fields of the dtype that store string IDs.
            uuid_fields: The fields of the dtype that store UUIDs.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.
        """
        super().construct(**kwargs)

        if id_fields is not None:
            self.id_fields.clear()
            self.id_fields.update(id_fields)

        if uuid_fields is not None:
            self.uuid_fields.clear()
            self.uuid_fields.update(uuid_fields)

    def build_id_arrays(self) -> None:
        for id_field in self.all_id_fields:
            if id_field in self.dtypes_dict:
                self._id_arrays[id_field] = self._dataset[id_field]
            else:
                raise KeyError(f"{id_field} is missing from the dataset fields")

    def load_ids(self):
        self.ids.clear()
        for id_field, array in self._id_arrays:
            a_iter = np.nditer(array, flags=['multi_index'])
            if id_field in self.uuid_fields:
                self.ids[id_field] = bidict({a_iter.multi_index: UUID(id_) for id_ in a_iter})
            else:
                self.ids[id_field] = bidict({a_iter.multi_index: id_ for id_ in a_iter})

    # ID Getters and Setters
    def _get_id(self, id_type: str, index: int | tuple) -> Any:
        try:
            id_ = self._id_arrays[id_type][index]
            return UUID(id_) if id_type in self.uuid_fields else id_
        except ValueError:
            return None

    def get_id(self, id_type: str, index: int | tuple) -> Any:
        id_ = self.ids[id_type].get(index, search_sentinel)
        if id_ is search_sentinel:
            id_ = self._get_id(id_type=id_type, index=index)
            if id_ is not None:
                self.ids[id_type][index] = id_
        return id_

    def set_id(self, id_type: str, index: int | tuple, id_: Any) -> None:
        if id_type in self.uuid_fields and isinstance(id_, str):
            id_ = UUID(id_)
        self.ids[id_type][index] = id_

        if isinstance(id_, UUID):
            id_ = str(id_)

        item = self[index]
        item[id_type] = id_
        self[index] = item

    def _find_id(self, id_type: str, id_: Any) -> Any:
        id_ = str(id_) if isinstance(id_, UUID) else id_
        indices = np.where(self.self._id_arrays[id_type] == id_)
        if isinstance(indices, tuple) and len(indices[0]) == 1:
            return tuple(axis[0] for axis in indices)
        elif len(indices) == 1:
            return indices[0]
        elif len(indices) == 0:
            return None
        else:
            raise KeyError(f"Multiple instances of ID: {id_}")

    def find_id(self, id_type: str, id_: Any) -> Any:
        index = self.ids[id_type].inverse.get(id_, search_sentinel)
        if index is search_sentinel:
            index = self._find_id(id_type=id_type, id_=id_)
            if index is not None:
                self.ids[id_type].inverse[id_] = index
        return index

    # Item Getters
    def item_from_id(self, id_type: str, id_: Any) -> Any:
        index = self.find_id(id_type=id_type, id_=id_)
        return self[index]

    def dict_from_id(self, id_type: str, id_: Any) -> dict:
        index = self.find_id(id_type=id_type, id_=id_)
        return self.item_to_dict(self[index])
