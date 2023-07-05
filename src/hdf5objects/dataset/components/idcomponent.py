"""iddataset.py
A component for a HDF5Dataset which gives it ID manipulation methods.
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
from collections.abc import Iterable
from typing import Any
from uuid import UUID

# Third-Party Packages #
from baseobjects import search_sentinel
from bidict import bidict
import numpy as np

# Local Packages #
from ..basedatasetcomponent import BaseDatasetComponent


# Definitions #
# Classes #
class IDComponent(BaseDatasetComponent):
    """A component for a HDF5Dataset which gives it ID manipulation methods.

    Class Attributes:
        default_id_fields: The default fields of the dtype that store string IDs.
        default_uuid_fields: The default fields of the dtype that store UUIDs.

    Attributes:
        id_fields: The fields of the dtype that store string IDs.
        uuid_fields: The fields of the dtype that store UUIDs.

        _id_arrays: dict = The IDs stored as arrays separated by type.
        ids:  The IDs stored as bidict separated by type.

    Args:
        composite: The object which this object is a component of.
        id_fields: The fields of the dtype that store string IDs.
        uuid_fields:  The fields of the dtype that store UUIDs.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    default_id_fields: set[str] = set()
    default_uuid_fields: set[str] = set()

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        composite: Any = None,
        id_fields: Iterable[str] | None = None,
        uuid_fields: set[str] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.id_fields: set[str] = self.default_id_fields.copy()
        self.uuid_fields: set[str] = self.default_uuid_fields.copy()

        self._id_arrays: dict = {}
        self.ids: dict[str:bidict] = {}

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                id_fields=id_fields,
                uuid_fields=uuid_fields,
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
        composite: Any = None,
        id_fields: set[str] | None = None,
        uuid_fields: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            id_fields: The fields of the dtype that store string IDs.
            uuid_fields: The fields of the dtype that store UUIDs.
            **kwargs: Keyword arguments for inheritance.
        """
        if id_fields is not None:
            self.id_fields.clear()
            self.id_fields.update(id_fields)

        if uuid_fields is not None:
            self.uuid_fields.clear()
            self.uuid_fields.update(uuid_fields)

        super().construct(composite=composite, **kwargs)

    def build_id_arrays(self) -> None:
        """Builds the arrays which store the IDs by type."""
        for id_field in self.all_id_fields:
            if id_field in self.composite.dtypes_dict:
                self._id_arrays[id_field] = self.composite[id_field]
            else:
                raise KeyError(f"{id_field} is missing from the dataset fields")

    def load_ids(self) -> None:
        """Loads the IDs from the id_arrays into dictionary of bidicts."""
        self.ids.clear()
        for id_field, array in self._id_arrays.items():
            a_iter = np.nditer(array, flags=["multi_index"])
            if id_field in self.uuid_fields:
                self.ids[id_field] = bidict({a_iter.multi_index: UUID(id_) for id_ in a_iter})
            else:
                self.ids[id_field] = bidict({a_iter.multi_index: id_ for id_ in a_iter})

    # ID Getters and Setters
    def _get_id(self, id_type: str, index: int | tuple) -> Any:
        """Get an ID from the id_arrays.

        Args:
            id_type: The type of ID to get.
            index: The index of the ID in the array.

        Returns:
            The requested ID.
        """
        try:
            id_ = self._id_arrays[id_type][index]
            return UUID(id_) if id_type in self.uuid_fields else id_
        except ValueError:
            return None

    def get_id(self, id_type: str, index: int | tuple) -> Any:
        """Get an ID from from the ids.

        Args:
            id_type: The type of ID to get.
            index: The index of the ID in the bidict.

        Returns:
            The requested ID.
        """
        id_ = self.ids[id_type].get(index, search_sentinel)
        if id_ is search_sentinel:
            id_ = self._get_id(id_type=id_type, index=index)
            if id_ is not None:
                self.ids[id_type][index] = id_
        return id_

    def set_id(self, id_type: str, index: int | tuple, id_: Any) -> None:
        """Sets an ID at an index.

        Args:
            id_type: The type of ID to set.
            index: The index of the ID in the bidict.
            id_: The ID to set.
        """
        if id_type in self.uuid_fields and isinstance(id_, str):
            id_ = UUID(id_)
        self.ids[id_type][index] = id_

        if isinstance(id_, UUID):
            id_ = str(id_)

        item = self.composite[index]
        item[id_type] = id_
        self.composite[index] = item

    def _find_id(self, id_type: str, id_: Any) -> Any:
        """Finds the index of an ID using the id_arrays.

        Args:
            id_type: The type of ID being use to find.
            id_: The ID to use to search for the item.

        Returns:
            The item requested.
        """
        id_ = str(id_) if isinstance(id_, UUID) else id_
        indices = np.where(self._id_arrays[id_type] == id_)
        if isinstance(indices, tuple) and len(indices[0]) == 1:
            return tuple(axis[0] for axis in indices)
        elif len(indices) == 1:
            return indices[0]
        elif len(indices) == 0:
            return None
        else:
            raise KeyError(f"Multiple instances of ID: {id_}")

    def find_id(self, id_type: str, id_: Any) -> Any:
        """Finds the index of an ID using ids.

        Args:
            id_type: The type of ID being use to find.
            id_: The ID to use to search for the item.

        Returns:
            The item requested.
        """
        index = self.ids[id_type].inverse.get(id_, search_sentinel)
        if index is search_sentinel:
            index = self._find_id(id_type=id_type, id_=id_)
            if index is not None:
                self.ids[id_type].inverse[id_] = index
        return index

    # Item Getters
    def item_from_id(self, id_type: str, id_: Any) -> Any:
        """Finds and returns an item using the ID.

        Args:
            id_type: The type of ID being use to find.
            id_: The ID to use to search for the item.

        Returns:
            The item requested.
        """
        index = self.find_id(id_type=id_type, id_=id_)
        return self.composite[index]

    def dict_from_id(self, id_type: str, id_: Any) -> dict:
        """Finds and returns an item as a dictionary using the ID.

        Args:
            id_type: The type of ID being use to find.
            id_: The ID to use to search for the item.

        Returns:
            The item requested as a dictionary.
        """
        index = self.find_id(id_type=id_type, id_=id_)
        return self.composite.item_to_dict(self[index])
