""" idaxiscomponent.py
A component and map for a HDF5Dataset which defines it as an axis with IDs for each datum.
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
import uuid
from uuid import UUID

# Third-Party Packages #
from baseobjects import search_sentinel
from bidict import bidict
import numpy as np

# Local Packages #
from .axiscomponent import AxisMap, AxisComponent


# Definitions #
# Classes #
class IDAxisComponent(AxisComponent):
    """A component for a HDF5Dataset which defines it as an axis with IDs for each datum.

    Args:
        composite: The object which this object is a component of.
        is_uuid: Determines if the IDs are UUIDs
        rate: The frequency of the data of the axis.
        size: The number of datum in the axis.
        create: Determines if the axis should be created and filled.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """
    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        composite: Any = None,
        is_uuid: bool = False,
        size: int | None = None,
        create: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.is_uuid: bool = False
        self.ids: bidict = bidict()

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                is_uuid=is_uuid,
                size=size,
                create=create,
                **kwargs,
            )

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        is_uuid: bool = False,
        size: int | None = None,
        create: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            is_uuid: Determines if the IDs are UUIDs
            rate: The frequency of the data of the axis.
            size: The number of datum in the axis.
            create: Determines if the axis should be created and filled.
            **kwargs: Keyword arguments for inheritance.
        """
        if is_uuid is not None:
            self.is_uuid = self.is_uuid

        super().construct(composite=composite,  **kwargs)

        if create and self.is_uuid:
            self.generate_uuids()

        if self._composite is not None and self.composite.exists:
            self.load_ids()

    def load_ids(self) -> None:
        """Loads the IDs from the id_arrays into dictionary of bidicts."""
        if self.composite.all_data.size != 0:
            self.ids.clear()
            self.ids.update(
                {i: UUID(id_[0].decode()) if self.is_uuid else id_[0].decode()
                 for i, id_ in enumerate(self.composite.all_data)}
            )

    def generate_uuids(self, size: int, **kwargs) -> None:
        """Generates the IDs for this axis.

        Args:
            size: The number of IDs to generate.
            **kwargs: Keyword arguments for setting the data.
        """
        self.composite.set_data(data=(str(uuid.uuid4()) for i in range(size)), **kwargs)

    # ID Getters and Setters
    def _get_id(self, index: int | tuple) -> Any:
        """Get an ID from the id_arrays.

        Args:
            index: The index of the ID in the array.

        Returns:
            The requested ID.
        """
        try:
            id_ = self._id_arrays[index]
            return UUID(id_) if self.is_uuid else id_
        except ValueError:
            return None

    def get_id(self, index: int | tuple) -> Any:
        """Get an ID from from the ids.

        Args:
            index: The index of the ID in the bidict.

        Returns:
            The requested ID.
        """
        id_ = self.ids.get(index, search_sentinel)
        if id_ is search_sentinel:
            id_ = self._get_id(index=index)
            if id_ is not None:
                self.ids[index] = id_
        return id_

    def set_id(self, index: int | tuple, id_: Any) -> None:
        """Sets an ID at an index.

        Args:
            index: The index of the ID in the bidict.
            id_: The ID to set.
        """
        if self.is_uuid and isinstance(id_, str):
            id_ = UUID(id_)
        self.ids[index] = id_

        if isinstance(id_, UUID):
            id_ = str(id_)

        self.composite[index] = id_

    def append_id(self, id_: str | UUID) -> None:
        """Appends an ID to this axis.

        Args:
            id_: The ID to set.
        """
        if self.is_uuid and isinstance(id_, str):
            id_ = UUID(id_)
        self.ids[len(self.composite)] = id_

        if isinstance(id_, UUID):
            id_ = str(id_)

        self.composite.append_data(np.array([id_]))

    def insert_id(self, index: int | slice | Iterable[int], id_: str | UUID) -> None:
        """Appends an ID to this axis.

        Args:
            index: The index or slice to insert the data into.
            id_: The ID to set.
        """
        if isinstance(id_, UUID):
            id_ = str(id_)

        self.composite.insert_data(index=index, data=np.array([id_]))
        self.load_ids()

    def _find_id(self, id_: Any) -> Any:
        """Finds the index of an ID using the id_arrays.

        Args:
            id_: The ID to use to search for the item.

        Returns:
            The item requested.
        """
        id_ = str(id_) if isinstance(id_, UUID) else id_
        indices = np.where(self.composite.all_data == id_)
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
            id_: The ID to use to search for the item.

        Returns:
            The item requested.
        """
        index = self.ids.inverse.get(id_, search_sentinel)
        if index is search_sentinel:
            index = self._find_id(id_=id_)
            if index is not None:
                self.ids.inverse[id_] = index
        return index

    # Item Getters
    def item_from_id(self, id_: Any) -> Any:
        """Finds and returns an item using the ID.

        Args:
            id_: The ID to use to search for the item.

        Returns:
            The item requested.
        """
        index = self.find_id(id_=id_)
        return self.composite[index]

    def dict_from_id(self, id_: Any) -> dict:
        """Finds and returns an item as a dictionary using the ID.

        Args:
            id_: The ID to use to search for the item.

        Returns:
            The item requested as a dictionary.
        """
        index = self.find_id(id_=id_)
        return self.composite.item_to_dict(self[index])


class IDAxisMap(AxisMap):
    """An outline which defines an HDF5Dataset as an Axis with IDs for each datum."""
    default_kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,)}
    default_dtype = (("ID", str),)
    default_component_types = {
        "axis": (IDAxisComponent, {}),
    }
