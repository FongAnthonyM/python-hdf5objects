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
from ...hdf5bases import HDF5Map, DatasetMap, HDF5Dataset


# Definitions #
# Classes #
class IDDataset(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_id_fields: set[str] = set()
    default_uuid_fields: set[str] = set()

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
        self.id_fields: set[str] = self.default_id_fields.copy()
        self.uuid_fields: set[str] = self.default_uuid_fields.copy()
        self.ids = bidict()

        self._id_arrays: dict = {}

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
    def all_id_fields(self) -> set[str]:
        return self.id_fields | self.uuid_fields

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        types: tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        id_fields: Iterable[str] | None = None,
        load: bool = False,
        require: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The data to fill in this dataset.
            load: Determines if this object will load the timeseries from the file on construction.
            require: Determines if this object will create and fill the timeseries in the file on construction.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.
        """
        if id_fields is not None:
            self.id_fields.clear()
            self.id_fields.update(id_fields)

        super().construct(data=data, types=types, casting_kwargs=casting_kwargs, load=load, require=require, **kwargs)

    def build_id_arrays(self) -> None:
        for id_field in self.all_id_fields:
            if id_field in self._types_dict:
                self._id_arrays[id_field] = self._dataset[id_field]
            else:
                raise KeyError(f"{id_field} is missing from the datasets fields")

    def load_ids(self):
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
