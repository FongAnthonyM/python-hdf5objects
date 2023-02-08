""" nodedatasetcomponent.py
A component which adds node heieratchy methods to a dataset.
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
from typing import Any

# Third-Party Packages #
from dspobjects.time import nanostamp
import h5py
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Map, HDF5Dataset
from ...dataset import BaseDatasetComponent


# Definitions #
# Classes #
class NodeDatasetComponent(BaseDatasetComponent):
    """Adds node heieratchy methods to a dataset.

    Class Attributes:
        default_reference_field: The default name of the data field that contains the object references.

    Attributes:
        reference_field: The name of the data field that contains the object references.

    Args:
        composite: The object which this object is a component of.
        reference_field: The name of the data field that contains the object references.
        **kwargs: Keyword arguments for inheritance.
    """
    default_refernce_field: str = "Node"

    # Magic Methods
    # Constructors/Destructors
    def __init__(
        self,
        composite: Any = None,
        reference_field: str | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.reference_field: str = self.default_refernce_field

        # Parent Attributes #
        super().__init__(self, init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                reference_field=reference_field,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        reference_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            reference_field: The name of the data field that contains the object references.
            **kwargs: Keyword arguments for inheritance.
        """
        if reference_field is not None:
            self.reference_field = reference_field

        super().construct(composite=composite, **kwargs)

    # Node
    def set_entry_dict(
        self,
        index: int,
        item: dict,
        map_: HDF5Map | None = None,
    ) -> None:
        """Set an entry's values to a given item as a dictionary.

        Args:
            index: The index to set to the given item.
            item: The item to set as a dictionary.
            map_: The map to the object that should be stored in the entry.
        """
        if map_ is not None:
            item[self.default_refernce_field] = map_.get_object(require=True, file=self.composite.file).ref

        self.composite.set_item_dict(index, item)

    def append_entry_dict(
        self,
        item: dict,
        map_: HDF5Map | None = None,
    ) -> None:
        """Append an entry item as a dictionary.

        Args:
            item: The item to set as a dictionary.
            map_: The map to the object that should be stored in the entry.
        """
        if map_ is not None:
            item[self.default_refernce_field] = map_.get_object(require=True, file=self.composite.file).ref
        elif self.default_refernce_field not in item:
            item[self.default_refernce_field] = h5py.Reference()

        self.composite.append_data_item_dict(item)

    def insert_entry_dict(
        self,
        index: int,
        item: dict,
        map_: HDF5Map | None = None,
    ) -> None:
        """Insert an entry item as a dictionary.

        Args:
            index: The index to insert the given item.
            item: The item to insert as a dictionary.
            map_: The map to the object that should be stored in the entry.
        """
        if map_ is not None:
            item[self.default_refernce_field] = map_.get_object(require=True, file=self.composite.file).ref
        elif self.default_refernce_field not in item:
            item[self.default_refernce_field] = h5py.Reference()

        self.composite.insert_data_item_dict(index=index, dict_=item)
