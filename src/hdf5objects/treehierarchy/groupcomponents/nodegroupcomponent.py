""" nodegroupcomponent.py
A component which adds node heieratchy methods to a group.
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
from datetime import datetime, date, tzinfo
from decimal import Decimal
from typing import Any
import uuid

# Third-Party Packages #
from dspobjects.time import Timestamp
from hdf5objects import HDF5Map, HDF5Dataset
from hdf5objects import HDF5BaseComponent
import numpy as np

# Local Packages #


# Definitions #
# Classes #
class NodeGroupComponent(HDF5BaseComponent):
    """Adds node heieratchy methods to a group.

    Class Attributes:
        default_child_map_type: The default type of map to create when creating a child node.
        default_child_component_name: The default name of the component in the child node which adds node methods.
        default_map_dataset_name: The default name of the dataset which maps all of the child nodes within this node.
        default_node_component_name: The default name of the component in the dataset which adds node methods.

    Attributes:
        child_map_type: The type of map to create when creating a child node.
        child_component_name: The name of the component in the child node which adds node methods.
        map_dataset_name: The name of the dataset which maps all of the child nodes within this node.
        node_component_name: The name of the component in the dataset which adds node methods.

    Args:
        composite: The object which this object is a component of.
        child_map_type: The type of map to create when creating a child node.
        child_component_name: The name of the component in the child node which adds node methods.
        map_dataset_name: The name of the dataset which maps all of the child nodes within this node.
        node_component_name: The name of the component in the dataset which adds node methods.
        **kwargs: Keyword arguments for inheritance.
    """
    default_child_map_type: type | None = None
    default_child_component_name: str = "tree_node"
    default_map_dataset_name: str = "map_dataset"
    default_node_component_name: str = "tree_node"

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        composite: Any = None,
        child_map_type: type | None = None,
        child_component_name: str | None = None,
        map_dataset_name: str | None = None,
        node_component_name: str| None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.child_map_type: type | None = self.default_child_map_type
        self.child_component_name: str = self.default_child_component_name
        self.map_dataset_name: str = self.default_map_dataset_name
        self.node_component_name: str = self.default_node_component_name

        self._map_dataset: HDF5Dataset | None = None

        # Parent Attributes #
        super().__init__(self, init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                child_map_type=child_map_type,
                map_dataset_name=map_dataset_name,
                node_component_name=node_component_name
                **kwargs,
            )

    @property
    def map_dataset(self) -> HDF5Dataset | None:
        "The dataset which maps all of the child nodes within this node."
        if self.map_dataset is None:
            self._map_dataset = self.composite[self.map_dataset_name]
        return self._map_dataset

    @map_dataset.setter
    def map_dataset(self, value: HDF5Dataset | None) -> None:
        self._map_dataset = value

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        child_map_type: type | None = None,
        child_component_name: str | None = None,
        map_dataset_name: str | None = None,
        node_component_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            child_map_type: The type of map to create when creating a child node.
            child_component_name: The name of the component in the child node which adds node methods.
            map_dataset_name: The name of the dataset which maps all of the child nodes within this node.
            node_component_name: The name of the component in the dataset which adds node methods.
            **kwargs: Keyword arguments for inheritance.
        """
        if child_map_type is not None:
            self.child_map_type = child_map_type

        if child_component_name is not None:
            self.child_component_name = child_component_name

        if map_dataset_name is not None:
            self.map_dataset_name = map_dataset_name

        if node_component_name is not None:
            self.node_component_name = node_component_name

        super().construct(composite=composite, **kwargs)

    def create_child_dict(
        self,
        index: int,
        item: dict,
        map_: HDF5Map | None = None,
    ) -> None:
        """Creates a child node and inserts it as an entry item dictionary.

        Args:
            index: The index to insert the child in the dataset.
            item: The item dictionary to insert the child with.
            map_: An optional map which the child would be created from.
        """
        if map_ is None:
            map_ = self.child_map_type(name=f"{self.composite.name}/{path}")
            self.composite.map.set_item(map_)

        self.map_dataset.components[self.node_component_name].insert_entry(
            index=index,
            item=item,
            map_=map_,
        )

        return self.composite[map_.name]
