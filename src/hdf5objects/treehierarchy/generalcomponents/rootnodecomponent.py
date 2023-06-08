"""rootnodecomponent.py
Adds methods for managing the root node of a tree hierarchy.
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

# Local Packages #
from ...hdf5bases import HDF5Group, HDF5BaseComponent
from ..groupcomponents import NodeGroupComponent


# Definitions #
# Classes #
class RootNodeComponent(HDF5BaseComponent):
    """Adds methods for managing the root node of a tree hierarchy.

    Class Attributes:
        default_root_location: The default location in the file where the root of the data hierarchy is.

    Attributes:
        root_location: The location in the file where the root of the data hierarchy is.

    Args:
        composite: The object which this object is a component of.
        root_location: The location in the file where the root of the data hierarchy is.
        **kwargs: Keyword arguments for inheritance.
    """

    default_root_location: str = ""
    default_node_component_name: str = "tree_node"

    # Magic Methods #
    # Constructors/Destructors
    def __init__(
        self,
        composite: Any = None,
        root_location: str | None = None,
        node_component_name: str | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.root_location: str = self.default_root_location
        self.node_component_name: str = self.default_node_component_name

        # Parent Attributes #
        super().__init__(self, init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                root_location=root_location,
                node_component_name=node_component_name,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        root_location: str | None = None,
        node_component_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            root_location: The location in the file where the root of the data hierarchy is.
            **kwargs: Keyword arguments for inheritance.
        """
        if root_location is not None:
            self.root_location = root_location

        if node_component_name:
            self.node_component_name = node_component_name

        super().construct(composite=composite, **kwargs)

    def get_root(self) -> HDF5Group:
        """Get the root node of a tree hierarchy within this file.

        Returns:
            The root node.
        """
        return self.composite[self.root_location]

    def get_root_node_component(self) -> NodeGroupComponent:
        return self.composite[self.root_location].components[self.node_component_name]
