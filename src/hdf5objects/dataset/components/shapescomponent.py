""" shapescomponent.py
A component for a HDF5Dataset which gives it shape manipulation methods.
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
import numpy as np

# Local Packages #
from ..basedatasetcomponent import BaseDatasetComponent


# Definitions #
# Classes #
class ShapesComponent(BaseDatasetComponent):
    """A component for a HDF5Dataset which gives it shape manipulation methods.

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
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(composite=composite, **kwargs)

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            id_fields: The fields of the dtype that store string IDs.
            uuid_fields: The fields of the dtype that store UUIDs.
            **kwargs: Keyword arguments for inheritance.
        """
        super().construct(composite=composite, **kwargs)

    def get_min_shape(self) -> np.ndarray:
        return tuple(np.amin(self.composite, 0)) if self.composite.size != 0 else (0,)

    def get_max_shape(self) -> np.ndarray:
        return tuple(np.amax(self.composite, 0)) if self.composite.size != 0 else (0,)

    def set_shape(self, index: int, shape: tuple):
        # Get the shapes of the dataset and the new data to be added
        s_shape = np.asarray(self.composite.shape)
        d_shape = [1, len(shape)]

        # Set the shape of the dataset if it needs to change
        if s_shape[1] < d_shape[1]:
            self.composite.resize((index + 1 if index == s_shape[0] else s_shape[0], max(s_shape[1], d_shape[1])))

        self.composite[index, :len(shape)] = shape
