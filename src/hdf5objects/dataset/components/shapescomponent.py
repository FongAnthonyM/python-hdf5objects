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
from collections.abc import Mapping, Iterable
from typing import Any
from uuid import UUID

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod, search_sentinel
from bidict import bidict
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Dataset
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
        if id_fields is not None:
            self.id_fields.clear()
            self.id_fields.update(id_fields)

        if uuid_fields is not None:
            self.uuid_fields.clear()
            self.uuid_fields.update(uuid_fields)

        super().construct(composite=composite, **kwargs)

    def get_min_shape(self) -> np.ndarray:
        return self.composite.min(self.composite.axis)

    def get_max_shape(self) -> np.ndarray:
        return self.composite.max(self.composite.axis)
