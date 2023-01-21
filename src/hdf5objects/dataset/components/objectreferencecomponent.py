""" regionreferencedataset.py
A component for a HDF5Dataset which implement methods for object and region references.
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

# Third-Party Packages #
from baseobjects import search_sentinel
from bidict import bidict
import h5py
import numpy as np

# Local Packages #
from ..basedatasetcomponent import BaseDatasetComponent


# Definitions #
# Classes #
class ObjectReferenceComponent(BaseDatasetComponent):
    """A component for a HDF5Dataset which implement methods for object references.

    Class Attributes:
        default_reference_fields: The default fields of the dtype that contain object references.
        default_primary_reference_field: The default name of the reference to get when the name is not given.

    Attributes:
        reference_fields: The fields of the dtype that contain object references.
        primary_reference_field: The name of the reference to get when the name is not given.

    Args:
        composite: The object which this object is a component of.
        reference_fields: The fields of the dtype that contain object references.
        primary_reference_field: The name of the reference to get when the name is not given.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """
    default_reference_fields: dict[str, str] = dict()
    default_primary_reference_field: str | None = None

    # Magic Methods
    # Constructors/Destructors
    def __init__(
        self,
        composite: Any = None,
        reference_fields: dict[str, str] | None = None,
        primary_reference_field: str | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.reference_fields: dict[str, str] = self.default_reference_fields.copy()
        self.primary_reference_field: str | None = self.default_primary_reference_field

        # Parent Attributes #
        super().__init__(self, init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                reference_fields=reference_fields,
                primary_reference_field=primary_reference_field,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        reference_fields: dict[str, str] | None = None,
        primary_reference_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            reference_fields: The fields of the dtype that contain object references.
            primary_reference_field: The name of the reference to get when the name is not given.
            **kwargs: Keyword arguments for inheritance.
        """
        if reference_fields is not None:
            self.reference_fields.clear()
            self.reference_fields.update(reference_fields)

        if primary_reference_field is not None:
            self.primary_reference_field = primary_reference_field

        super().construct(composite=composite, **kwargs)

    def generate_object_reference(
        self,
        object_: h5py.Dataset | h5py.Reference | None = None,
    ) -> tuple:
        """Creates reference objects for a given HDF5 object.

        Args:
            object_: The HDF5 object to make a reference for.

        Returns:
            The object and region references.
        """
        if not isinstance(object_, h5py.Reference):
            object_ = object_.ref

        return object_

    def get_object_reference(self, index: int | tuple, ref_name: str | None = None) -> Any:
        """Gets the object reference.

        Args:
            index: The index in the dataset to get the reference from.
            ref_name: The name of the type of reference to get.

        Returns:
            The object reference.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        return self.composite[index][self.composite.dtypes_dict[self.reference_fields[ref_name]]]

    def get_object(self, index: int | tuple, ref_name: str | None = None) -> Any:
        """Gets the HDF5 object referenced.

        Args:
            index: The index in the dataset to get the reference from.
            ref_name: The name of the type of reference to get.

        Returns:
            The requested HDF5 object.
        """
        return self.composite.file[self.get_object_reference(index=index, ref_name=ref_name)]

    def get_objects(self, ref_name: str | None = None) -> tuple[Any]:
        """Gets the HDF5 objects referenced in the dataset.

        Args:
            ref_name: The name of the type of references to get.

        Returns:
            The requested HDF5 objects.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        references = self.composite[self.composite.dtypes_dict[self.reference_fields[ref_name]]]
        return tuple(self.composite.file[ref] for ref in references)

    def get_objects_iter(self, ref_name: str | None = None) -> Iterable:
        """Gets the HDF5 objects referenced in the dataset as an Iterable.

        Args:
            ref_name: The name of the type of reference to get.

        Returns:
            An Iterable of the requested HDF5 objects.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        references = self.composite[self.composite.dtypes_dict[self.reference_fields[ref_name]]]
        return (self.composite.file[ref] for ref in references)

    def set_object_reference(
        self,
        index: int | tuple,
        object_: h5py.Dataset | h5py.Reference,
        ref_name: str | None = None,
    ) -> None:
        """Sets an object reference at a given index.

        Args:
            index: The index in the dataset to set the object reference to.
            object_: The HDF5 object to set, can be either the HDF5 object or a reference to that object.
            ref_name: The name of the type of reference to set.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        if not isinstance(object_, h5py.Reference):
            object_ = object_.ref

        item = self.composite[index]
        item[self.composite.dtypes_dict[self.reference_fields[ref_name]]] = object_
        self.composite[index] = item
