"""regionreferenceaxiscomponent.py
A component for a HDF5Dataset which defines it as an axis with region references for each datum.
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
from collections.abc import Mapping
from typing import Any

# Third-Party Packages #
import h5py

# Local Packages #
from ...hdf5bases import HDF5BaseObject
from ..basedatasetcomponent import BaseDatasetComponent
from .axiscomponent import AxisMap, AxisComponent


# Definitions #
# Classes #
class RegionReferenceAxisComponent(AxisComponent):
    """A component for a HDF5Dataset which defines it as an axis with region references for each datum.

    Class Attributes:
        default_object_attribute: The default attribute name with the object reference.
        default_object_field: The default field the dtype that contains the object references.
        default_region_field: The default field the dtype that contains the region references.

    Attributes:
        object_attribute: The attribute name with the object reference.
        object_field: The field the dtype that contains the object references.
        region_field: The field the dtype that contains the region references.
        _object: If there is a singel object reference, it its object is stored here for quick access.

    Args:
        composite: The object which this object is a component of.
        object_attribute: The attribute name with the object reference.
        object_field: The field the dtype that contains the object references.
        region_field: The field the dtype that contains the region references.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    defualt_object_attribute: str | None = "object_reference"
    default_object_field: str | None = None
    default_region_field: str = "Region"

    # Magic Methods
    # Constructors/Destructors
    def __init__(
        self,
        composite: Any = None,
        object_attribute: str | None = None,
        object_field: str | None = None,
        region_field: str | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.object_attribute: str | None = self.defualt_object_attribute
        self.object_field: str | None = self.default_object_field
        self.region_field: str = self.default_region_field

        self._object: HDF5BaseObject | None = None

        # Parent Attributes #
        super().__init__(self, init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                object_field=object_field,
                region_field=region_field,
                **kwargs,
            )

    @property
    def object(self) -> HDF5BaseObject | None:
        """The object referenced if it stored in the attributes."""
        if self._object is None:
            ref = self.composite.attributes.get(self.object_attribute, None)
            self._object = None if ref is None else self.composite.file[ref]

        return self._object

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        object_attribute: str | None = None,
        object_field: str | None = None,
        region_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            object_attribute: The attribute name with the object reference.
            object_field: The field the dtype that contains the object references.
            region_field: The field the dtype that contains the region references.
            **kwargs: Keyword arguments for inheritance.
        """
        if object_attribute is not None:
            self.object_attribute = object_attribute

        if object_field is not None:
            self.object_field = object_field

        if region_field is not None:
            self.region_field = region_field

        super().construct(composite=composite, **kwargs)

    def generate_region_reference(
        self,
        region: int | tuple | slice | h5py.RegionReference,
        object_: h5py.Dataset | h5py.Reference | None = None,
    ) -> tuple:
        """Creates reference objects for a given HDF5 object and slicing for that HDF5 object.

        Args:
            region: The region of dataset to make a reference for.
            object_: The HDF5 object to make a reference for.

        Returns:
            The object and region references.
        """
        if object_ is None:
            object_ = self.object

        if not isinstance(object_, h5py.Reference):
            object_rf = object_.ref
            object_ = self.composite.file[object_]
        else:
            object_ref = object_.ref

        if not isinstance(region, h5py.RegionReference):
            region = object_.regionref[region]

        return object_ref, region

    def get_object_reference(self, index: int | tuple | None = None) -> Any:
        """Gets the object reference.

        Args:
            index: The index in the dataset to get the reference from.

        Returns:
            The object reference.
        """
        if self.object_field is None:
            return self.composite.attributes[self.object_attribute]
        else:
            return self.composite[self.object_field][index]

    def get_object(self, index: int | tuple | None = None) -> Any:
        """Gets the HDF5 object referenced.

        Args:
            index: The index in the dataset to get the reference from.

        Returns:
            The requested HDF5 object.
        """
        if self.object_field is None:
            return self.object
        else:
            return self.composite.file[self.composite[self.object_field][index]]

    def get_region_reference(self, index: int | tuple) -> tuple:
        """Gets the region reference at a given index in the dataset.

        Args:
            index: The index in the dataset to get the reference from

        Returns:
            The object and region references.
        """
        region_ref = self.composite[self.region_field][index]

        if self.object_field is None:
            return self.composite.attributes[self.object_attribute], region_ref
        else:
            return (
                self.composite.file[self.composite[self.object_field][index]],
                region_ref,
            )

    def set_region_reference(
        self,
        index: int | tuple,
        region: int | tuple | slice | h5py.Reference,
        object_: h5py.Dataset | h5py.Reference | None = None,
    ) -> None:
        """Sets a region reference at a given index.

        Args:
            index: The index in the dataset to set the region reference to.
            region: The region to set, can be either the original slicing or the reference object.
            object_: The HDF5 object to set, can be either the HDF5 object or a reference to that object.
        """
        item = self.composite[index]
        item[self.composite.dtypes_dict[self.region_field]] = region
        if self.object_field is not None:
            if not isinstance(object_, h5py.Reference):
                object_ = object_.ref
            item[self.composite.dtypes_dict[self.object_field]] = object_
        elif object_ is not None:
            if isinstance(object_, h5py.Reference):
                self._object = None
                self.composite.attributes[self.object_attribute] = object_
            else:
                self.composite.attributes[self.object_attribute] = object_.ref
                self._object = object_

        self.composite[index] = item

    def get_from_reference(self, index: int | tuple) -> Any:
        """Get the item from the reference at the given index.

        Args:
            index: The index of the reference to get the item from.

        Returns:
            The item which the reference points to.
        """
        return self.get_object(index=index)[self.composite[self.region_field][index]]

    def get_from_reference_dict(self, index: int | tuple) -> dict:
        """Get the item from the reference at the given index as a dictionary.

        Args:
            index: The index of the reference to get the item from.

        Returns:
            The item which the reference points to as a dictionary.
        """
        return self.get_object(index=index).get_item_dict(
            self.composite[self.region_field][index]
        )

    def set_reference_to(self, index: int | tuple, value: Any) -> None:
        """Set the item referenced by the reference at the given index.

        Args:
            index: The index of the reference pointing to the item to set.
            value: The value to set the item to at the reference.
        """
        self.get_object(index=index)[self.composite[self.region_field][index]] = value

    def set_reference_to_dict(self, index: int | tuple, value: Any) -> None:
        """Set the item referenced by the reference at the given index to a dictionary of data.

        Args:
            index: The index of the reference pointing to the item to set.
            value: The dictionary to set the item to at the reference.
        """
        self.get_object(index=index).set_item_dict(
            self.composite[self.region_field][index], value
        )


class RegionReferenceAxisMap(AxisMap):
    """An outline which defines an HDF5Dataset as an Axis with region references for each datum."""

    default_attribute_names: Mapping[str, str] = {
        "object_reference": "object_reference"
    }
    default_kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,)}
    default_dtype = (("Region", h5py.ref_dtype),)
    default_component_types = {
        "axis": (RegionReferenceAxisComponent, {}),
    }
