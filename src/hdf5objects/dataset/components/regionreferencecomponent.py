"""regionreferencedataset.py
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
from typing import Any

# Third-Party Packages #
from baseobjects import search_sentinel
import h5py

# Local Packages #
from ..basedatasetcomponent import BaseDatasetComponent


# Definitions #
# Classes #
class RegionReferenceComponent(BaseDatasetComponent):
    """A component for a HDF5Dataset which implement methods for object and region references.

    A full reference has two parts, an HDF5 object reference and a region reference to index within the HDF5 object.
    Common scenarios are to have a single object with many region references or many objects with many region
    references. It would be inefficient to repeatedly store the same object for every entry if we know every entry will
    refer to the same object, but with different regions, therefore there are two methods to store full references.
    The Single method only has one field in the dataset dtype which is the region reference while the object reference
    is stored as an attribute. The Multiple method has two fields in the dtype which will contain the object reference
    and the region reference.

    single_reference_fields: {reference_name: (object_attribute_name, region_field_name)}
    multiple_reference_fields: {reference_name: (object_field_name, region_field_name)}

    Class Attributes:
        default_single_reference_fields: The default single fields of the dtype that contain references.
        default_multiple_reference_fields: The default multiple fields of the dtype that contain references.
        default_primary_reference_field: The default name of the reference to get when the name is not given.

    Attributes:
        single_reference_fields: The single fields of the dtype that contain references.
        multiple_reference_fields: The multiple fields of the dtype that contain references.
        primary_reference_field: The name of the reference to get when the name is not given.

    Args:
        composite: The object which this object is a component of.
        single_reference_fields: The single fields of the dtype that contain references.
        multiple_reference_fields: The multiple fields of the dtype that contain references.
        primary_reference_field: The name of the reference to get when the name is not given.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    default_single_reference_fields: dict[str, tuple[str, str]] = dict()
    default_multiple_reference_fields: dict[str, tuple[str, str]] = dict()
    default_primary_reference_field: str | None = None

    # Magic Methods
    # Constructors/Destructors
    def __init__(
        self,
        composite: Any = None,
        single_reference_fields: dict[str, tuple[str, str]] | None = None,
        multiple_reference_fields: dict[str, tuple[str, str]] | None = None,
        primary_reference_field: str | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.single_reference_fields: dict[
            str, tuple[str, str]
        ] = self.default_single_reference_fields.copy()
        self.multiple_reference_fields: dict[
            str, tuple[str, str]
        ] = self.default_multiple_reference_fields.copy()
        self.primary_reference_field: str | None = self.default_primary_reference_field

        # Parent Attributes #
        super().__init__(self, init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                single_reference_fields=single_reference_fields,
                multiple_reference_fields=multiple_reference_fields,
                primary_reference_field=primary_reference_field,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        single_reference_fields: dict[str, tuple[str, str]] | None = None,
        multiple_reference_fields: dict[str, tuple[str, str]] | None = None,
        primary_reference_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            single_reference_fields: The single fields of the dtype that contain references.
            multiple_reference_fields: The multiple fields of the dtype that contain references.
            primary_reference_field: The name of the reference to get when the name is not given.
            **kwargs: Keyword arguments for inheritance.
        """
        if single_reference_fields is not None:
            self.single_reference_fields.clear()
            self.single_reference_fields.update(single_reference_fields)

        if multiple_reference_fields is not None:
            self.multiple_reference_fields.clear()
            self.multiple_reference_fields.update(multiple_reference_fields)

        if primary_reference_field is not None:
            self.primary_reference_field = primary_reference_field

        super().construct(composite=composite, **kwargs)

    def generate_region_reference(
        self,
        region: int | tuple | slice | h5py.RegionReference,
        object_: h5py.Dataset | h5py.Reference | None = None,
        ref_name: str | None = None,
    ) -> tuple:
        """Creates reference objects for a given HDF5 object and slicing for that HDF5 object.

        Args:
            region: The region of dataset to make a reference for.
            object_: The HDF5 object to make a reference for.
            ref_name: The name of the type reference that the reference objects will be for.

        Returns:
            The object and region references.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        full_ref = self.single_reference_fields.get(ref_name, search_sentinel)
        if object_ is None and full_ref is not search_sentinel:
            object_ = self.composite.attributes[full_ref[0]]
        elif not isinstance(object_, h5py.Reference):
            object_ = object_.ref

        if not isinstance(region, h5py.RegionReference):
            region = self.composite.file[object_].regionref[region]

        return object_, region

    def get_object_reference(
        self, index: int | tuple | None = None, ref_name: str | None = None
    ) -> Any:
        """Gets the object reference.

        Args:
            index: The index in the dataset to get the reference from.
            ref_name: The name of the type of reference to get.

        Returns:
            The object reference.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        object_ref_name = self.single_reference_fields.get(ref_name, [None])[0]
        if object_ref_name is None:
            object_ref_name = self.multiple_reference_fields[ref_name][0]
            return self.composite[index][self.composite.dtypes_dict[object_ref_name]]
        else:
            return self.composite.attributes[object_ref_name]

    def set_object_reference(
        self,
        object_: h5py.Dataset | h5py.Reference | None,
        index: int | tuple | None = None,
        ref_name: str | None = None,
    ) -> Any:
        """Sets the object reference.

        Args:
            object_: The HDF5 object to set, can be either the HDF5 object or a reference to that object.
            index: The index in the dataset to get the reference from.
            ref_name: The name of the type of reference to get.

        Returns:
            The object reference.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        if not isinstance(object_, h5py.Reference) and not None:
            object_ = object_.ref

        object_ref_name = self.single_reference_fields.get(ref_name, [None])[0]
        if object_ref_name is None:
            object_ref_name = self.multiple_reference_fields[ref_name][0]
            self.composite[index][self.composite.dtypes_dict[object_ref_name]] = object_
        else:
            self.composite.attributes[object_ref_name] = object_

    def get_object(
        self, index: int | tuple | None = None, ref_name: str | None = None
    ) -> Any:
        """Gets the HDF5 object referenced.

        Args:
            index: The index in the dataset to get the reference from.
            ref_name: The name of the type of reference to get.

        Returns:
            The requested HDF5 object.
        """
        ref = self.get_object_reference(index=index, ref_name=ref_name)
        return self.composite.file[ref] if ref else None

    def get_region_reference(
        self, index: int | tuple, ref_name: str | None = None
    ) -> tuple:
        """Gets the region reference at a given index in the dataset.

        Args:
            index: The index in the dataset to get the reference from
            ref_name: The name of the type of reference to get.

        Returns:
            The object and region references.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        item = self.composite[index]
        full_ref = self.single_reference_fields.get(ref_name, search_sentinel)
        if full_ref is search_sentinel:
            full_ref = self.multiple_reference_fields[ref_name]
            object_ref = item[self.composite.dtypes_dict[full_ref[0]]]
        else:
            object_ref = self.composite.attributes[full_ref[0]]

        region_ref = item[self.composite.dtypes_dict[full_ref[1]]]
        return object_ref, region_ref

    def set_region_reference(
        self,
        index: int | tuple,
        region: int | tuple | slice | h5py.Reference,
        object_: h5py.Dataset | h5py.Reference | None = None,
        ref_name: str | None = None,
    ) -> None:
        """Sets a region reference at a given index.

        Args:
            index: The index in the dataset to set the region reference to.
            region: The region to set, can be either the original slicing or the reference object.
            object_: The HDF5 object to set, can be either the HDF5 object or a reference to that object.
            ref_name: The name of the type of reference to set.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        was_set = False
        full_ref = self.single_reference_fields.get(ref_name, search_sentinel)
        if object_ is None and full_ref is not search_sentinel:
            object_ = self.composite.attributes[full_ref[0]]
            was_set = True
        elif not isinstance(object_, h5py.Reference):
            object_ = object_.ref

        if not isinstance(region, h5py.Reference):
            region = self.composite.file[object_].regionref[region]

        item = self.composite[index]
        if full_ref is search_sentinel:
            full_ref = self.multiple_reference_fields[ref_name]
            item[self.composite.dtypes_dict[full_ref[0]]] = object_
        elif not was_set:
            self.composite.attributes[full_ref[0]] = object_

        item[self.composite.dtypes_dict[full_ref[1]]] = region

        self.composite[index] = item

    def get_from_reference(
        self, index: int | tuple, ref_name: str | None = None
    ) -> Any:
        """Get the item from the reference at the given index.

        Args:
            index: The index of the reference to get the item from.
            ref_name: The name of the type of reference to get the item from.

        Returns:
            The item which the reference points to.
        """
        object_ref, region_ref = self.get_region_reference(
            index=index, ref_name=ref_name
        )
        return self.composite.file[object_ref][region_ref]

    def get_from_reference_dict(
        self, index: int | tuple, ref_name: str | None = None
    ) -> dict:
        """Get the item from the reference at the given index as a dictionary.

        Args:
            index: The index of the reference to get the item from.
            ref_name: The name of the type of reference to get the item from.

        Returns:
            The item which the reference points to as a dictionary.
        """
        object_ref, region_ref = self.get_region_reference(
            index=index, ref_name=ref_name
        )
        return self.composite.file[object_ref].get_item_dict(region_ref)

    def set_reference_to(
        self, index: int | tuple, value: Any, ref_name: str | None = None
    ) -> None:
        """Set the item referenced by the reference at the given index.

        Args:
            index: The index of the reference pointing to the item to set.
            value: The value to set the item to at the reference.
            ref_name: The name of the type of reference to get the item from.
        """
        object_ref, region_ref = self.get_region_reference(
            index=index, ref_name=ref_name
        )
        self.composite.file[object_ref][region_ref] = value

    def set_reference_to_dict(
        self, index: int | tuple, value: Any, ref_name: str | None = None
    ) -> None:
        """Set the item referenced by the reference at the given index to a dictionary of data.

        Args:
            index: The index of the reference pointing to the item to set.
            value: The dictionary to set the item to at the reference.
            ref_name: The name of the type of reference to get the item from.
        """
        object_ref, region_ref = self.get_region_reference(
            index=index, ref_name=ref_name
        )
        self.composite.file[object_ref].set_item_dict(region_ref, value)
