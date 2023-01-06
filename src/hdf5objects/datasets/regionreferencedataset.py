""" regionreferencedataset.py

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
from ...hdf5bases import HDF5Map, HDF5Dataset, DatasetMap


# Definitions #
# Classes #
class RegionReferenceMap(DatasetMap):
    """The map for a dataset with object and region references.
    
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
        name: The name of this map.
        type_: The type of the hdf5 object this map represents.
        attribute_names: The name map of python name vs hdf5 name of the attribute.
        attributes: The default values of the attributes of the represented hdf5 object.
        map_names: The name map of python name vs hdf5 name of the maps contained within this map.
        maps: The nested maps.
        parent: The parent of this map.
        dtype: The dtype of the dataset this object will map.
        casting_kwargs: The keyword arguments for the dtype casting.
        single_ref_fields: The single fields of the dtype that contain references.
        multiple_ref_fields: The multiple fields of the dtype that contain references.
        prime_ref_field: The name of the reference to get when the name is not given.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the object this map represents.
    """
    default_single_reference_fields: dict[str, tuple[str, str]] = dict()
    default_multiple_reference_fields: dict[str, tuple[str, str]] = dict()
    default_primary_reference_field: str | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        name: str | None = None,
        type_: type | None = None,
        attribute_names: Mapping[str, str] | None = None,
        attributes: Mapping[str, Any] | None = None,
        map_names: Mapping[str, str] | None = None,
        maps: Mapping[str, HDF5Map] | None = None,
        parent: str | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        single_ref_fields: dict[str, tuple[str, str]] | None = None,
        multiple_ref_fields: dict[str, tuple[str, str]] | None = None,
        prime_ref_field: str | None = None,
        init: bool = True,
        **kwargs: Any
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self.single_reference_fields: dict[str, tuple[str, str]] = self.default_single_reference_fields.copy()
        self.multiple_reference_fields: dict[str, tuple[str, str]] = self.default_multiple_reference_fields.copy()
        self.primary_reference_field: str | None = self.default_primary_reference_field

        # Object Construction #
        if init:
            name = name if name is not None else self.default_name
            parent = parent if parent is not None else self.default_parent
            self.construct(
                name=name,
                type_=type_,
                attribute_names=attribute_names,
                attributes=attributes,
                map_names=map_names,
                maps=maps,
                parent=parent,
                single_ref_fields=single_ref_fields,
                multiple_ref_fields=multiple_ref_fields,
                prime_ref_field=prime_ref_field,
                dtype=dtype,
                casting_kwargs=casting_kwargs,
                **kwargs,
            )
    
    # Instance Methods #
    # Constructors/Destructors        
    def construct(
        self,
        name: str | None = None,
        type_: type | None = None,
        attribute_names: Mapping[str, str] | None = None,
        attributes: Mapping[str, Any] | None = None,
        map_names: Mapping[str, str] | None = None,
        maps: Mapping[str, HDF5Map] | None = None,
        parent: str | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        single_ref_fields: dict[str, tuple[str, str]] | None = None,
        multiple_ref_fields: dict[str, tuple[str, str]] | None = None,
        prime_ref_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object, setting attributes and sets nested maps' parents.

        Args:
            name: The name of this map.
            type_: The type of the hdf5 object this map represents.
            attribute_names: The name map of python name vs hdf5 name of the attribute.
            attributes: The default values of the attributes of the represented hdf5 object.
            map_names: The name map of python name vs hdf5 name of the maps contained within this map.
            maps: The nested maps.
            parent: The parent of this map.
            dtype: The dtype of the dataset this object will map.
            casting_kwargs: The keyword arguments for the dtype casting.
            single_ref_fields: The single fields of the dtype that contain references.
            multiple_ref_fields: The multiple fields of the dtype that contain references.
            prime_ref_field: The name of the reference to get when the name is not given.
            **kwargs: The keyword arguments for the object this map represents.
        """
        if single_ref_fields is not None:
            self.single_reference_fields.clear()
            self.single_reference_fields.update(single_ref_fields)
            
        if multiple_ref_fields is not None:
            self.multiple_reference_fields.clear()
            self.multiple_reference_fields.update(multiple_ref_fields)
            
        if prime_ref_field is not None:
            self.primary_reference_field = prime_ref_field
        
        super().construct(
            name=name,
            type_=type_,
            attribute_names=attribute_names,
            attributes=attributes,
            map_names=map_names,
            maps=maps,
            parent=parent,
            dtype=dtype,
            casting_kwargs=casting_kwargs,
            **kwargs,
        )


class RegionReferenceDataset(HDF5Dataset):
    """A dataset with object and region references.

    A full reference has two parts, an HDF5 object reference and a region reference to index within the HDF5 object.
    Common scenarios are to have a single object with many region references or many objects with many region
    references. It would be inefficient to repeatedly store the same object for every entry if we know every entry will
    refer to the same object, but with different regions, therefore there are two methods to store full references.
    The Single method only has one field in the dataset dtype which is the region reference while the object reference
    is stored as an attribute. The Multiple method has two fields in the dtype which will contain the object reference
    and the region reference.

    single_reference_fields: {reference_name: (object_attribute_name, region_field_name)}
    multiple_reference_fields: {reference_name: (object_field_name, region_field_name)}

    Args:
        data: The data to fill in this dataset.
        dataset: The HDF5 dataset to build this dataset around.
        name: The HDF5 name of this object.
        map_: The map for this HDF5 object.
        file: The file object that this dataset object originates from.
        load: Determines if this object will load the dataset from the file on construction.
        require: Determines if this object will create and fill the dataset in the file on construction.
        parent: The HDF5 name of the parent of this HDF5 object.
        dtype: The dtype of this dataset.
        casting_kwargs: The keyword arguments for casting HDF5 dtypes to python types.
        single_ref_fields: The single fields of the dtype that contain references.
        multiple_ref_fields: The multiple fields of the dtype that contain references.
        prime_ref_field: The name of the reference to get when the name is not given.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments to construct the base HDF5 dataset.
    """
    # Magic Methods
    # Constructors/Destructors
    @property
    def single_reference_fields(self) -> dict[str, tuple[str, str]]:
        """The single fields of the dtype that contain references."""
        return self.map.single_reference_fields
    
    @single_reference_fields.setter
    def single_reference_fields(self, value: dict[str, tuple[str, str]]) -> None:
        self.map.single_reference_fields = value

    @property
    def multiple_reference_fields(self) -> dict[str, tuple[str, str]]:
        """The multiple fields of the dtype that contain references."""
        return self.map.multiple_reference_fields

    @multiple_reference_fields.setter
    def multiple_reference_fields(self, value: dict[str, tuple[str, str]]) -> None:
        self.map.multiple_reference_fields = value

    @property
    def primary_reference_field(self) -> str:
        """The primary fields of the dtype that contain references."""
        return self.map.primary_reference_field

    @primary_reference_field.setter
    def primary_reference_field(self, value: str) -> None:
        self.map.primary_reference_field = value

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        dtype: np.dtype | str | tuple[tuple[str, type]] | None = None,
        single_ref_fields: dict[str, tuple[str, str]] | None = None,
        multiple_ref_fields: dict[str, tuple[str, str]] | None = None,
        prime_ref_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The data to fill in this dataset.
            dtype: The dtype of this dataset.
            single_ref_fields: The single fields of the dtype that contain references.
            multiple_ref_fields: The multiple fields of the dtype that contain references.
            prime_ref_field: The name of the reference to get when the name is not given.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.
        """
        super().construct(data=data, dtype=dtype, **kwargs)

        if single_ref_fields is not None:
            self.single_reference_fields.clear()
            self.single_reference_fields.update(single_ref_fields)

        if multiple_ref_fields is not None:
            self.multiple_reference_fields.clear()
            self.multiple_reference_fields.update(multiple_ref_fields)

        if prime_ref_field is not None:
            self.primary_reference_field = prime_ref_field

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
            object_ = self.attributes[full_ref[0]]
        elif not isinstance(object_, h5py.Reference):
            object_ = object_.ref

        if not isinstance(region, h5py.RegionReference):
            region = self.file[object_].regionref[region]

        return object_, region

    def get_region_reference(self, index: int | tuple, ref_name: str | None = None) -> tuple:
        """Gets the region reference at a given index in the dataset.

        Args:
            index: The index in the dataset to get the reference from
            ref_name: The name of the type of reference to get.

        Returns:
            The object and region references.
        """
        if ref_name is None:
            ref_name = self.primary_reference_field

        item = self[index]
        full_ref = self.single_reference_fields.get(ref_name, search_sentinel)
        if full_ref is search_sentinel:
            full_ref = self.multiple_reference_fields[ref_name]
            object_ref = item[self.dtypes_dict[full_ref[0]]]
        else:
            object_ref = self.attributes[full_ref[0]]

        region_ref = item[self.dtypes_dict[full_ref[1]]]
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
            object_ = self.attributes[full_ref[0]]
            was_set = True
        elif not isinstance(object_, h5py.Reference):
            object_ = object_.ref

        if not isinstance(region, h5py.Reference):
            region = self.file[object_].regionref[region]

        item = self[index]
        if full_ref is search_sentinel:
            full_ref = self.multiple_reference_fields[ref_name]
            item[self.dtypes_dict[full_ref[0]]] = object_
        elif not was_set:
            self.attributes[full_ref[0]] = object_

        item[self.dtypes_dict[full_ref[1]]] = region

        self[index] = item

    def get_from_reference(self, index: int | tuple, ref_name: str | None = None) -> Any:
        """Get the item from the reference at the given index.

        Args:
            index: The index of the reference to get the item from.
            ref_name: The name of the type of reference to get the item from.

        Returns:
            The item which the reference points to.
        """
        object_ref, region_ref = self.get_region_reference(index=index, ref_name=ref_name)
        return self.file[object_ref][region_ref]

    def get_from_reference_dict(self, index: int | tuple, ref_name: str | None = None) -> dict:
        """Get the item from the reference at the given index as a dictionary.

        Args:
            index: The index of the reference to get the item from.
            ref_name: The name of the type of reference to get the item from.

        Returns:
            The item which the reference points to as a dictionary.
        """
        object_ref, region_ref = self.get_region_reference(index=index, ref_name=ref_name)
        return self.file[object_ref].get_item_dict(region_ref)

    def set_reference_to(self, index: int | tuple, value: Any, ref_name: str | None = None) -> None:
        """Set the item referenced by the reference at the given index.

        Args:
            index: The index of the reference pointing to the item to set.
            value: The value to set the item to at the reference.
            ref_name: The name of the type of reference to get the item from.
        """
        object_ref, region_ref = self.get_region_reference(index=index, ref_name=ref_name)
        self.file[object_ref][region_ref] = value

    def set_reference_to_dict(self, index: int | tuple, value: Any, ref_name: str | None = None) -> None:
        """Set the item referenced by the reference at the given index to a dictionary of data.

        Args:
            index: The index of the reference pointing to the item to set.
            value: The dictionary to set the item to at the reference.
            ref_name: The name of the type of reference to get the item from.
        """
        object_ref, region_ref = self.get_region_reference(index=index, ref_name=ref_name)
        self.file[object_ref].set_item_dict(region_ref, value)


# Assign Cyclic Definitions
RegionReferenceMap.default_type = RegionReferenceDataset
