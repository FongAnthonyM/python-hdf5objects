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
from bidict import bidict
import h5py
import numpy as np

# Local Packages #
from .multitypedataset import MultiTypeDataset


# Definitions #
# Classes #
class RegionReferenceDataset(MultiTypeDataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_region_reference_fields: dict[str, tuple[str, str]] = dict()
    default_primary_reference_field: str | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        data: np.ndarray | None = None,
        types: tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        region_fields: Iterable[tuple[str, str]] | None = None,
        load: bool = False,
        require: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self.region_reference_fields: dict[str, tuple[str, str]] = self.default_region_reference_fields.copy()
        self.primary_reference_field: str | None = self. default_primary_reference_field

        # Object Construction #
        if init:
            self.construct(
                data=data,
                types=types,
                casting_kwargs=casting_kwargs,
                region_fields=region_fields,
                load=load,
                require=require,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        types: tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        region_fields: dict[str, tuple[str, str]] | None = None,
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
        if region_fields is not None:
            self.region_reference_fields.clear()
            self.region_reference_fields.update(region_fields)

        super().construct(data=data, types=types, casting_kwargs=casting_kwargs, load=load, require=require, **kwargs)

    def get_region_reference(self, index: int | tuple, ref_name: str | None = None):
        if ref_name is None:
            ref_name = self.region_reference_fields

        item = self[index]
        ds_ref = item[self._types_dict[self.region_reference_fields[ref_name][0]]]
        sl_ref = item[self._types_dict[self.region_reference_fields[ref_name][1]]]
        return ds_ref, sl_ref

    def set_region_reference(
        self,
        index: int | tuple,
        ds: h5py.Dataset | h5py.Reference,
        sl: int | tuple | slice | h5py.Reference,
        ref_name: str | None = None,
    ) -> None:
        if ref_name is None:
            ref_name = self.region_reference_fields

        if not isinstance(ds, h5py.ref_dtype):
            ds = ds.ref

        if not isinstance(sl, h5py.ref_dtype):
            sl = self._file[ds].regionref[sl]

        item = self[index]
        item[self._types_dict[self.region_reference_fields[ref_name][0]]] = ds
        item[self._types_dict[self.region_reference_fields[ref_name][1]]] = sl
        self[index] = item

    def get_from_reference(self, index: int | tuple, ref_name: str | None = None):
        ds_ref, sl_ref = self.get_region_reference(index=index, ref_name=ref_name)
        return self._file[ds_ref][sl_ref]

    def set_to_reference(self, index: int | tuple, value: Any, ref_name: str | None = None):
        ds_ref, sl_ref = self.get_region_reference(index=index, ref_name=ref_name)
        self._file[ds_ref][sl_ref] = value
