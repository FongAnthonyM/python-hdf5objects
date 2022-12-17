""" hdf5caster.py
A class that contains methods for casting data types to type that can be stored in an HDF5 file.
"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import datetime as datetime
import uuid as uuid
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject, singlekwargdispatchmethod, search_sentinel
from baseobjects.operations import timezone_offset
import numpy as np
import h5py

# Local Packages #


# Definitions #
STRING_TYPE = h5py.string_dtype(encoding='utf-8')


# Classes #
class HDF5Caster(BaseObject):
    """A class that casts python to and from types that can be stored in an HDF5 file.

    Class Attributes:
        _pass_types: A type union of the types to return with no modification.
        pass_types: The types to return with no modification.
        type_map: A map of python types and their HDF5 representation.
        to_registry: A map of methods to use to cast an HDF5 representation to its correct python type.
    """
    _pass_types = int | float | bytes | np.dtype | h5py.Reference
    pass_types = {int, float, bytes, np.dtype, h5py.Reference}
    type_map = {
        bytes: STRING_TYPE,
        str: STRING_TYPE,
        datetime.datetime: float,
        datetime.tzinfo: float,
        datetime.timedelta: float,
        uuid.UUID: STRING_TYPE,
    }

    # Class Methods #
    # Map Type
    @classmethod
    def map_type(cls, type_: Any) -> Any:
        """Gets the HDF5 representation of the given type_.

        Args:
            type_: The type to get an HDF5 representation of.

        Returns:
            The type which is an HDF5 representation of the given type.
        """
        if type_ in cls.pass_types:
            return type_

        new_type = cls.type_map.get(type_, search_sentinel)
        if new_type is not search_sentinel:
            return new_type
        elif isinstance(type_, np.dtype):
            return type_

    # Casting From
    @classmethod
    def from_datetime(cls, item: datetime.datetime) -> float:
        """Casts a datetime to a type that can be stored in an HDF5."""
        return item.timestamp()

    @classmethod
    def from_timezone(cls, item: datetime.tzinfo) -> float:
        """Casts a timezone to a type that can be stored in an HDF5."""
        return timezone_offset(item).total_seconds()

    @classmethod
    def from_timedelta(cls, item: datetime.timedelta) -> float:
        """Casts a timedelta to a type that can be stored in an HDF5."""
        return item.total_seconds()

    @classmethod
    def from_uuid(cls, item: uuid.UUID) -> bytes:
        """Casts a UUID to a type that can be stored in an HDF5."""
        return item.hex
    
    @singlekwargdispatchmethod("item")
    @classmethod
    def cast_from(cls, item: Any) -> Any:
        """Casts an item to a type that can be stored in an HDF5.

        Args:
            item: The item to cast a type that can be stored in and HDF5 object.

        Returns:
            The item as the new type.
        """
        if isinstance(item, np.dtype):
            return item
        else:
            raise TypeError(f"{type(item)} does not have a cast method from.")

    @cast_from.register(str)
    @cast_from.register(int)
    @cast_from.register(float)
    @cast_from.register(np.dtype)
    @cast_from.register(h5py.Reference)
    @classmethod
    def _cast_from(cls, item: _pass_types) -> _pass_types:
        """Returns the item because it does not need to cast to another type."""
        return item

    @cast_from.register
    @classmethod
    def _cast_from(cls, item: datetime.datetime) -> float:
        """Casts a datetime to a type that can be stored in an HDF5."""
        return cls.from_datetime(item)

    @cast_from.register
    @classmethod
    def _cast_from(cls, item: datetime.tzinfo) -> float:
        """Casts a timezone to a type that can be stored in an HDF5."""
        return cls.from_timezone(item)

    @cast_from.register
    @classmethod
    def _cast_from(cls, item: datetime.timedelta) -> float:
        """Casts a timedelta to a type that can be stored in an HDF5."""
        return cls.from_timedelta(item)

    @cast_from.register
    @classmethod
    def _cast_from(cls, item: uuid.UUID) -> bytes:
        """Casts a UUID to a type that can be stored in an HDF5."""
        return cls.from_uuid(item)

    # Casting To
    @classmethod
    def to_pass(cls, item: _pass_types) -> _pass_types:
        """Returns the item without casting it to another type."""
        return item

    @classmethod
    def to_str(cls, item: bytes) -> str:
        """Casts an HDF5 representation to a string."""
        return item.decode(encoding='utf-8')

    @classmethod
    def to_datetime(cls, item: float, tzinfo: datetime.tzinfo | None = None) -> datetime.datetime:
        """Casts an HDF5 representation to a datetime."""
        return datetime.datetime.fromtimestamp(item, tzinfo)

    @classmethod
    def to_timezone(cls, item: float) -> datetime.tzinfo:
        """Casts an HDF5 representation to a tzinfo."""
        return datetime.timezone(datetime.timedelta(seconds=item))

    @classmethod
    def to_timedelta(cls, item: float) -> datetime.timedelta:
        """Casts an HDF5 representation to a timedelta."""
        return datetime.timedelta(seconds=item)

    @classmethod
    def to_uuid(cls, item: bytes) -> uuid.UUID:
        """Casts an HDF5 representation to an UUID."""
        return uuid.UUID(item.decode(encoding='utf-8'))

    @classmethod
    def cast_to(cls, type_: type, item: Any, **kwargs: Any) -> Any:
        """Casts an HDF5 representation to its python type.

        Args:
            type_: The type to cast the HDF5 item to.
            item: The HDF5 item to cast into a python type.
            **kwargs: Keyword argument to use to build the python object.

        Returns:
            The python object which the HDF5 represents.
        """
        to_method = cls.to_registry.get(type_, search_sentinel)
        if to_method is search_sentinel:
            raise TypeError(f"{type(item)} does not have a cast method to.")
        elif isinstance(type_, np.dtype):
            return item
        else:
            return to_method.__get__(cls, cls.__class__)(item, **kwargs)

    to_registry = {
        int: to_pass,
        float: to_pass,
        str: to_str,
        datetime.datetime: to_datetime,
        datetime.tzinfo: to_timezone,
        datetime.timedelta: to_timedelta,
        uuid.UUID: to_uuid,
    }
