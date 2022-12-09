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
    """

    Class Attributes:

    """
    _pass_types = int | float | bytes
    pass_types = {int, float, bytes}
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
        return item.timestamp()

    @classmethod
    def from_timezone(cls, item: datetime.tzinfo) -> float:
        return timezone_offset(item).total_seconds()

    @classmethod
    def from_timedelta(cls, item: datetime.timedelta) -> float:
        return item.total_seconds()

    @classmethod
    def from_uuid(cls, item: uuid.UUID) -> bytes:
        return item.bytes
    
    @singlekwargdispatchmethod("item")
    @classmethod
    def cast_from(cls, item: Any) -> Any:
        if isinstance(item, np.dtype):
            return item
        else:
            raise TypeError(f"{type(item)} does not have a cast method from.")

    @cast_from.register(str)
    @cast_from.register(int)
    @cast_from.register(float)
    @classmethod
    def _cast_from(cls, item: _pass_types) -> _pass_types:
        return item

    @cast_from.register
    @classmethod
    def _cast_from(cls, item: datetime.datetime) -> float:
        return cls.from_datetime(item)

    @cast_from.register
    @classmethod
    def _cast_from(cls, item: datetime.tzinfo) -> float:
        return cls.from_timezone(item)

    @cast_from.register
    @classmethod
    def _cast_from(cls, item: datetime.timedelta) -> float:
        return cls.from_timedelta(item)

    @cast_from.register
    @classmethod
    def _cast_from(cls, item: uuid.UUID) -> bytes:
        return cls.from_uuid(item)

    # Casting To
    @classmethod
    def to_pass(cls, item: _pass_types) -> _pass_types:
        return item

    @classmethod
    def to_str(cls, item: bytes) -> str:
        return item.decode(encoding='utf-8')

    @classmethod
    def to_datetime(cls, item: float, tzinfo: datetime.tzinfo | None = None) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(item, tzinfo)

    @classmethod
    def to_timezone(cls, item: float) -> datetime.tzinfo:
        return datetime.timezone(datetime.timedelta(seconds=item))

    @classmethod
    def to_timedelta(cls, item: float) -> datetime.timedelta:
        return datetime.timedelta(seconds=item)

    @classmethod
    def to_uuid(cls, item: bytes) -> uuid.UUID:
        return uuid.UUID(bytes=item)

    @classmethod
    def cast_to(cls, type_: type, item: Any, **kwargs: Any) -> Any:
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
