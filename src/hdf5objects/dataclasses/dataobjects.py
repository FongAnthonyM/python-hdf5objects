""" dataobjects.py

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
import datetime
from typing import Any, NamedTuple

# Third-Party Packages #
import numpy as np

# Local Packages #


# Definitions #
# Classes #
# Index #
class IndexValue(NamedTuple):
    """A named tuple for returning an index with a value."""
    index: int | None
    value: Any


class IndexDateTime(NamedTuple):
    """A named tuple for returning an index with a datetime."""
    index: int | None
    datetime: datetime.datetime | None
    timestamp: float | None


# Found Ranges #
class FoundRange(NamedTuple):
    """A named tuple for returning a range with its start and end."""
    data: tuple[int | float] | np.ndarray | None
    start: int | None
    end: int | None


class FoundTimeRange(NamedTuple):
    """A name tuple for returning a range of times with its start and end."""
    data: tuple[datetime.datetime] | np.ndarray | None
    start: int | None
    end: int | None


class FoundData(NamedTuple):
    """A named tuple for returning a found data range."""
    data: np.ndarray | None
    axis: np.ndarray | None
    start: int | float | None
    end: int | float | None
    start_index: int | None
    end_index: int | None


__all__ = ["IndexValue", "IndexDateTime", "FoundRange", "FoundTimeRange", "FoundData"]
