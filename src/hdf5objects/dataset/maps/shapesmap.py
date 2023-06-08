"""basetimeseriesmap.py
A base outline which defines a time series and its methods.
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

# Local Packages #
from ...hdf5bases import DatasetMap
from ..components import ShapesComponent


# Definitions #
# Classes #
class ShapesMap(DatasetMap):
    """An outline which contains shapes and its methods."""

    default_kwargs: dict[str, Any] = {
        "shape": (0, 0),
        "maxshape": (None, None),
        "dtype": "u8",
    }
    default_component_types = {"shapes": (ShapesComponent, {})}
