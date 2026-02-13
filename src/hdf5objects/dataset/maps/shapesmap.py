"""basetimeseriesmap.py
A base outline which defines a time series and its methods.
"""
# Header #
__package_name__ = "hdf5objects"

__author__ = "Anthony Fong"
__credits__ = ["Anthony Fong"]
__copyright__ = "Copyright 2021, Anthony Fong"
__license__ = "MIT"

__version__ = "0.6.0"
__maintainer__ = "Anthony Fong"
__email__ = """


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
