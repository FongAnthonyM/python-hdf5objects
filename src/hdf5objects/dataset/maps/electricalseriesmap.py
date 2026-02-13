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
from .basetimeseriesmap import BaseTimeSeriesMap
from ..axes import LabelAxisMap
from ..axes import CoordinateAxisMap
from ..components import GeometryComponent

# Definitions #
# Classes #
class ElectricalSeriesMap(BaseTimeSeriesMap):
    """A base outline which defines a time series and its methods."""

    default_attributes: Mapping[str, Any] = BaseTimeSeriesMap.default_attributes | {"units": "volts"}
    default_axis_maps: list[dict[str, Any], ...] = [BaseTimeSeriesMap.default_axis_maps[0], {"channellabel_axis": LabelAxisMap(), "channelcoord_axis": CoordinateAxisMap()}]
    default_component_types: dict[str, Any] = BaseTimeSeriesMap.default_component_types | {"geometry": (GeometryComponent, {"label_scale_name": "channellabel_axis", "coordinate_scale_name": "channelcoord_axis"})}
