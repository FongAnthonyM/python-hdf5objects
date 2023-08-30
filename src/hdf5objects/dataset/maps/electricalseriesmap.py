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
from .basetimeseriesmap import BaseTimeSeriesMap
from ..axes import LabelAxisMap
from ..axes import CoordinateAxisMap

# Definitions #
# Classes #
class ElectricalSeriesMap(BaseTimeSeriesMap):
    """A base outline which defines a time series and its methods."""

    default_attributes: Mapping[str, Any] = BaseTimeSeriesMap.default_attributes | {"units": "volts"}
    default_axis_maps: list[dict[str, Any], ...] = [BaseTimeSeriesMap.default_axis_maps[0], {"channellabel_axis": LabelAxisMap(), "channelsensor_axis": LabelAxisMap(), "channelcoord_axis": CoordinateAxisMap()}]
