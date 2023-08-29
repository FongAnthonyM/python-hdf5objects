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
from ..axes import TimeAxisMap
from ..axes import LabelAxisMap
from ..axes import CoordinateAxisMap
from ..components import TimeSeriesComponent


# Definitions #
# Classes #
class BaseTimeSeriesMap(DatasetMap):
    """A base outline which defines a time series and its methods."""

    default_attribute_names: Mapping[str, str] = {"t_axis": "t_axis", "c_axis": "c_axis"}
    default_attributes: Mapping[str, Any] = {"t_axis": 0, "c_axis": 1}
    default_axis_maps: list[dict[str, Any], ...] = [{"time_axis": TimeAxisMap()}, {"label_axis": LabelAxisMap(), "coordinate_axis": CoordinateAxisMap()}]
    default_component_types = {"timeseries": (TimeSeriesComponent, {"scale_name": "time_axis"})}
