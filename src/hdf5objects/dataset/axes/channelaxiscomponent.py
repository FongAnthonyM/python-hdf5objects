""" channelaxiscomponent.py
A component and map for a HDF5Dataset which defines it as an axis that represents channel number.
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
from typing import Any

# Third-Party Packages #
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Map
from .axiscomponent import AxisMap, AxisComponent


# Definitions #
# Classes #
class ChannelAxisComponent(AxisComponent):
    """A component for a HDF5Dataset which defines it as an axis that represents channel number."""
    @property
    def channels(self) -> np.ndarray:
        """Returns all the channels of this axis.

        Returns:
            All the channel numbers.
        """
        try:
            return self.composite.get_all_data.caching_call()
        except AttributeError:
            return self.composite.get_all_data()

    def get_channels(self) -> np.ndarray:
        """Returns all the channels of this axis.

        Returns:
            All the channel numbers.
        """
        return self.composite.get_all_data()


class ChannelAxisMap(AxisMap):
    """An outline which defines an HDF5Dataset as an Axis that represents channel number."""
    default_attribute_names = {"map_type": "map_type"}
    default_attributes = {"map_type": "ChannelAxisMap"}
    default_kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,), "dtype": "i"}
    default_component_types = {
        "axis": (ChannelAxisComponent, {}),
    }
