""" channelaxis.py
An Axis that represents channel number.
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
from .axis import AxisMap, Axis


# Definitions #
# Classes #
class ChannelAxisMap(AxisMap):
    """A map for the ChannelAxis object."""
    default_kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,), "dtype": "i"}


class ChannelAxis(Axis):
    """An Axis that represents channel number.

    Class_Attributes:
        default_scale_name: The default name of this axis.

    Attributes:
        default_kwargs: The default keyword arguments to use when creating the dataset.
        _scale_name: The scale name of this axis.

    Args:
        start: The start of the axis.
        stop: The end of the axis.
        step: The interval between each datum of the axis.
        rate: The frequency of the data of the axis.
        size: The number of datum in the axis.
        s_name: The name of the axis (scale).
        require: Determines if the axis should be created and filled.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the HDF5Dataset.
    """
    default_scale_name: str | None = "channel axis"

    @property
    def channels(self) -> np.ndarray:
        """Returns all the channels of this axis.

        Returns:
            All the channel numbers.
        """
        try:
            return self.get_all_data.caching_call()
        except AttributeError:
            return self.get_all_data()

    def get_channels(self) -> np.ndarray:
        """Returns all the channels of this axis.

        Returns:
            All the channel numbers.
        """
        return self.get_all_data()


# Assign Cyclic Definitions
ChannelAxisMap.default_type = ChannelAxis
ChannelAxis.default_map = ChannelAxisMap()
