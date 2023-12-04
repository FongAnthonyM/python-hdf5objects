"""coordinateaxiscomponent.py
A component and map for a HDF5Dataset which defines it as an axis that represents channel coords.
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
import numpy as np
import h5py

# Local Packages #
from .axiscomponent import AxisMap, AxisComponent


# Definitions #
# Classes #
class CoordinateAxisComponent(AxisComponent):
    """A component for a HDF5Dataset which defines it as an axis that represents channel coords."""

    @property
    def channels(self) -> np.ndarray:
        """Returns all the channels of this axis.

        Returns:
            All the channel coordinates.
        """
        try:
            return self.composite.get_all_data.caching_call()
        except AttributeError:
            return self.composite.get_all_data()

    def get_channels(self) -> np.ndarray:
        """Returns all the channels of this axis.

        Returns:
            All the channel coordinates.
        """
        return self.composite.get_all_data()


    def calculate_channel_distance(self) -> np.ndarray:
        """
        Compute the Euclidean distance between channels using coordinates.

        Returns:
            Euclidean distances between channels.
        """

        # Get number of channels
        n_chan = len(self.channels)

        # Construct a meshgrid to efficiently compute all possible distances
        c_i, c_j = np.meshgrid(np.arange(n_chan), np.arange(n_chan))

        # Compute distances using Euclidean geometry
        channel_dist = np.sqrt(
            np.sum((self.channels[c_i, :] - self.channels[c_j, :])**2, axis=-1))

        return channel_dist

class CoordinateAxisMap(AxisMap):
    """An outline which defines an HDF5Dataset as an Axis that represents channel coords."""

    default_attribute_names: Mapping[str, str] = {
        "coordinate_system": "coordinate_system"
    }
    default_attributes: Mapping[str, Any] = {
        "coordinate_system": ""
    }
    default_kwargs: dict[str, Any] = {"shape": (0,3), "maxshape": (None,3), "dtype": float}
    default_component_types = {
        "axis": (CoordinateAxisComponent, {}),
    }
