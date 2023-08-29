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


class CoordinateAxisMap(AxisMap):
    """An outline which defines an HDF5Dataset as an Axis that represents channel coords."""

    default_kwargs: dict[str, Any] = {"shape": (0,3), "maxshape": (None,3), "dtype": float}
    default_component_types = {
        "axis": (CoordinateAxisComponent, {}),
    }
