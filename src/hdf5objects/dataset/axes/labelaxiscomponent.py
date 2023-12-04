"""labelaxiscomponent.py
A component and map for a HDF5Dataset which defines it as an axis that represents channel labels.
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
class LabelAxisComponent(AxisComponent):
    """A component for a HDF5Dataset which defines it as an axis that represents channel label."""

    @property
    def channels(self) -> np.ndarray:
        """Returns all the channels of this axis.

        Returns:
            All the channel labels.
        """
        try:
            return self.composite.get_all_data.caching_call()
        except AttributeError:
            return self.composite.get_all_data()

    @property
    def complete_labels(self) -> np.ndarray:
        """Returns all the complete channel names of this axis.

        Returns:
            All the channel labels.
        """
        return self.get_complete_labels()

    @property
    def sensors(self) -> np.ndarray:
        """Returns all the unique sensor in this axis.

        Returns:
            All the sensors.
        """
        return self.get_sensors()

    def get_channels(self) -> np.ndarray:
        """Returns all the channels of this axis.

        Returns:
            All the channel labels.
        """
        return self.composite.get_all_data()

    def get_complete_labels(self) -> np.ndarray:
        """Returns all the complete channel names of this axis.

        Returns:
            All complete channel labels.
        """
        return np.char.add(*self.channels.astype(str).T)

    def get_sensors(self) -> np.ndarray:
        """Returns all the unique sensors in this axis.

        Returns:
            All the sensors.
        """
        return np.unique(self.channels[:,0])

    def group_channels_by_sensor(self) -> dict:
        """Returns a dictionary all the unique sensors in this axis.

        Returns:
            All the sensors.
        """
        return dict([(sensor, np.flatnonzero(self.channels[:,0] == sensor)) for sensor in self.sensors])


class LabelAxisMap(AxisMap):
    """An outline which defines an HDF5Dataset as an Axis that represents channel label."""

    default_kwargs: dict[str, Any] = {"shape": (0,2), "maxshape": (None,2), "dtype": h5py.special_dtype(vlen=str)}
    default_component_types = {
        "axis": (LabelAxisComponent, {}),
    }
