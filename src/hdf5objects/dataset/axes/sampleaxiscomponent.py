""" sampleaxiscomponent.py
A component and map for a HDF5Dataset which defines it as an axis that represents samples of a singal.
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
class SampleAxisComponent(AxisComponent):
    """A component for a HDF5Dataset which defines it as an axis that represents samples of a singal."""
    @property
    def samples(self) -> np.ndarray:
        """Returns all the sample numbers of this axis.

        Returns:
            All the sample numbers.
        """
        try:
            return self.composite.get_all_data.caching_call()
        except AttributeError:
            return self.composite.get_all_data()

    def get_samples(self) -> np.ndarray:
        """Returns all the sample numbers of this axis.

        Returns:
            All the sample numbers.
        """
        return self.composite.get_all_data()


class SampleAxisMap(AxisMap):
    """An outline which defines an HDF5Dataset as an Axis that represents samples of a signal."""
    default_kwargs: dict[str, Any] = {"shape": (0, 0), "maxshape": (None, 1), "dtype": "i"}
    default_component_types = {
        "axis": (SampleAxisComponent, {}),
    }
