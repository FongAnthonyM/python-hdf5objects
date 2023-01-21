""" sampleaxis.py
An Axis that represents the samples of a signal.
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
from .axis import AxisMap, AxisComponent


# Definitions #
# Classes #
class SampleAxisComponent(AxisComponent):
    """An Axis that represents the samples of a signal."""
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
    """A map for the SampleAxis object."""
    default_kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,), "dtype": "i"}
    default_component_types = {
        "axis": (SampleAxisComponent, {}),
    }
