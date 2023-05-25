""" hdf5basecomponent.py

"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from typing import Any

# Third-Party Packages #
from baseobjects.composition import BaseComponent

# Local Packages #


# Definitions #
# Classes #
class HDF5BaseComponent(BaseComponent):
    """The base implementation for a component for an HDF5 Object."""
    # Instance Methods #
    # Data
    def create_component(self, **kwargs: Any) -> None:
        """Creates all the required parts of the dataset for this component, errors if a part already exists.

        Args:
            **kwargs: The keyword arguments to create this component.
        """
        pass

    def require_component(self, **kwargs: Any) -> None:
        """Creates all the required parts of the dataset for this component if it does not exists.

        Args:
            **kwargs: The keyword arguments to require this component.
        """
        pass
