"""basedatasetcomponent.py
The base implementation for a component of a HDF5Dataset.
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
from collections.abc import Iterable
from typing import Any

# Third-Party Packages #

# Local Packages #
from ..hdf5bases import HDF5BaseComponent


# Definitions #
# Classes #
class BaseDatasetComponent(HDF5BaseComponent):
    """The base implementation for a component of a HDF5Dataset."""

    # Magic Methods #
    # Construction/Destruction
    @property
    def data(self) -> Any:
        """The data of the composite."""
        return self.composite

    @data.setter
    def data(self, value: Any) -> None:
        pass

    # Instance Methods #
    # Data
    def set_data_component(self, **kwargs: Any) -> None:
        """Sets the data pertaining to this component.

        Args:
            **kwargs: The keyword arguments for creating the component.
        """
        pass

    def append_component(self, **kwargs: Any) -> None:
        """Append data to this component.

        Args:
            **kwargs: The keyword arguments for appending data.
        """
        pass

    def insert_component(self, index: int | slice | Iterable[int], **kwargs: Any) -> None:
        """Inserts data to this component.

        Args:
            index: The index or slice to insert the data into.
            **kwargs: The keyword arguments for inserting data.
        """
        pass

    def delete_component(self, index: int | slice | Iterable[int], **kwargs: Any) -> None:
        """Deletes data from this component.

        Args:
            index: The index or slice to delete from the data.
            **kwargs: The keyword arguments for deleting data.
        """
        pass
