""" linkedgroup.py

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
import pathlib
from typing import Any

# Third-Party Packages #
import h5py

# Local Packages #
from ..hdf5bases import HDF5Group, HDF5Dataset


# Definitions #
# Classes #
class LinkedGroup(HDF5Group):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        group: h5py.Group | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        build: bool = False,
        parent: str | None = None,
        init: bool = True,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #

        # Object Construction  #
        if init:
            self.construct(group=group, name=name, map_=map_, file=file, load=load, build=build, parent=parent)

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        group: h5py.Group | HDF5BaseObject | None = None,
        name: str | None = None,
        map_: HDF5Map | None = None,
        file: str | pathlib.Path | h5py.File | None = None,
        load: bool = False,
        build: bool = False,
        parent: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object from the provided arguments.

        Args:
            group: The HDF5 group to build this dataset around.
            name: The HDF5 name of this object.
            map_: The map for this HDF5 object.
            file: The file object that this group object originates from.
            load: Determines if this object will load the group from the file on construction.
            build: Determines if this object will create and fill the group in the file on construction.
            parent: The HDF5 name of the parent of this HDF5 object.
        """
        pass

    def get_reference_item_dict(self, name, index, ref_name=None):
        ds_ref, sl_ref = self[name].get_region_reference(index, ref_name)
