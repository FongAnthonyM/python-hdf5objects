"""__init__.py
Objects that expand on HDF5 and h5py.
"""
# Package Header #
from .header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Local Packages #
from .hdf5bases import *
from .fileobjects import *
from .dataset import AxisMap

# Assign Cyclic Definitions
HDF5Dataset.default_axis_map_type = AxisMap
