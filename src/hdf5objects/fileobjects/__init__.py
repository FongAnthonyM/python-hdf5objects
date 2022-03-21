""" __init__.py
General HDF5 objects for different file types.
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

# Third-Party Packages #

# Local Packages #
from .basehdf5 import BaseHDF5
from .hdf5eeg import HDF5EEG, HDF5EEGMap
