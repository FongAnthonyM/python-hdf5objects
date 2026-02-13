"""__init__.py
General HDF5 objects for different file types.
"""
# Header #
__package_name__ = "hdf5objects"

__author__ = "Anthony Fong"
__credits__ = ["Anthony Fong"]
__copyright__ = "Copyright 2021, Anthony Fong"
__license__ = "MIT"

__version__ = "0.6.0"
__maintainer__ = "Anthony Fong"
__email__ = """


# Imports #
# Standard Libraries #

# Third-Party Packages #

# Local Packages #
from .basehdf5 import BaseHDF5, BaseHDF5Map
from .hdf5eeg import HDF5EEG, HDF5EEGMap
