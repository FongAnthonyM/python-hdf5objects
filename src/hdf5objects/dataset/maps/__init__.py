"""__init__.py
Genreric maps for HDF5Datasets
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
# Local Packages #
from .basetimeseriesmap import BaseTimeSeriesMap
from .electricalseriesmap import ElectricalSeriesMap
from .shapesmap import ShapesMap
