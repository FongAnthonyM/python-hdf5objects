""" timereferencedataset.py

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
from collections.abc import Mapping, Iterable
import datetime
from decimal import Decimal
from typing import Any

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch
from dspobjects.dataclasses import FoundData
from framestructure import TimeSeriesContainer
import h5py
import numpy as np

# Local Packages #
from ..hdf5bases import HDF5Map, DatasetMap, HDF5Dataset
from .timeseriesdataset import TimeSeriesDataset, TimeSeriesMap
from .regionreferencedataset import RegionReferenceDataset, RegionReferenceMap


# Definitions #
# Classes #
class TimeReferenceMap(TimeSeriesMap, RegionReferenceMap):
    ...


class TimeReferenceDataset(TimeSeriesDataset, RegionReferenceDataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
