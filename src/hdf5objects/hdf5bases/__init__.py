"""__init__.py
The base objects for HDF5 objects.
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
from .hdf5map import HDF5Map
from .hdf5basecomponent import HDF5BaseComponent
from .hdf5baseobject import HDF5BaseObject
from .hdf5attributes import HDF5Attributes
from .hdf5group import HDF5Group, GroupMap
from .hdf5dataset import HDF5Dataset, DatasetMap
from .hdf5file import HDF5File, FileMap
from .hdf5caster import HDF5Caster


# Assign Cyclic Definitions
HDF5BaseObject.file_type = HDF5File
HDF5BaseObject.default_map = HDF5Map()
