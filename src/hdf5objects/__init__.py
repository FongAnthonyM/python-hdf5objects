#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" __init__.py
Description:
"""
# Package Header #
from .__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Local Packages #
from .hdf5objects import HDF5Map, HDF5BaseObject, HDF5Attributes, HDF5Group, HDF5Dataset, HDF5File
from .fileobjects import *
