#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5dataset.py
Description:
"""
__author__ = "Anthony Fong"
__copyright__ = "Copyright 2021, Anthony Fong"
__credits__ = ["Anthony Fong"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Anthony Fong"
__email__ = ""
__status__ = "Prototype"

# Default Libraries #
import collections
import copy
import datetime
import pathlib
import time
import uuid
from warnings import warn

# Downloaded Libraries #
from baseobjects import BaseObject, DynamicWrapper, StaticWrapper
from bidict import bidict
from classversioning import VersionedClass, VersionType, TriNumberVersion
import h5py
import numpy as np

# Local Libraries #


# Todo: Adapt this to new style
# Definitions #
# Classes #

