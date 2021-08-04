#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" xltekdayframe.py
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
import datetime

# Downloaded Libraries #
from ...studies.directorytimeframe.directorytimeframe import DirectoryTimeFrame

# Local Libraries #
from .hdf5xltekframe import HDF5XLTEKFrame


# Definitions #
# Classes #
# Todo: Maybe skip to XLTEKDayFrame
class XLTEKDayFrame(DirectoryTimeFrame):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_frame_type = HDF5XLTEKFrame

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, path=None, frames=None, update=True, init=True):
        super().__init__(init=False)

        self.glob_condition = "*.h5"

        if init:
            self.construct(path=path, frames=frames, update=update)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, path=None, frames=None, update=True):
        super().construct(path=path, frames=frames, update=update)

        if not self.frames:
            try:
                self.date_from_path()
            except (ValueError, IndexError):
                pass

    def date_from_path(self):
        date_string = self.path.parts[-1].split('_')[1]
        self._date = datetime.datetime.strptime(date_string, self.date_format).date()

