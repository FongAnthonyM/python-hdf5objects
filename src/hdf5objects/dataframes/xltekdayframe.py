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
from framestructure import DirectoryTimeFrame

# Local Libraries #
from .hdf5xltekframe import HDF5XLTEKFrame


# Definitions #
# Classes #
class XLTEKDayFrame(DirectoryTimeFrame):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_frame_type = HDF5XLTEKFrame

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, path=None, frames=None, update=True, open_=False, init=True, **kwargs):
        super().__init__(init=False)

        self.glob_condition = "*.h5"

        if init:
            self.construct(path=path, frames=frames, update=update, open_=open_, **kwargs)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, path=None, frames=None, update=True, open_=False, **kwargs):
        super().construct(path=path, frames=frames, update=update, open_=True, **kwargs)

        if not self.frames:
            try:
                self.date_from_path()
            except (ValueError, IndexError):
                pass

        if not open_:
            self.close()

    # Constructors/Destructors
    def construct_frames(self, **kwargs):
        for path in self.path.glob(self.glob_condition):
            if path not in self.frame_names:
                file_frame = self.frame_type.new_validated(path, **kwargs)
                if self.frame_creation_condition(file_frame):
                    self.frames.append(file_frame)
                    self.frame_names.add(path)
        self.frames.sort(key=lambda frame: frame.start)

    # Frames
    def frame_creation_condition(self, path):
        return True

    # File
    def date_from_path(self):
        date_string = self.path.parts[-1].split('_')[1]
        self._date = datetime.datetime.strptime(date_string, self.date_format).date()
