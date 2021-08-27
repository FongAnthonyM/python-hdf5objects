#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" xltekstudyframe.py
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
import pathlib

# Downloaded Libraries #
from framestructure import DirectoryTimeFrame

# Local Libraries #
from .xltekdayframe import XLTEKDayFrame


# Definitions #
# Classes #
class XLTEKStudyFrame(DirectoryTimeFrame):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    default_frame_type = XLTEKDayFrame

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, s_id=None, studies_path=None, path=None, frames=None, update=True,
                 open_=True, init=True, **kwargs):
        super().__init__(init=False)

        self._studies_path = None
        self.subject_id = ""

        if init:
            self.construct(s_id=s_id, studies_path=studies_path, path=path, frames=frames, update=update,
                           open_=open_, **kwargs)

    @property
    def studies_path(self):
        return self._studies_path

    @studies_path.setter
    def studies_path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._studies_path = value
        else:
            self._studies_path = pathlib.Path(value)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, s_id=None, studies_path=None, path=None, frames=None, update=True, open_=True, **kwargs):
        if s_id is not None:
            self.subject_id = s_id

        if studies_path is not None:
            self.studies_path = studies_path

        if path is None:
            path = pathlib.Path(self.studies_path, self.subject_id)

        super().construct(path=path, frames=frames, update=update, open_=open_, **kwargs)
