#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" directorytimeframe.py
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

# Local Libraries #
from ..timeseriesframe import TimeSeriesFrame


# Definitions #
# Classes #
class DirectoryTimeFrame(TimeSeriesFrame):
    default_return_frame_type = TimeSeriesFrame
    default_frame_type = None

    # Class Methods #
    @classmethod
    def validate_path(cls, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        return path.is_dir()

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, path=None, frames=None, update=True, init=True, **kwargs):
        super().__init__(init=False)
        self._path = None

        self.glob_condition = "*"

        self.is_updating_all = False
        self.is_updating_last = True

        self.frame_type = self.default_frame_type
        self.frame_names = set()

        if init:
            self.construct(path=path, frames=frames, update=update, **kwargs)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    # Context Managers
    def __enter__(self):
        """The context enter which opens the HDF5 file.

        Returns:
            This object.
        """
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """The context exit which closes the file."""
        self.close()

    # Instance Methods
    # Constructors/Destructors
    def construct(self, path=None, frames=None, update=True, **kwargs):
        super().construct(frames=frames, update=update)

        if path is not None:
            self.path = path

        if path is not None:
            self.construct_frames(**kwargs)

    def construct_frames(self, **kwargs):
        for path in self.path.glob(self.glob_condition):
            if path not in self.frame_names:
                if self.frame_creation_condition(path):
                    self.frames.append(self.frame_type(path, **kwargs))
                    self.frame_names.add(path)
        self.frames.sort(key=lambda frame: frame.start)

    # Frames
    def frame_creation_condition(self, path):
        return self.frame_type.validate_path(path)

    # Get a Range of Frames
    def get_range(self, start=None, stop=None, step=None, frame=None):
        with self:
            return super().get_range(start=start, stop=stop, step=step, frame=frame)

    # Path and File System
    def open(self, mode='a', **kwargs):
        for frame in self.frames:
            frame.open(mode, **kwargs)
        return self

    def close(self):
        for frame in self.frames:
            frame.close()

    def require_path(self):
        if not self.path.is_dir():
            self.path.mkdir()

    def require_frames(self):
        for frame in self.frames():
            try:
                frame.require()
            except AttributeError:
                continue

    def require(self):
        self.require_path()
        self.require_frames()
