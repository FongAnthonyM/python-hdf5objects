"""hdf5eegframe.py
A frame that interfaces with a HDF5EEG file.
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
from abc import abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
import datetime
import pathlib
from typing import Any, Union

# Third-Party Packages #
from dspobjects.dataclasses import IndexDateTime, FoundTimeRange, FoundData
from framestructure import (
    TimeSeriesFrame,
    TimeSeriesFrameInterface,
    FileTimeContainerInterface,
)
import h5py
import numpy as np

# Local Packages #
from ..hdf5bases import HDF5Map
from ..fileobjects import HDF5EEG


# Definitions #
# Classes #
class HDF5EEGFrame(FileTimeContainerInterface):
    """A frame that interfaces with a HDF5EEG file.

    Class Attributes:
        file_type: The type of file this object will be wrapping.
        default_data_container: The default data container to use when making new data container frames.

    Attributes:
        subject_id: The subject id.

    Args:
        file: Either the file object or the path to the file.
        s_id: The subject id.
        s_dir: The directory where subjects data are stored.
        start: The start time of the data, if creating.
        mode: The mode this frame and file will be in.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the open method.
    """

    file_type: type = HDF5EEG
    default_data_container: type | None = None

    # Class Methods #
    @classmethod
    def validate_path(cls, path: pathlib.Path | str) -> bool:
        """Validates if path to the file exists and is usable.

        Args:
            path: The path to the file to validate.

        Returns:
            Whether this path is valid or not.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if path.is_file():
            return cls.file_type.validate_file_type(path)
        else:
            return False

    @classmethod
    def new_validated(
        cls, path: pathlib.Path | str, mode: str = "r+", **kwargs: Any
    ) -> Union["HDF5EEGFrame", None]:
        """Checks if the given path is a valid file and returns an instance of this object if valid.

        Args:
            path: The path to the file.
            mode: The mode to put the file in.
            **kwargs: The keyword arguments for constructing the file.

        Returns:
            An instance of this object using the path.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if path.is_file():
            file = cls.file_type.new_validated(path, mode=mode, open_=True)
            if file:
                return cls(file=file, **kwargs)

        return None

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        file: str | pathlib.Path | h5py.File | HDF5EEG | None = None,
        s_id: str | None = None,
        start: datetime.datetime | float | None = None,
        mode: str = "r",
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.manual_close: bool = True

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(file=file, s_id=s_id, start=start, mode=mode, **kwargs)

    @property
    def subject_id(self) -> str:
        """The subject id of the file."""
        return self.file.subject_id

    @subject_id.setter
    def subject_id(self, value: str) -> None:
        self.file.subject_id = value

    # Context Managers
    def __enter__(self) -> "HDF5EEGFrame":
        """The context enter which opens the HDF5 file.

        Returns:
            This object.
        """
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """The context exit which closes the file."""
        return self.close()

    # Instance Methods #
    # File
    @contextmanager
    def temp_open(self, **kwargs: Any) -> "HDF5EEGFrame":
        """Temporarily opens the file if it is not already open.

        Args:
            **kwargs: The keyword arguments for opening the HDF5 file.

        Returns:
            This object.
        """
        if not self.file.is_open:
            self.file.open(**kwargs)
        try:
            yield self
        finally:
            if not self.manual_close:
                self.file.close()

    def require(
        self,
        name: str | pathlib.Path = None,
        open_: bool = True,
        map_: HDF5Map = None,
        load: bool = False,
        build: bool = False,
        **kwargs: Any,
    ) -> None:
        """Creates the HDF5 file or loads it if it exists.

        Args:
            name: The file name as path.
            open_: Determines if this object will remain open after creation.
            map_: The map for this HDF5 object.
            load: Determines if the values of this file will be loaded.
            build: Determines if the values of this file will be filled.
            **kwargs: The keyword arguments for the open method.
        """
        self.file.require_file(
            name=name, open_=open_, map_=map_, load=load, build=build, **kwargs
        )

    def load(self) -> None:
        """Loads all data from the file into the file object."""
        self.file.data.get_all_data()

    # Setters/Getters
    def get_data(self) -> Any:
        """Gets the data from the file.

        Returns:
            The data.
        """
        with self.temp_open():
            return self.file.data

    def set_data(self, value: np.ndarray) -> None:
        """Sets the data in the file.

        Args:
            value: The data that will replace what is in the file.
        """
        if self.mode == "r":
            raise IOError("not writable")
        self.file.data.set_data(value)

    def get_time_axis(self) -> Any:
        """Gets the time axis as numpy array of timestamps.

        Returns:
            The timestamps of the data.
        """
        with self.temp_open():
            return self.data.components["timeseries"].time_axis

    def set_time_axis(self, value: Any) -> None:
        """Sets the time axis

        Args:
            value: A time axis object.
        """
        if self.mode == "r":
            raise IOError("not writable")
        self.data.time_axis.set_dataset(value)

    def get_shape(self) -> tuple[int]:
        """Gets the shape of the data.

        Returns:
            The shape of the data.
        """
        with self.temp_open():
            return self.data.shape

    def get_sample_rate(self) -> int | float:
        """Gets the sampling rate of the data.

        Returns:
            The sampling rate.
        """
        return self.data.sample_rate

    def set_sample_rate(self, value: float) -> None:
        """Sets the sample rate.

        Args:
            value: The sample rate to set the file to.
        """
        if self.mode == "r":
            raise IOError("not writable")
        self.data.sample_rate = value
