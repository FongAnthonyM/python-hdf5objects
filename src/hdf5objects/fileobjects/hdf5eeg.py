"""hdf5eeg.py
A HDF5 file which contains data for EEG data.
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
import pathlib
import datetime
from typing import Any

# Third-Party Packages #
from classversioning import VersionType, TriNumberVersion, Version
from dspobjects.time import Timestamp, nanostamp
import h5py

# Local Packages #
from ..hdf5bases import HDF5Map, HDF5Dataset
from ..dataset import BaseTimeSeriesMap
from .basehdf5 import BaseHDF5Map, BaseHDF5


# Definitions #
# Classes #
class HDF5EEGMap(BaseHDF5Map):
    """A map for HDF5EEG files."""

    default_attribute_names = {
        "file_type": "FileType",
        "file_version": "FileVersion",
        "subject_id": "subject_id",
        "age": "age",
        "sex": "sex",
        "species": "species",
        "start": "start",
        "end": "end",
    }
    default_map_names = {"data": "EEG Array"}
    default_maps = {"data": BaseTimeSeriesMap()}


class HDF5EEG(BaseHDF5):
    """A HDF5 file which contains data for EEG data.

    Class Attributes:
        _registration: Determines if this class will be included in class registry.
        _VERSION_TYPE: The type of versioning to use.
        FILE_TYPE: The file type name of this class.
        VERSION: The version of this class.
        default_map: The HDF5 map of this object.

    Attributes:
        _subject_id: The ID of the EEG subject data.
        _subject_dir: The directory where subjects data are stored.

    Args:
        file: Either the file object or the path to the file.
        s_id: The subject id.
        s_dir: The directory where subjects data are stored.
        start: The start time of the data, if creating.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the open method.
    """

    _registration: bool = False
    _VERSION_TYPE: VersionType = VersionType(name="HDF5EEG", class_=TriNumberVersion)
    VERSION: Version = TriNumberVersion(0, 0, 0)
    FILE_TYPE: str = "EEG"
    default_map: HDF5Map = HDF5EEGMap()

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        file: str | pathlib.Path | h5py.File | None = None,
        s_id: str | None = None,
        start: datetime.datetime | float | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._subject_id: str = ""

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(file=file, s_id=s_id, start=start, **kwargs)

    @property
    def subject_id(self) -> str:
        """The subject ID from the file attributes."""
        return self.attributes["subject_id"]

    @subject_id.setter
    def subject_id(self, value: str) -> None:
        self.attributes.set_attribute("subject_id", value)
        self._subject_id = value

    @property
    def start_datetime(self) -> Timestamp | None:
        """The start datetime of this file."""
        ns = self.start_nanostamp
        return None if ns is None else Timestamp.fromnanostamp(ns)

    @property
    def start_nanostamp(self) -> float | None:
        """The start timestamp of this file."""
        if (ns := self.attributes.get("start", None)) is None:
            if len(self["data"]) > 0:
                return self["data"].components["timeseries"].get_nanostamp(0)
        return ns

    @property
    def start_timestamp(self) -> float | None:
        """The start timestamp of this file."""
        ns = self.start_nanostamp
        return None if ns is None else ns * 10**9

    @property
    def end_datetime(self) -> Timestamp | None:
        """The end datetime of this file."""
        ns = self.attributes.get("end", None)
        return None if ns is None else Timestamp.fromnanostamp(ns)

    @property
    def end_nanostamp(self) -> float | None:
        """The end timestamp of this file."""
        return self.attributes.get("end", None)

    @property
    def end_timestamp(self) -> float | None:
        """The end timestamp of this file."""
        ns = self.attributes.get("end", None)
        return None if ns is None else ns * 10**9

    @property
    def sample_rate(self) -> float | int:
        """The sample rate of the data."""
        return self["data"].components["timeseries"].sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self["data"].components["timeseries"].sample_rate = value

    @property
    def time_axis(self) -> HDF5Dataset:
        """The time axis of the data."""
        return self["data"].axes[0]["time_axis"]

    @property
    def data(self) -> HDF5Dataset:
        return self["data"]

    # Representation
    def __hash__(self) -> int:
        """Overrides hash to make the class hashable.

        Returns:
            The system ID of the class.
        """
        return id(self)

    # Comparison
    def __eq__(self, other: Any) -> bool:
        """The equals operator implementation."""
        if isinstance(other, HDF5EEG):
            return self.start == other.start
        else:
            return self.start == other

    def __ne__(self, other: Any) -> bool:
        """The not equals operator implementation."""
        if isinstance(other, HDF5EEG):
            return self.start != other.start
        else:
            return self.start != other

    def __lt__(self, other: Any) -> bool:
        """The less than operator implementation."""
        if isinstance(other, HDF5EEG):
            return self.start < other.start
        else:
            return self.start < other

    def __gt__(self, other: Any) -> bool:
        """The greater than operator implementation."""
        if isinstance(other, HDF5EEG):
            return self.start > other.start
        else:
            return self.start > other

    def __le__(self, other: Any) -> bool:
        """The less than or equals operator implementation."""
        if isinstance(other, HDF5EEG):
            return self.start <= other.start
        else:
            return self.start <= other

    def __ge__(self, other: Any) -> bool:
        """The greater than or equals operator implementation."""
        if isinstance(other, HDF5EEG):
            return self.start >= other.start
        else:
            return self.start >= other

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        file: str | pathlib.Path | h5py.File | None = None,
        s_id: str | None = None,
        start: datetime.datetime | float | None = None,
        **kwargs: Any,
    ) -> "HDF5EEG":
        """Constructs this object.

        Args:
            file: Either the file object or the path to the file.
            s_id: The subject id.
            s_dir: The directory where subjects data are stored.
            start: The start time of the data, if creating.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if s_id is not None:
            self._subject_id = s_id

        super().construct(file=file, **kwargs)

        return self

    def construct_file_attributes(
        self,
        start: datetime.datetime | float | None = None,
        map_: HDF5Map = None,
        load: bool = False,
        require: bool = False,
    ) -> None:
        """Creates the attributes for this group.

        Args:
            start: The start time of the data, if creating.
            map_: The map to use to create the attributes.
            load: Determines if this object will load the attribute values from the file on construction.
            require: Determines if this object will create and fill the attributes in the file on construction.
        """
        super().construct_file_attributes(map_=map_, load=load, require=require)
        self.attributes["subject_id"] = self._subject_id
        if start is not None:
            self.attributes["start"] = nanostamp(start)

    def construct_dataset(self, load: bool = False, require: bool = False, **kwargs: Any) -> None:
        """Constructs the main EEG dataset.

        Args:
            load: Determines if this object will load the dataset.
            require: Determines if this object will create and fill the dataset.
            **kwargs: The keyword arguments for creating the dataset.
        """
        self._group.get_member(name="data", load=load, require=require, **kwargs)

    def require_dataset(self, load: bool = False, require: bool = True, **kwargs: Any) -> Any:
        """Requires the main EEG dataset.

        Args:
            load: Determines if this object will load the dataset.
            require: Determines if this object will create and fill the dataset.
            **kwargs: The keyword arguments for creating the dataset.
        """
        return self._group.construct_member(name="data", load=load, require=require, **kwargs)

    # File
    def create_file(
        self,
        name: str | pathlib.Path = None,
        s_id: str | None = None,
        s_dir: pathlib.Path | str | None = None,
        start: datetime.datetime | float | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates a file, can supply a file name or one can be generated.

        Args:
            name: The file name for this file.
            s_id: The subject id.
            s_dir: The directory where subjects data are stored.
            start: The start time of the data, if creating.
            **kwargs: The keyword arguments for creating the file.
        """
        if s_id is not None:
            self._subject_id = s_id

        super().create_file(name=name, **kwargs)

    def close(self) -> bool:
        """Closes the HDF5 file.

        Returns:
            If the file was successfully closed.
        """
        try:
            self.standardize_attributes()
        finally:
            return super().close()

    # Attributes Modification
    def validate_attributes(self) -> bool:
        """Checks if the attributes that correspond to data match what is in the data.

        Returns:
            If the attributes are valid.
        """
        return self.start == self.time_axis.start and self.end == self.data._time_axis.end

    def standardize_attributes(self) -> None:
        """Sets attributes that correspond to values somewhere else to their current values."""
        if self.mode in {"w", "a"} and not self.swmr_mode:
            if self.data.exists:
                self.data.standardize_attributes()

            if self.time_axis.exists:
                self.attributes["start"] = self.time_axis.components["axis"].start_nanostamp
                self.attributes["end"] = self.time_axis.components["axis"].end_nanostamp
