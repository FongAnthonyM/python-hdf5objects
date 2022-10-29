""" hdf5eeg.py
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
import h5py
import numpy as np

# Local Packages #
from ..hdf5bases import HDF5Map
from .basehdf5 import BaseHDF5Map, BaseHDF5
from ..datasets import TimeSeriesDataset, TimeSeriesMap, ChannelAxis, SampleAxis, TimeAxis


# Definitions #
# Classes #
class HDF5EEGMap(BaseHDF5Map):
    """A map for HDF5EEG files."""
    default_attribute_names = {"file_type": "FileType",
                               "file_version": "FileVersion",
                               "subject_id": "subject_id",
                               "start": "start",
                               "end": "end"}
    default_map_names = {"data": "EEG Array"}
    default_maps = {"data": TimeSeriesMap()}


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
        s_dir: str | pathlib.Path | None = None,
        start: datetime.datetime | float | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._subject_id: str = ""
        self._subject_dir: pathlib.Path | None = None

        # Object Construction #
        if init:
            self.construct(file=file, s_id=s_id, s_dir=s_dir, start=start, **kwargs)

    @property
    def subject_id(self) -> str:
        """The subject ID from the file attributes."""
        return self.attributes["subject_id"]

    @subject_id.setter
    def subject_id(self, value: str) -> None:
        self.attributes.set_attribute("subject_id", value)
        self._subject_id = value

    @property
    def start_timestamp(self) -> float:
        """The start time of the data as a unix timestamp from the file attributes."""
        return self.attributes["start"]

    @start_timestamp.setter
    def start_timestamp(self, value: float) -> None:
        self.attributes.set_attribute("start", value)

    @property
    def start_datetime(self) -> datetime.datetime:
        """The start time of the data as a datetime from the file attributes."""
        return datetime.datetime.fromtimestamp(self.start_timestamp)

    @property
    def end_timestamp(self) -> float:
        """The end time of the data as a unix timestamp from the file attributes."""
        return self.attributes["end"]

    @end_timestamp.setter
    def end_timestamp(self, value: float) -> None:
        self.attributes.set_attribute("end", value)

    @property
    def end_datetime(self) -> datetime:
        """The end time of the data as a datetime from the file attributes."""
        return datetime.datetime.fromtimestamp(self.end_timestamp)

    @property
    def subject_dir(self) -> pathlib.Path | None:
        """The directory where subjects data are stored."""
        return self._subject_dir

    @subject_dir.setter
    def subject_dir(self, value: pathlib.Path | str) -> None:
        if isinstance(value, pathlib.Path) or value is None:
            self._subject_dir = value
        else:
            self._subject_dir = pathlib.Path(value)

    @property
    def sample_rate(self) -> float | int:
        """The sample rate of the data."""
        return self["data"].sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self["data"].sample_rate = value

    @property
    def n_samples(self) -> int:
        """The number of samples in the data."""
        return self["data"].n_samples

    @property
    def channel_axis(self) -> ChannelAxis:
        """The channel axis of the data."""
        return self["data"].channel_axis

    @property
    def sample_axis(self) -> SampleAxis:
        """The sample axis of the data."""
        return self["data"].sample_axis

    @property
    def time_axis(self) -> TimeAxis:
        """The time axis of the data."""
        return self["data"].time_axis

    @property
    def data(self) -> TimeSeriesDataset:
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
        s_dir: str | pathlib.Path | None = None,
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
        if s_dir is not None:
            self.subject_dir = s_dir

        if s_id is not None:
            self._subject_id = s_id

        if file is None and self.path is None and start is not None:
            self.path = self.subject_dir.joinpath(self.generate_file_name(s_id=s_id, start=start))

        super().construct(file=file, **kwargs)

        if self.path is not None and self.subject_dir is None:
            self.subject_dir = self.path.parent

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
        if isinstance(start, datetime.datetime):
            self.attributes["start"] = start.timestamp()
        elif isinstance(start, float):
            self.attributes["start"] = start
        self.attributes["subject_id"] = self._subject_id

    def construct_dataset(self, load: bool = False, require: bool = False, **kwargs: Any) -> None:
        """Constructs the main EEG dataset.

        Args:
            load: Determines if this object will load the dataset.
            require: Determines if this object will create and fill the dataset.
            **kwargs: The keyword arguments for creating the dataset.
        """
        self._group.get_member(name="data", load=load, require=require, **kwargs)

    # File
    def generate_file_name(self, s_id: str | None = None, start: datetime.datetime | float | None = None) -> str:
        """Generates a file name based on the subject ID and start time.

        Args:
            s_id: The subject id.
            start: The start time of the data, if creating.

        Returns:
            The file name.
        """
        if s_id is None:
            s_id = self.subject_id

        if start is None:
            start = self.start

        if isinstance(start, float):
            start = datetime.datetime.fromtimestamp(start)

        return s_id + '_' + start.isoformat('_', 'seconds').replace(':', '~') + ".h5"

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
        if s_dir is not None:
            self.subject_dir = s_dir

        if name is None and self.path is None and start is not None:
            self.path = self.subject_dir.joinpath(self.generate_file_name(s_id=s_id, start=start))

        super().create_file(name=name, **kwargs)

    # Attributes Modification
    def validate_attributes(self) -> bool:
        """Checks if the attributes that correspond to data match what is in the data.

        Returns:
            If the attributes are valid.
        """
        return self.start == self.data._time_axis.start and self.end == self.data._time_axis.end

    def standardize_attributes(self) -> None:
        """Sets the attributes that correspond to data the actual data values."""
        if self.data.exists:
            self.data.standardize_attributes()
            self.start = self.data._time_axis.start
            self.end = self.data._time_axis.end
