""" hdf5eegframe.py
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
import datetime
import pathlib
from typing import Any, Union

# Third-Party Packages #
from dspobjects.dataclasses import IndexDateTime, FoundTimeRange, FoundData
from framestructure import TimeSeriesFrame, TimeSeriesFrameInterface, FileTimeFrame
import h5py
import numpy as np

# Local Packages #
from ..hdf5bases import HDF5Map
from ..fileobjects import HDF5EEG


# Definitions #
# Classes #
class HDF5EEGFrame(FileTimeFrame):
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
    def new_validated(cls, path: pathlib.Path | str, mode: str = "r+", **kwargs: Any) -> Union["HDF5EEGFrame", None]:
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
        s_dir: str | pathlib.Path | None = None,
        start: datetime.datetime | float | None = None,
        mode: str = 'r',
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(file=file, s_id=s_id, s_dir=s_dir, start=start, mode=mode, **kwargs)

    @property
    def subject_id(self) -> str:
        """The subject id of the file."""
        return self.file.subject_id

    @subject_id.setter
    def subject_id(self, value: str) -> None:
        self.file.subject_id = value

    # Instance Methods #
    # File
    def require(
        self,
        name: str | pathlib.Path = None,
        open_: bool = True,
        map_: HDF5Map = None,
        load: bool = False,
        build: bool = False,
        **kwargs: Any
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
        self.file.require_file(name=name, open_=open_, map_=map_, load=load, build=build, **kwargs)

    def load(self) -> None:
        """Loads all data from the file into the file object."""
        self.file.data.get_all_data()

    # Setters/Getters
    def get_data(self) -> Any:
        """Gets the data from the file.

        Returns:
            The data.
        """
        return self.file.data

    def set_data(self, value: np.ndarray) -> None:
        """Sets the data in the file.

        Args:
            value: The data that will replace what is in the file.
        """
        if self.mode == 'r':
            raise IOError("not writable")
        self.file.data.set_data(value)

    def get_time_axis(self) -> Any:
        """Gets the time axis as numpy array of timestamps.

        Returns:
            The timestamps of the data.
        """
        return self.data.time_axis

    def set_time_axis(self, value: Any) -> None:
        """Sets the time axis

        Args:
            value: A time axis object.
        """
        if self.mode == 'r':
            raise IOError("not writable")
        self.data.time_axis.set_dataset(value)

    def get_shape(self) -> tuple[int]:
        """Gets the shape of the data.

        Returns:
            The shape of the data.
        """
        return self.data.shape

    def get_start_timestamp(self) -> float | None:
        """Gets the start_timestamp timestamp of this frame.

        Returns:
            The start_timestamp timestamp of this frame.
        """
        return self.file.start_timestamp

    def get_end_timestamp(self) -> float | None:
        """Gets the end_timestamp timestamp of this frame.

        Returns:
            The end_timestamp timestamp of this frame.
        """
        return self.file.end_timestamp

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
        if self.mode == 'r':
            raise IOError("not writable")
        self.data.sample_rate = value

    # Shape
    def resize(self, shape: tuple[int] | None = None, dtype: np.dtype | None = None, **kwargs: Any) -> None:
        """Changes the size of the data to a new size filling empty areas with NaN.

        Args:
            shape: The new shape to change the data to.
            dtype: The type the data will be.
            **kwargs: The keyword arguments for generating new blank data.
        """
        if self.mode == 'r':
            raise IOError("not writable")

        if shape is None:
            shape = self.target_shape

        if dtype is None:
            dtype = self.data.dtype

        new_slices = [0] * len(shape)
        old_slices = [0] * len(self.shape)
        for index, (n, o) in enumerate(zip(shape, self.shape)):
            slice_ = slice(None, n if n > o else o)
            new_slices[index] = slice_
            old_slices[index] = slice_

        new_ndarray = self.blank_generator(shape, dtype, **kwargs)
        new_ndarray[tuple(new_slices)] = self.data[tuple(old_slices)]

        self.data.set_data(new_ndarray)

    # Sample Rate
    def resample(self, sample_rate: int | float, **kwargs):
        """Resamples the data.

        Args:
            sample_rate: The new sample rate to change the data to.
            **kwargs: The keyword arguments for the resampler.
        """
        if self.mode == 'r':
            raise IOError("not writable")

        if not self.validate_sample_rate():
            raise ValueError("the data needs to have a uniform sample rate before resampling")

        if sample_rate is not None:
            self.sample_rate = sample_rate

        self.data.replace_data(self.resampler(new_fs=self.sample_rate, **kwargs))
        self.time_axis.replace_data(np.arange(self.time_axis[0], self.time_axis[-1], self.sample_period, dtype="f8"))

        # Time Correction

    def fill_time_correction(self, axis: int | None = None, tolerance: float | None = None, **kwargs: Any) -> None:
        """Corrects the data's discontinuities in time.

        Args:
            axis: The axis to do the correction.
            tolerance: The tolerance for samples to be misaligned.
            **kwargs: The keyword arguments
        """
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        discontinuities = self.where_discontinuous(tolerance=tolerance)

        if discontinuities:
            offsets = np.empty((0, 2), dtype="i")
            gap_discontinuities = []
            previous_discontinuity = 0
            for discontinuity in discontinuities:
                timestamp = self.time_axis[discontinuity]
                previous = discontinuity - 1
                previous_timestamp = self.time_axis[previous]
                if (timestamp - previous_timestamp) >= (2 * self.sample_period):
                    real = discontinuity - previous_discontinuity
                    blank = round((timestamp - previous_timestamp) * self.sample_rate) - 1
                    offsets = np.append(offsets, [[real, blank]], axis=0)
                    gap_discontinuities.append(discontinuities)
                    previous_discontinuity = discontinuity
            offsets = np.append(offsets, [[self.time_axis - discontinuities[-1], 0]], axis=0)

            new_size = np.sum(offsets)
            new_shape = list(self.data.shape)
            new_shape[axis] = new_size
            old_data = self.data
            old_times = self.time_axis
            self.data.set_data(self.blank_generator(shape=new_shape, **kwargs))
            self.time_axis.replace_data(np.empty((new_size,), dtype="f8"))
            old_start = 0
            new_start = 0
            for discontinuity, offset in zip(gap_discontinuities, offsets):
                previous = discontinuity - 1
                new_mid = new_start + offset[0]
                new_end = new_mid + offset[1]
                mid_timestamp = old_times[previous] + self.sample_period
                end_timestamp = offset[1] * self.sample_period

                slice_ = slice(start=old_start, stop=old_start + offset[0])
                slices = [slice(None, None)] * len(old_data.shape)
                slices[axis] = slice_

                self.set_range(old_data[tuple(slices)], start=new_start)

                self.time_axis[new_start:new_mid] = old_times[slice_]
                self.time_axis[new_mid:new_end] = np.arange(mid_timestamp, end_timestamp, self.sample_period)

                old_start = discontinuity
                new_start += sum(offset)

    # Get Timestamps
    def get_timestamps(self) -> np.ndarray:
        """Gets all the timestamps of this frame.

        Returns:
            A numpy array of the timestamps of this frame.
        """
        return self.time_axis.get_timestamps()

    def get_datetimes(self) -> tuple[datetime.datetime]:
        """Gets all the datetimes of this frame.

        Returns:
            All the times as a tuple of datetimes.
        """
        return self.time_axis.get_datetimes()

    # Data
    def append(
        self,
        data: np.ndarray,
        time_axis: np.ndarray | None = None,
        axis: int | None = None,
        tolerance: float | None = None,
        correction: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Appends data and timestamps onto the contained data and timestamps

        Args:
            data: The data to append.
            time_axis: The timestamps of the data.
            axis: The axis to append the data to.
            tolerance: The allowed deviation a sample can be away from the sample period.
            correction: Determines if time correction will be run on the data and the type if a str.
            **kwargs: The keyword arguments for the time correction.
        """
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        if correction is None or (isinstance(correction, bool) and correction):
            correction = self.tail_correction
        elif isinstance(correction, str):
            correction = self.get_correction(correction)

        if correction:
            data, time_axis = correction(data, time_axis, axis=axis, tolerance=tolerance, **kwargs)

        self.data.append(data, axis)
        self.time_axis.append(time_axis)

    def add_frames(
        self,
        frames: Iterable[TimeSeriesFrameInterface],
        axis: int | None = None,
        truncate: bool | None = None,
    ) -> None:
        """Appends data and timestamps from other frames to this frame.

        Args:
            frames: The frames to append data from.
            axis: The axis to append the data along.
            truncate: Determines if the other frames' data will be truncated to fit this frame's shape.
        """
        if self.mode == 'r':
            raise IOError("not writable")

        frames = list(frames)

        if self.data is None:
            frame = frames.pop(0)
            if not frame.validate_sample_rate():
                raise ValueError("the frame's sample rate must be valid")
            self.data.set_data(frame[...])
            self.time_axis.replace_data(frame.get_time_axis())

        for frame in frames:
            self.append_frame(frame, axis=axis, truncate=truncate)

    def get_intervals(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> np.ndarray:
        """Get the intervals between each time in the axis.

        Args:
            start: The start index to get the intervals.
            stop: The last index to get the intervals.
            step: The step of the indices to the intervals.

        Returns:
            The intervals between each time in the axis.
        """
        return self.time_axis.get_intervals(start=start, stop=stop, step=step)

    def fill_timestamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with timestamps from the contained frames/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        data_array[array_slice] = self.time_axis.timestamps[slice_]
        return data_array

    # Find Index
    def find_time_index(
        self,
        timestamp: datetime.datetime | float,
        approx: bool = False,
        tails: bool = False,
    ) -> IndexDateTime:
        """Finds the index with given time, can give approximate values.

        Args:
            timestamp:
            approx: Determines if an approximate index will be given if the time is not present.
            tails: Determines if the first or last index will be give the requested time is outside the axis.

        Returns:
            The requested closest index and the value at that index.
        """
        return self.time_axis.find_time_index(timestamp=timestamp, approx=approx, tails=tails)

    # Find Data
    def find_timestamp_range(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = False,
        tails: bool = False,
    ) -> FoundTimeRange:
        """Finds the timestamp range on the axis inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The timestamp range on the axis and the start and stop indices.
        """
        return self.time_axis.find_timestamp_range(start=start, stop=stop, step=step, approx=approx, tails=tails)

    def find_data_range(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = False,
        tails: bool = False,
    ) -> FoundData:
        """Finds the data range on the axis inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The data range on the axis and the time axis as timestamps.
        """
        return self.data.find_data_range_timestamp(start=start, stop=stop, step=step, approx=approx, tails=tails)


