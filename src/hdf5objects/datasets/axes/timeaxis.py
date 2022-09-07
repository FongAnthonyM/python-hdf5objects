""" timeaxis.py
An Axis that represents the time at each sample of a signal.
"""
# Package Header #
from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from collections.abc import Iterable, Mapping
import datetime
from typing import Any
import zoneinfo

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod
from baseobjects.cachingtools import timed_keyless_cache
from dspobjects.dataclasses import IndexDateTime, FoundTimeRange
import h5py
import numpy as np
import tzlocal

# Local Packages #
from .axis import AxisMap, Axis


# Definitions #
# Classes #
class TimeAxisMap(AxisMap):
    """A map for the TimeAxis object."""
    default_attribute_names: Mapping[str, str] = {"sample_rate": "sample_rate", "time_zone": "time_zone"}
    default_attributes: Mapping[str, Any] = {"sample_rate": h5py.Empty('f8'), "time_zone": ""}
    default_kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,), "dtype": "f8"}


class TimeAxis(Axis):
    """An Axis that represents the time at each sample of a signal.

    Class Attributes:
        local_timezone: The name of the timezone this program is running in.

    Attributes:
        default_kwargs: The default keyword arguments to use when creating the dataset.
        _scale_name: The scale name of this axis.

    Args:
        start: The start of the axis.
        stop: The end of the axis.
        step: The interval between each datum of the axis.
        rate: The frequency of the data of the axis.
        size: The number of datum in the axis.
        datetimes: The datetimes to populate this axis.
        s_name: The name of the axis (scale).
        build: Determines if the axis should be created and filled.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the HDF5Dataset.
    """
    local_timezone: str = tzlocal.get_localzone_name()

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        rate: int | float | None = None,
        size: int | None = None,
        datetimes: Iterable[datetime.datetime | float] | np.ndarray | None = None,
        s_name: str | None = None,
        build: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # Overriden Attributes #
        self._scale_name = "time axis"

        # Object Construction #
        if init:
            self.construct(
                start=start,
                stop=stop,
                step=step,
                rate=rate,
                size=size,
                datetimes=datetimes,
                s_name=s_name,
                build=build,
                **kwargs,
            )

    @property
    def start_timestamp(self) -> float:
        """Get the first element of this axis."""
        try:
            return self.get_start.caching_call()
        except AttributeError:
            return self.get_start()

    @property
    def end_timestamp(self) -> float:
        """Get the last element of this axis."""
        try:
            return self.get_end.caching_call()
        except AttributeError:
            return self.get_end()
    
    @property
    def start_datetime(self) -> datetime.datetime:
        """Returns the start as a datetime."""
        return datetime.datetime.fromtimestamp(self.start_timestamp, self.time_zone)

    @property
    def end_datetime(self) -> datetime.datetime:
        """Returns the end as a datetime."""
        return datetime.datetime.fromtimestamp(self.end_timestamp, self.time_zone)

    @property
    def sample_rate(self) -> float | h5py.Empty:
        """The sample rate of this timeseries."""
        return self.attributes["sample_rate"]

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self.attributes.set_attribute("sample_rate", value)

    @property
    def time_zone(self) -> zoneinfo.ZoneInfo | None:
        """The timezone of the timestamps for this axis. Setter validates before assigning."""
        return self.get_time_zone(refresh=False)

    @time_zone.setter
    def time_zone(self, value: str | zoneinfo.ZoneInfo) -> None:
        self.set_time_zone(value)

    @property
    def timestamps(self) -> np.ndarray:
        """Returns all the data for this object as unix timestamps."""
        try:
            return self.get_all_data.caching_call()
        except AttributeError:
            return self.get_all_data()

    @property
    def datetimes(self) -> tuple[datetime.datetime]:
        """Returns all the data for this object as datetime objects."""
        try:
            return self.get_datetimes.caching_call()
        except AttributeError:
            return self.get_datetimes()

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        rate: float | None = None,
        size: int | None = None,
        datetimes: Iterable[datetime.datetime | float] | np.ndarray | None = None,
        s_name: str | None = None,
        build: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            start: The start of the axis.
            stop: The end of the axis.
            step: The interval between each datum of the axis.
            rate: The frequency of the data of the axis.
            size: The number of datum in the axis.
            datetimes: The datetimes to populate this axis.
            s_name: The name of the axis (scale).
            build: Determines if the axis should be created and filled.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        if "data" in kwargs:
            kwargs["build"] = build
            build = False

        super().construct(s_name=s_name, **kwargs)

        if build:
            if datetimes is not None:
                self.from_datetimes(datetimes=datetimes)
            elif start is not None:
                self.from_range(start=start, stop=stop, step=step, rate=rate, size=size)
            else:
                self.require(shape=(0,), maxshape=(None,), dtype="f8")

    def from_range(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        rate: float | None = None,
        size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates the axis from a range style of input.

        Args:
            start: The start of the axis.
            stop: The end of the axis.
            step: The interval between each datum of the axis.
            rate: The frequency of the data of the axis.
            size: The number of datum in the axis.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)

        if step is None and rate is not None:
            step = 1 / rate
        elif isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if start is None:
            start = stop - step * size
        elif isinstance(start, datetime.datetime):
            start = start.timestamp()

        if stop is None:
            stop = start + step * size
        elif isinstance(stop, datetime.datetime):
            stop = stop.timestamp()

        if size is not None:
            self.set_data(data=np.linspace(start, stop, size), **d_kwargs)
        else:
            self.set_data(data=np.arange(start, stop, step), **d_kwargs)

    @singlekwargdispatchmethod("datetimes")
    def from_datetimes(self, datetimes: Iterable[datetime.datetime | float] | np.ndarray, **kwargs: Any) -> None:
        """Sets the axis values to a series of datetimes.

        Args:
            datetimes: The datetimes of the axis.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        raise TypeError(f"A {type(datetimes)} cannot be used to construct the time axis.")

    @from_datetimes.register(Iterable)
    def _(self, datetimes: Iterable[datetime.datetime | float], **kwargs: Any) -> None:
        """Sets the axis values to a series of datetimes.

        Args:
            datetimes: The datetimes of the axis.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)
        datetimes = list(datetimes)

        stamps = np.zeros(shape=(len(datetimes),))
        for index, dt in enumerate(datetimes):
            if isinstance(dt, datetime.datetime):
                stamps[index] = dt.timestamp()
            else:
                stamps[index] = dt
        self.set_data(data=stamps, **d_kwargs)

    @from_datetimes.register
    def _(self, datetimes: np.ndarray, **kwargs: Any) -> None:
        """Sets the axis values to a series of timestamps.

        Args:
            datetimes: The timestamps of the axis.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        d_kwargs = self.default_kwargs.copy()
        d_kwargs.update(kwargs)
        self.set_data(data=datetimes, **d_kwargs)

    # File
    def create(self, name: str = None, **kwargs: Any) -> "TimeAxis":
        """Creates this TimeAxis in the HDF5File.

        Args:
            name: The name of this axis.
            **kwargs: The keyword arguments for creating this HDF5
        """
        super().create(name=name, **kwargs)
        if "time_zone" not in self.attributes:
            tz = self.map.attributes.get("time_zone", None)
            self.set_time_zone(tz)
        return self

    def require(self, name: str = None, **kwargs: Any) -> "TimeAxis":
        """Creates this TimeAxis in the HDF5File if it does not exist.

        Args:
            name: The name of this axis.
            **kwargs: The keyword arguments for creating this HDF5
        """
        super().require(name=name, **kwargs)
        if "time_zone" not in self.attributes:
            tz = self.map.attributes.get("time_zone", None)
            self.set_time_zone(tz)
        return self

    def refresh(self) -> None:
        """Reloads the time axis and attributes."""
        super().refresh()
        self.get_datetimes.clear_cache()

    # Getters/Setter
    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_all_data(self) -> np.ndarray:
        """Gets all the data in the dataset.

        Returns:
            All the data in the dataset.
        """
        self.get_datetimes.clear_cache()
        with self:
            return self._dataset[...]

    def get_time_zone(self, refresh: bool = True) -> zoneinfo.ZoneInfo | None:
        """Get the timezone of this axis.

        Args:
            refresh: Determines if the attributes will refresh before checking the timezone.
        """
        if refresh:
            self.attributes.refresh()
        tz_str = self.attributes.get("time_zone", self.sentinel)
        if tz_str is self.sentinel or isinstance(tz_str, h5py.Empty) or tz_str == "":
            return None
        else:
            return zoneinfo.ZoneInfo(tz_str)

    def set_time_zone(self, value: str | zoneinfo.ZoneInfo | None = None) -> None:
        """Sets the timezone of this axis.

        Args:
            value: The timezone to set this axis to.
        """
        if value is None:
            value = ""
        elif isinstance(value, zoneinfo.ZoneInfo):
            value = str(value)
        elif value.lower() == "local" or value.lower() == "localtime":
            value = self.local_timezone
        else:
             zoneinfo.ZoneInfo(value)  # Raises an error if the given string is not a time zone.
        self.attributes["time_zone"] = value

    # Get Data
    def get_timestamps(self) -> np.ndarray:
        """Returns all the data for this object as unix timestamps.

        Returns:
            All the unix timestamps in this axis.
        """
        return self.get_all_data()

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_datetimes(self, tz=None) -> tuple[datetime.datetime]:
        """Returns all the data for this object as datetimes.

        Args:
            tz: The timezone information that can be assigned to the datetimes.

        Returns:
            All the unix datetimes in this axis.
        """
        origin_tz = self.time_zone
        timestamps = self.get_all_data()
        if tz is not None:
            return tuple(datetime.datetime.fromtimestamp(t, origin_tz).astimezone(tz) for t in timestamps)
        else:
            return tuple(datetime.datetime.fromtimestamp(t, origin_tz) for t in timestamps)

    def get_datetime(self, index: int) -> datetime.datetime:
        """Get a datatime from this axis with an index.

        Args:
            index: The index to get the datetime from.

        Returns:
            The requested datetime.
        """
        return self.datetimes[index]

    def get_timestamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Get a range of timestamps with indices.

        Args:
            start: The start index.
            stop: The stop index.
            step: The interval between indices to get timestamps.

        Returns:
            The requested range of timestamps.
        """
        return self.timestamps[slice(start, stop, step)]

    def get_datetime_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> tuple[datetime.datetime]:
        """Get a range of datetimes with indices.

        Args:
            start: The start index.
            stop: The stop index.
            step: The interval between indices to get datetimes.

        Returns:
            The requested range of datetimes.
        """
        return self.datetimes[slice(start, stop, step)]

    # Find
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
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()

        samples = self.timestamps.shape[0]
        if timestamp < self.timestamps[0]:
            if tails:
                return IndexDateTime(0, self.start_datetime, self.start)
        elif timestamp > self.timestamps[-1]:
            if tails:
                return IndexDateTime(samples, self.end_datetime, self.end)
        else:
            index = int(np.searchsorted(self.timestamps, timestamp, side="right") - 1)
            true_timestamp = self.timestamps[index]
            if approx or timestamp == true_timestamp:
                return IndexDateTime(index, datetime.datetime.fromtimestamp(true_timestamp), true_timestamp)
        
        raise IndexError("Timestamp out of range.")

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
        if isinstance(step, datetime.timedelta):
            step = step.total_seconds()
        
        if start is None:
            start_index = 0
        else:
            start_index, _ = self.find_time_index(timestamp=start, approx=approx, tails=tails)

        if stop is None:
            stop_index = self.shape[0] - 1
        else:
            stop_index, _ = self.find_time_index(timestamp=stop, approx=approx, tails=tails)

        if start_index is None and stop_index is None:
            return FoundTimeRange(None, None, None)
        else:
            data = self.timestamps[start_index:stop_index:step]

            return FoundTimeRange(data, start_index, stop_index)

    def find_datetime_range(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = False,
        tails: bool = False,
    ) -> FoundTimeRange:
        """Finds the datetime range on the axis inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The datetime range on the axis and the start and stop indices.
        """
        if isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if start is None:
            start_index = 0
        else:
            start_index, _ = self.find_time_index(timestamp=start, approx=approx, tails=tails)

        if stop is None:
            stop_index = self.shape[0] - 1
        else:
            stop_index, _ = self.find_time_index(timestamp=stop, approx=approx, tails=tails)

        if start_index is None and stop_index is None:
            return FoundTimeRange(None, None, None)
        else:
            data = self.datetimes[start_index:stop_index:step]

            return FoundTimeRange(data, start_index, stop_index)

    # Todo: Add Fill Methods


# Assign Cyclic Definitions
TimeAxisMap.default_type = TimeAxis
TimeAxis.default_map = TimeAxisMap()
