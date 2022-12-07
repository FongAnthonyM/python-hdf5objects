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
from decimal import Decimal
from typing import Any
import zoneinfo

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod
from baseobjects.cachingtools import timed_keyless_cache
from baseobjects.operations import timezone_offset
from framestructure import TimeAxisContainer
import h5py
import numpy as np
import tzlocal

# Local Packages #
from ...hdf5bases import HDF5Map
from .axis import AxisMap, Axis


# Definitions #
# Classes #
class TimeAxisMap(AxisMap):
    """A map for the TimeAxis object."""
    default_attribute_names: Mapping[str, str] = {
        "sample_rate": "sample_rate",
        "time_zone": "time_zone",
        "time_zone_offset": "time_zone_offset",
    }
    default_attributes: Mapping[str, Any] = {
        "sample_rate": h5py.Empty('f8'),
        "time_zone": "",
        "time_zone_offset": h5py.Empty('f8'),
    }
    default_kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,), "dtype": "f8"}


class TimeAxis(Axis, TimeAxisContainer):
    """An Axis that represents the time at each sample of a signal.

    Class Attributes:
        local_timezone: The name of the timezone this program is running in.
        default_scale_name: The default name of this axis.

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
    default_map: HDF5Map = TimeAxisMap()
    default_scale_name: str | None = "time axis"
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
        require: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Override Attributes #
        self._precise: bool = False

        # Parent Attributes #
        Axis.__init__(self, init=False)
        TimeAxisContainer.__init__(self, init=False)

        # New Attributes #
        self._time_zone_mask: datetime.tzinfo | None = None

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
                require=require,
                **kwargs,
            )

    @property
    def precise(self) -> bool:
        """Determines if this frame returns nanostamps (True) or timestamps (False)."""
        return self._precise

    @precise.setter
    def precise(self, value: bool) -> None:
        self.set_precision(nano=value)

    @property
    def _nanostamps(self) -> np.ndarray | None:
        """The nanosecond timestamps of this frame."""
        if self.get_original_precision():
            return self.get_all_data.caching_method()
        else:
            return None

    @_nanostamps.setter
    def _nanostamps(self, value: np.ndarray | None) -> None:
        pass

    @property
    def _timestamps(self) -> np.ndarray | None:
        """The timestamps of this frame."""
        if not self.get_original_precision():
            return self.get_all_data.caching_method()
        else:
            return None

    @_timestamps.setter
    def _timestamps(self, value: np.ndarray | None) -> None:
        pass

    @property
    def nanostamps(self) -> np.ndarray | None:
        """The nanosecond timestamps of this frame."""
        try:
            return self.get_nanostamps.caching_call()
        except AttributeError:
            return self.get_nanostamps()

    @nanostamps.setter
    def nanostamps(self, value: np.ndarray | None) -> None:
        pass

    @property
    def timestamps(self) -> np.ndarray | None:
        """The timestamps of this frame."""
        try:
            return self.get_timestamps.caching_call()
        except AttributeError:
            return self.get_timestamps()

    @timestamps.setter
    def timestamps(self, value: np.ndarray | None) -> None:
        pass

    @property
    def _sample_rate(self) -> Decimal | None:
        """The sample rate of this timeseries."""
        try:
            return Decimal(self.attributes["sample_rate"])
        except TypeError:
            return None

    @_sample_rate.setter
    def _sample_rate(self, value: Decimal) -> None:
        if self.attributes is not None:
            self.attributes.set_attribute("sample_rate", float(value))

    @property
    def time_zone(self) -> zoneinfo.ZoneInfo | None:
        """The time zone of the timestamps for this axis. Setter validates before assigning."""
        if self._time_zone_mask is None:
            return self.get_time_zone(refresh=False)
        else:
            return self._time_zone_mask

    @time_zone.setter
    def time_zone(self, value: str | zoneinfo.ZoneInfo | None) -> None:
        self.set_time_zone(value)

    @property
    def tzinfo(self) -> datetime.tzinfo:
        """The timezone of the timestamps for this axis. Setter validates before assigning."""
        if self._time_zone_mask is None:
            return self.get_time_zone(refresh=False)
        else:
            return self._time_zone_mask

    @tzinfo.setter
    def tzinfo(self, value: str | datetime.datetime | None) -> None:
        if self.attributes is not None:
            self.set_time_zone(value)

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
        require: bool = False,
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
            require: Determines if the axis should be created and filled.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        # Construct the dataset and handle creation here unless data is present.
        Axis.construct(self, s_name=s_name, require=False, **kwargs)

        if require and "data" not in kwargs:
            if datetimes is not None:
                self.from_datetimes(datetimes=datetimes)
            elif start is not None:
                self.from_range(start=start, stop=stop, step=step, rate=rate, size=size)
            else:
                self.require(shape=(0,), maxshape=(None,), dtype="f8")

        TimeAxisContainer.construct(self, sample_rate=rate)

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

    # Masking
    def mask_time_zone(self, tz: datetime.tzinfo | None) -> None:
        """Masks the time zone of this another timezone.

        Args:
            tz: The time zone to use instead or None to use the original time zone.
        """
        self._time_zone_mask = tz
        self.get_datetimes.cache_clear()

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

    def get_original_precision(self) -> bool:
        """Gets the presision of the timestamps from the orignial file.

        Args:
            nano: Determines if this frame returns nanostamps (True) or timestamps (False).
        """
        with self:
            return self._dataset.dtype == np.uint64

    def get_time_zone(self, refresh: bool = True) -> datetime.tzinfo | None:
        """Get the timezone of this axis.

        Args:
            refresh: Determines if the attributes will refresh before checking the timezone.
        """
        if refresh:
            self.attributes.refresh()

        tz_name = self.attributes.get("time_zone", self.sentinel)
        tz_offset = self.attributes.get("time_zone_offset", self.sentinel)
        if tz_name is not self.sentinel and not isinstance(tz_name, h5py.Empty) and tz_name != "":
            try:
                return zoneinfo.ZoneInfo(tz_name)
            except zoneinfo.ZoneInfoNotFoundError as e:
                if tz_offset is not self.sentinel and not isinstance(tz_offset, h5py.Empty):
                    return datetime.timezone(datetime.timedelta(seconds=tz_offset))
                else:
                    raise e
        elif tz_offset is not self.sentinel and not isinstance(tz_offset, h5py.Empty):
            return datetime.timezone(datetime.timedelta(seconds=tz_offset))
        else:
            return None

    def set_time_zone(self, value: str | datetime.tzinfo | None = None, offset: float | None = None) -> None:
        """Sets the timezone of this axis.

        Args:
            value: The time zone to set this axis to.
            offset: The time zone offset from UTC.
        """
        if value is None:
            value = ""
            offset = h5py.Empty('f8')
        elif isinstance(value, datetime.tzinfo):
            offset = timezone_offset(value).total_seconds()
            value = str(value)
        elif value.lower() == "local" or value.lower() == "localtime":
            offset = timezone_offset(zoneinfo.ZoneInfo(self.local_timezone)).total_seconds()
            value = self.local_timezone
        else:
             zoneinfo.ZoneInfo(value)  # Raises an error if the given string is not a time zone.

        self.attributes["time_zone"] = value
        self.attributes["time_zone_offset"] = offset


# Assign Cyclic Definitions
TimeAxisMap.default_type = TimeAxis
