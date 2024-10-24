"""timeaxiscomponent.py
A component and map for a HDF5Dataset which defines it as an axis that represents time.
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
import time
from typing import Any
import zoneinfo

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch
from baseobjects.cachingtools import timed_keyless_cache
from baseobjects.operations import timezone_offset
from proxyarrays import ContainerTimeAxis
import h5py
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Dataset
from .axiscomponent import AxisMap, AxisComponent


# Definitions #
# Classes #
class TimeAxisComponent(AxisComponent, ContainerTimeAxis):
    """A component for a HDF5Dataset which defines it as an axis that represents time.

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
        require: Determines if the axis should be created and filled.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the HDF5Dataset.
    """

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        composite: Any = None,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        rate: int | float | None = None,
        size: int | None = None,
        datetimes: Iterable[datetime.datetime | float] | np.ndarray | None = None,
        require: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._time_zone_mask: datetime.tzinfo | None = None

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                start=start,
                stop=stop,
                step=step,
                rate=rate,
                size=size,
                datetimes=datetimes,
                require=require,
                **kwargs,
            )

    @property
    def precise(self) -> bool:
        """Determines if this proxy returns nanostamps (True) or timestamps (False)."""
        if self._precise is None:
            return self.get_original_precision()
        else:
            return self._precise

    @precise.setter
    def precise(self, value: bool | None) -> None:
        self.set_precision(nano=value)

    @property
    def _nanostamps(self) -> np.ndarray | None:
        """The nanosecond timestamps of this proxy."""
        if self.get_original_precision():
            return self.composite[...]
        else:
            return None

    @_nanostamps.setter
    def _nanostamps(self, value: np.ndarray | None) -> None:
        pass

    @property
    def _timestamps(self) -> np.ndarray | None:
        """The timestamps of this proxy."""
        if not self.get_original_precision():
            return self.get_all_data()
        else:
            return None

    @_timestamps.setter
    def _timestamps(self, value: np.ndarray | None) -> None:
        pass

    @property
    def nanostamps(self) -> np.ndarray | None:
        """The nanosecond timestamps of this proxy."""
        return self.get_nanostamps()

    @property
    def timestamps(self) -> np.ndarray | None:
        """The timestamps of this proxy."""
        return self.get_timestamps()

    @property
    def _sample_rate(self) -> Decimal | None:
        """The sample rate of this timeseries."""
        try:
            return Decimal(self.composite.attributes["sample_rate"])
        except TypeError:
            return None

    @_sample_rate.setter
    def _sample_rate(self, value: Decimal) -> None:
        if self.composite is not None:
            self.composite.attributes.set_attribute("sample_rate", float(value))

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
        if self.composite is not None:
            self.set_time_zone(value)

    @property
    def datetimes(self) -> tuple[datetime.datetime]:
        """Returns all the data for this object as datetime objects."""
        try:
            return self.get_datetimes.caching_call()
        except AttributeError:
            return self.get_datetimes()

    @property
    def _data(self) -> Any:
        """The data of the composite."""
        return self.composite

    @_data.setter
    def _data(self, value: Any) -> None:
        pass

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        rate: float | None = None,
        size: int | None = None,
        datetimes: Iterable[datetime.datetime | float] | np.ndarray | None = None,
        require: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            start: The start of the axis.
            stop: The end of the axis.
            step: The interval between each datum of the axis.
            rate: The frequency of the data of the axis.
            size: The number of datum in the axis.
            datetimes: The datetimes to populate this axis.
            require: Determines if the axis should be created and filled.
            **kwargs: Keyword arguments for inheritance.
        """
        super().construct(composite=composite, sample_rate=rate, **kwargs)

        if require:
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
            **kwargs: Keyword arguments for inheritance.
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

    @singlekwargdispatch("datetimes")
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
    def refresh(self) -> None:
        """Reloads the time axis and attributes."""
        super().refresh()
        self.get_datetimes.clear_cache()

    # Getters/Setter
    def get_all_data(self) -> np.ndarray:
        """Gets all the data in the dataset.

        Returns:
            All the data in the dataset.
        """
        return self.composite[...]

    def get_original_precision(self) -> bool:
        """Gets the presision of the timestamps from the orignial file.

        Args:
            nano: Determines if this proxy returns nanostamps (True) or timestamps (False).
        """
        return self.composite.dtype == np.uint64

    def set_precision(self, nano: bool | None) -> None:
        """Sets if this proxy returns nanostamps (True) or timestamps (False).

        Args:
            nano: Determines if this proxy returns nanostamps (True) or timestamps (False).
        """
        if nano is None:
            pass
        elif nano:
            self._data_method = self._get_nanostamps.__func__
        else:
            self._data_method = self._get_timestamps.__func__
        self._precise = nano

    def get_time_zone(self, refresh: bool = True) -> datetime.tzinfo | None:
        """Get the timezone of this axis.

        Args:
            refresh: Determines if the attributes will refresh before checking the timezone.
        """
        if refresh:
            self.composite.attributes.refresh()

        tz_name = self.composite.attributes.get("time_zone", None)
        tz_offset = self.composite.attributes.get("time_zone_offset", None)
        if tz_name is not None and not isinstance(tz_name, h5py.Empty) and tz_name != "":
            try:
                return zoneinfo.ZoneInfo(tz_name)
            except zoneinfo.ZoneInfoNotFoundError as e:
                if tz_offset is not None and not isinstance(tz_offset, h5py.Empty):
                    return datetime.timezone(datetime.timedelta(seconds=tz_offset))
                else:
                    raise e
        elif tz_offset is not None and not isinstance(tz_offset, h5py.Empty):
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
            offset = h5py.Empty("f8")
            value = ""
        elif isinstance(value, datetime.tzinfo):
            offset = timezone_offset(value).total_seconds()
            value = str(value)
        elif value.lower() == "local" or value.lower() == "localtime":
            local_time = time.localtime()
            offset = local_time.tm_gmtoff
            value = local_time.tm_zone
        else:
            zoneinfo.ZoneInfo(value)  # Raises an error if the given string is not a time zone.

        self.composite.attributes["time_zone"] = value
        self.composite.attributes["time_zone_offset"] = offset

    # Masking
    def mask_time_zone(self, tz: datetime.tzinfo | None) -> None:
        """Masks the time zone of this another timezone.

        Args:
            tz: The time zone to use instead or None to use the original time zone.
        """
        self._time_zone_mask = tz
        self.get_datetimes.cache_clear()

    # Data
    def create_component(self) -> None:
        """Creates all the required parts of the dataset for this component, errors if a part already exists."""
        if "time_zone" not in self.attributes:
            tz = self.map.attributes.get("time_zone", None)
            self.set_time_zone(tz)

    def create(self, **kwargs: Any) -> HDF5Dataset:
        """Creates both the data and this component, errors if a part already exists.

        Args:
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        self.composite.create_data(**kwargs)
        self.create_component()
        return self.composite

    def require_component(self) -> None:
        """Creates all the required parts of the dataset for this component if it does not exists."""
        if "time_zone" not in self.composite.attributes:
            tz = self.map.attributes.get("time_zone", None)
            self.set_time_zone(tz)

    def require(self, **kwargs: Any) -> HDF5Dataset:
        """Creates both the data and this component if it does not exists.

        Args:
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        self.composite.require_data(**kwargs)
        self.require_component()
        return self.composite

    def append(
        self,
        data: np.ndarray,
        axis: int | None = None,
        tolerance: float | None = None,
        correction: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Appends data and timestamps onto the contained data and timestamps

        Args:
            data: The data to append.
            axis: The axis to append the data to.
            tolerance: The allowed deviation a sample can be away from the sample period.
            correction: Determines if time correction will be run on the data and the type if a str.
            **kwargs: The keyword arguments for the time correction.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if not any(data.shape):
            return

        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        if isinstance(correction, bool) and correction:
            correction = self.tail_correction
        elif isinstance(correction, str):
            correction = self.get_correction(correction)

        if correction and self.data.size != 0:
            data = correction(data, tolerance=tolerance)

        self.data.append(data, axis)

    def fill_nanostamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with nanostamps from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        try:
            data_array[array_slice] = self.nanostamps[slice_]
        except ValueError:
            data_array[array_slice] = self.get_all_data()[slice_]
        return data_array


class TimeAxisMap(AxisMap):
    """An outline which defines an HDF5Dataset as an Axis that represents time."""

    default_attribute_names: Mapping[str, str] = {
        "sample_rate": "sample_rate",
        "time_zone": "time_zone",
        "time_zone_offset": "time_zone_offset",
    }
    default_attributes: Mapping[str, Any] = {
        "sample_rate": h5py.Empty("f8"),
        "time_zone": "",
        "time_zone_offset": h5py.Empty("f8"),
    }
    default_kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,), "dtype": "u8"}
    default_component_types = {
        "axis": (TimeAxisComponent, {}),
    }
