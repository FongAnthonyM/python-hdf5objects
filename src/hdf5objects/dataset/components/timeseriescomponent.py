"""timeseriescomponent.py
A component for a HDF5Dataset which gives it time series functionality.
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
from collections.abc import Iterable
import datetime
from decimal import Decimal
from typing import Any

# Third-Party Packages #
from framestructure import TimeSeriesContainer
import h5py
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Dataset
from ..basedatasetcomponent import BaseDatasetComponent


# Definitions #
# Classes #
class TimeSeriesComponent(BaseDatasetComponent, TimeSeriesContainer):
    """A component for a HDF5Dataset which gives it time series functionality.

    Attributes:
        _sample_rate_: The temporrary sample rate of this time series.
        _time_axis: The time axis object of this time series.
        _t_axis: The dim number of the time axis.
        scale_name: The scale name of the time axis.

    Args:
        composite: The object which this object is a component of.
        t_axis: The dim number of the time axis.
        scale_name: The scale name of the time axis.
        sample_rate: The sample rate of the data in Hz.
        precise: Determines if this frame returns nanostamps (True) or timestamps (False).
        tzinfo: The time zone of the timestamps.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        composite: Any = None,
        t_axis: int = 0,
        scale_name: str | None = None,
        sample_rate: Decimal | int | float | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._sample_rate_: Decimal | float | None = None
        self._time_axis: HDF5Dataset | None = None

        self._t_axis: int | None = None
        self.scale_name: str = "time_axis"

        # Parent Attributes #
        super().__init__(self, init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                t_axis=t_axis,
                scale_name=scale_name,
                sample_rate=sample_rate,
                precise=precise,
                tzinfo=tzinfo,
                **kwargs,
            )

    @property
    def _sample_rate(self) -> Decimal | h5py.Empty:
        """The sample rate of this timeseries."""
        return self.time_axis.sample_rate if self.time_axis is not None else self._sample_rate_

    @_sample_rate.setter
    def _sample_rate(self, value: Decimal | int | float | None) -> None:
        if value is not None and not isinstance(value, Decimal):
            value = Decimal(value)
        self._sample_rate_ = value
        if self.time_axis is not None:
            self.time_axis.sample_rate = value

    @property
    def t_axis(self) -> int:
        """The axis which the time axis is attached."""
        return self.get_t_axis()

    @property
    def time_axis(self) -> HDF5Dataset | None:
        """Loads and returns the time axis."""
        if self._time_axis is None:
            self._time_axis = self.composite.axes[self.t_axis][self.scale_name]
        return self._time_axis.components["axis"]

    @time_axis.setter
    def time_axis(self, value: HDF5Dataset | None) -> None:
        self._time_axis = value

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        t_axis: int = 0,
        scale_name: str | None = None,
        sample_rate: Decimal | int | float | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            composite: The object which this object is a component of.
            t_axis: The dim number of the time axis.
            scale_name: The scale name of the time axis.
            sample_rate: The sample rate of the data in Hz.
            precise: Determines if this frame returns nanostamps (True) or timestamps (False).
            tzinfo: The time zone of the timestamps.
            **kwargs: Keyword arguments for inheritance.
        """
        if t_axis is not None:
            self._t_axis

        if sample_rate is not None:
            self._sample_rate = sample_rate

        if scale_name is not None:
            self.scale_name = scale_name

        super().construct(composite=composite, precise=precise, tzinfo=tzinfo, **kwargs)

    def get_t_axis(self) -> int:
        """Gets the dim number of the time axis.

        Returns:
            The dim number of the time axis.
        """
        if self._t_axis is not None:
            return self._t_axis
        else:
            return self.composite.attributes["t_axis"]

    def set_t_axis_local(self, value: int | None) -> None:
        """Sets the t axis to a value, but does not update the attribute in the file.

        Args:
            value: The dim number of the time axis.
        """
        self._t_axis = value

    def set_t_axis_attribute(self, value: int | None) -> None:
        """Sets the t axis to a value and updates the attribute in the file.

        Args:
            value: The dim number of the time axis.
        """
        self.composite.attributes.set_attribute("t_axis", value)
        self._t_axis = None

    def set_time_axis(self, t_axis: int | None = None, scale_name: str | None = None) -> None:
        """Sets the time axis for this component.

        Args:
            t_axis: The dim number of the time axis.
            scale_name: The scale name of the time axis.
        """
        if scale_name is not None:
            self.scale_name = scale_name

        if t_axis is not None:
            self._t_axis = t_axis

        self._time_axis = self.composite.axes[self.t_axis][self.scale_name]

    # Axes
    def create_time_axis(
        self,
        t_axis: int | None = None,
        scale_name: str | None = None,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        rate: int | float | None = None,
        size: int | None = None,
        axis: int | None = None,
        datetimes: Iterable[datetime.datetime | float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates and fills the time axis, gives an error if it already exists.

        Args:
            t_axis: The dim number of the time axis.
            scale_name: The scale name of the time axis.
            start: The start of the time axis.
            stop: The end of the time axis.
            step: The interval between each time of the axis.
            rate: The frequency of the time of the axis.
            size: The number of times in the axis.
            axis: The axis the time axis will be attached to.
            datetimes: The datetimes to populate this axis.
            **kwargs: The keyword arguments for the TimeAxis.
        """
        if t_axis is not None:
            self._t_axis = t_axis

        if scale_name is not None:
            self.scale_name = scale_name

        self._time_axis = self.composite.create_axis(
            dim=self.t_axis,
            name=self.scale_name,
            start=start,
            stop=stop,
            step=step,
            rate=self._sample_rate if rate is None else rate,
            size=size,
            datetimes=datetimes,
            scale_name=self.scale_name,
            require=True,
            file=self.file,
            **kwargs,
        )

    def require_time_axis(
        self,
        t_axis: int | None = None,
        scale_name: str | None = None,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        rate: int | float | None = None,
        size: int | None = None,
        axis: int | None = None,
        datetimes: Iterable[datetime.datetime | float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates and fills the time axis if it does not exists.

        Args:
            t_axis: The dim number of the time axis.
            scale_name: The scale name of the time axis.
            start: The start of the time axis.
            stop: The end of the time axis.
            step: The interval between each time of the axis.
            rate: The frequency of the time of the axis.
            size: The number of times in the axis.
            axis: The axis the time axis will be attached to.
            datetimes: The datetimes to populate this axis.
            **kwargs: The keyword arguments for the TimeAxis.
        """
        if t_axis is not None:
            self._t_axis = t_axis

        if scale_name is not None:
            self.scale_name = scale_name

        self._time_axis = self.composite.require_axis(
            dim=self.t_axis,
            name=self.scale_name,
            scale_name=self.scale_name,
            require=True,
            file=self.composite.file,
            component_kwargs={
                "axis": {
                    "start": start,
                    "stop": stop,
                    "step": step,
                    "rate": self._sample_rate_ if rate is None else rate,
                    "size": size,
                    "datetimes": datetimes,
                }
            },
            **kwargs,
        )

    # Data
    def create_component(
        self,
        t_axis: int | None = None,
        scale_name: str | None = None,
        sample_rate: Decimal | int | float | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates all the required parts of the dataset for this component, errors if a part already exists.

        Args:
            t_axis: The dim number of the time axis.
            scale_name: The scale name of the time axis.
            sample_rate: The sample rate of the data in Hz.
            **kwargs: The keyword to create the time axis.
        """
        if t_axis is not None:
            self._t_axis = t_axis

        if scale_name is not None:
            self.scale_name = scale_name

        if sample_rate is not None:
            self._sample_rate = sample_rate

        self.create_time_axis(**kwargs)

    def create(
        self,
        t_axis: int | None = None,
        scale_name: str | None = None,
        sample_rate: Decimal | int | float | None = None,
        **kwargs: Any,
    ) -> HDF5Dataset:
        """Creates both the data and this component, errors if a part already exists.

        Args:
            data: The data to fill in this timeseries.
            start: The start of the data as a timestamp.
            sample_rate: The sample rate of the data in Hz.
            channels: An object to build the channel axis from.
            samples: An object to build the sample axis from.
            timestamps: An object to build the time axis from.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        self.composite.create_data(**kwargs)
        self.create_component(t_axis=t_axis, scale_name=scale_name, sample_rate=sample_rate)
        return self.composite

    def require_component(
        self,
        t_axis: int | None = None,
        scale_name: str | None = None,
        sample_rate: Decimal | int | float | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates all the required parts of the dataset for this component if it does not exists.

        Args:
            t_axis: The dim number of the time axis.
            scale_name: The scale name of the time axis.
            sample_rate: The sample rate of the data in Hz.
            **kwargs: The keyword to require the time axis.
        """
        if t_axis is not None:
            self._t_axis = t_axis

        if scale_name is not None:
            self.scale_name = scale_name

        if sample_rate is not None:
            self._sample_rate = sample_rate

        self.require_time_axis(**kwargs)

    def require(
        self,
        t_axis: int | None = None,
        scale_name: str | None = None,
        sample_rate: Decimal | int | float | None = None,
        **kwargs: Any,
    ) -> HDF5Dataset:
        """Creates both the data and this component if it does not exists.

        Args:
            data: The data to fill in this timeseries.
            start: The start of the data as a timestamp.
            sample_rate: The sample rate of the data in Hz.
            channels: An object to build the channel axis from.
            samples: An object to build the sample axis from.
            timestamps: An object to build the time axis from.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        self.composite.require_data(**kwargs)
        self.require_component(t_axis=t_axis, scale_name=scale_name, sample_rate=sample_rate)
        return self.composite

    def set_data_component(self, data: np.ndarray, **kwargs: Any) -> None:
        """Sets the data pertaining to this component.

        Args:
            data: The replacement data.
            **kwargs: The keyword arguments for creating the component.
        """
        if self._time_axis is None:
            self.require_component(**kwargs)
        else:
            self._time_axis.set_data(data=data, **kwargs)

    def set_data(
        self,
        data: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Sets the data of the timeseries and creates it if it does not exist.

        Args:
            data: The replacement data.
            timestamps: The replacement data for the time axis.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        self.composite.require_data(data=data, **kwargs)
        self.set_data_component(data=timestamps)

    def append_component(self, data: np.ndarray, **kwargs: Any) -> None:
        """Append data to this component.

        Args:
            data: The data to append to the time axis.
            **kwargs: The keyword arguments for appending data to the time axis.
        """
        self.time_axis.append(data=data, **kwargs)

    def append(
        self,
        data: np.ndarray,
        time_axis: np.ndarray = np.ndarray,  # Todo: Infer timestamps when none is given.
        axis: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Append data to the dataset along a specified axis and the given axis object.

        Args:
            data: The data to append.
            axis: The axis to append the data along.
            **kwargs: The data to append to an axis. The kwarg name should be the name of the axis.
        """
        axis = self.t_axis if axis is None else axis
        self.composite.append_data(data=data, axis=axis)
        self.append_component(data=time_axis, **kwargs)
