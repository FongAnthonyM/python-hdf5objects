""" timeseriesdataset.py
A Dataset designed to be a timeseries.
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
from collections.abc import Mapping, Iterable
import datetime
from typing import Any

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod
from dspobjects.dataclasses import FoundData, FoundTimeRange
import h5py
import numpy as np

# Local Packages #
from ..hdf5bases import HDF5Map, DatasetMap, HDF5Dataset
from .axes import ChannelAxisMap, ChannelAxis
from .axes import SampleAxisMap, SampleAxis
from .axes import TimeAxisMap, TimeAxis


# Definitions #
# Classes #
class TimeSeriesMap(DatasetMap):
    """A map for a timeseries dataset."""
    default_attribute_names: Mapping[str, str] = {"n_samples": "n_samples",
                                                  "c_axis": "c_axis",
                                                  "t_axis": "t_axis"}
    default_attributes: Mapping[str, Any] = {"n_samples": 0,
                                             "c_axis": 1,
                                             "t_axis": 0}
    default_map_names: Mapping[str, str] = {"channel_axis": "channel_axis",
                                            "sample_axis": "sample_axis",
                                            "time_axis": "time_axis"}
    default_maps: Mapping[str, HDF5Map] = {"channel_axis": ChannelAxisMap(),
                                           "sample_axis": SampleAxisMap(),
                                           "time_axis": TimeAxisMap()}


class TimeSeriesDataset(HDF5Dataset):
    """A Dataset designed to be a timeseries.

    Class Attributes:
        default_map: The default map that outlines this object in the HDF5 file.
        
    Attributes:
        _channel_axis: The channel axis for this dataset.
        _sample_axis: The sample axis for this dataset.
        _time_axis: The time axis for this dataset.
        channel_scale_name: The scale name for the channel axis. 
        sample_scale_name: The scale name for the sample axis. 
        time_scale_name: The scale name for the time axis. 
        
    Args:
        data: The data to fill in this timeseries.
        sample_rate: The sample rate of the data in Hz.
        channels: An object to build the channel axis from.
        samples: An object to build the sample axis from.
        timestamps: An object to build the time axis from.
        load: Determines if this object will load the timeseries from the file on construction.
        build: Determines if this object will create and fill the timeseries in the file on construction.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments to construct the base HDF5 dataset.
    """
    default_map: HDF5Map = TimeSeriesMap()

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self, 
        data: np.ndarray | None = None, 
        sample_rate: int | float | None = None,
        channels: ChannelAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        samples: SampleAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        timestamps: TimeAxis | np.ndarray | Iterable[datetime.datetime] | Mapping[str, Any] | None = None,
        load: bool = False, 
        build: bool = False, 
        init: bool = True, 
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._sample_rate: float | None = None
        self._channel_axis: ChannelAxis | None = None
        self._sample_axis: SampleAxis | None = None
        self._time_axis: TimeAxis | None = None
        
        self.channel_scale_name: str = "channel axis"
        self.sample_scale_name: str = "sample axis"
        self.time_scale_name: str = "time axis"

        # Object Construction #
        if init:
            self.construct(
                data=data, 
                sample_rate=sample_rate, 
                channels=channels, 
                samples=samples, 
                timestamps=timestamps, 
                load=load, 
                build=build, 
                **kwargs,
            )

    @property
    def sample_rate(self) -> float | h5py.Empty:
        """The sample rate of this timeseries."""
        return self._time_axis.sample_rate if self.time_axis is not None else self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self._sample_rate = value
        if self.time_axis is not None:
            self._time_axis.sample_rate = value

    @property
    def n_samples(self) -> int:
        """The number of samples in this timeseries."""
        return self.attributes["n_samples"]

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        self.attributes.set_attribute("n_samples", value)

    @property
    def c_axis(self) -> int:
        """The axis which the channel axis is attached."""
        return self.attributes["c_axis"]
    
    @c_axis.setter
    def c_axis(self, value: int) -> None:
        self.attributes.set_attribute("c_axis", value)

    @property
    def t_axis(self) -> int:
        """The axis which the time axis is attached."""
        return self.attributes["t_axis"]

    @t_axis.setter
    def t_axis(self, value: int) -> None:
        self.attributes.set_attribute("t_axis", value)
    
    @property
    def channel_axis(self) -> ChannelAxis | None:
        """Loads and returns the channel axis."""
        if self._channel_axis is None:
            self.load_axes()
        return self._channel_axis

    @property
    def sample_axis(self) -> SampleAxis | None:
        """Loads and returns the sample axis."""
        if self._sample_axis is None:
            self.load_axes()
        return self._sample_axis

    @property
    def time_axis(self) -> TimeAxis | None:
        """Loads and returns the time axis."""
        if self._time_axis is None:
            self.load_axes()
        return self._time_axis
    
    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        sample_rate: int | float | None = None,
        channels: ChannelAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        samples: SampleAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        timestamps: TimeAxis | np.ndarray | Iterable[datetime.datetime] | Mapping[str, Any] | None = None,
        load: bool = False,
        build: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The data to fill in this timeseries.
            sample_rate: The sample rate of the data in Hz.
            channels: An object to build the channel axis from.
            samples: An object to build the sample axis from.
            timestamps: An object to build the time axis from.
            load: Determines if this object will load the timeseries from the file on construction.
            build: Determines if this object will create and fill the timeseries in the file on construction.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.
        """
        if data is not None:
            kwargs["data"] = data
            kwargs["build"] = build

        super().construct(build=build, make_axes=False, **kwargs)

        if sample_rate is not None:
            self._sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]
        
        if load and self.exists:
            self.load()

        if build and self.exists:
            self.construct_axes(channels=channels, samples=samples, timestamps=timestamps)

    def construct_axes(
        self,
        channels: ChannelAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        samples: SampleAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        timestamps: TimeAxis | np.ndarray | Iterable[datetime.datetime] | Mapping[str, Any] | None = None,
    ) -> None:
        """Constructs the axes of this timeseries.

        Args:
            channels: An object to build the channel axis from.
            samples: An object to build the sample axis from.
            timestamps: An object to build the time axis from.
        """
        self.construct_channel_axis(channels=channels)
        self.construct_sample_axis(samples=samples)
        self.construct_time_axis(timestamps=timestamps)

    @singlekwargdispatchmethod("channels")
    def construct_channel_axis(
        self,
        channels: ChannelAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs the channel axis based on the arguments.

        Args:
            channels: An object to build the channel axis with.
            *args: Extra arguments to build the channel axis with.
            **kwargs: Extra keyword arguments to build the channel axis with.
        """
        raise TypeError(f"A {type(channels)} cannot be used to construct the channel axis.")

    @construct_channel_axis.register
    def _(self, channels: ChannelAxis) -> None:
        """Constructs the channel axis by attaching ChannelAxis object to this object.

        Args:
            channels: The ChannelAxis to attach.
        """
        self.attach_channel_axis(channels)

    @construct_channel_axis.register(np.ndarray)
    @construct_channel_axis.register(Iterable)
    def _(self, channels: np.ndarray | Iterable[int], **kwargs) -> None:
        """Constructs the channel axis by creating a ChannelAxis object from a numpy array.

        Args:
            channels: The numpy array to make into the channel axis.
        """
        if self._channel_axis is None:
            self.create_channel_axis(build=True, data=channels, **kwargs)
        else:
            self._channel_axis.set_data(channels, **kwargs)

    @construct_channel_axis.register(Mapping)
    def _(self, channels: Mapping[str, Any]) -> None:
        """Constructs the channel axis by creating a ChannelAxis object from a Mapping of kwargs.

        Args:
            channels: The keyword arguments to create a ChannelAxis from.
        """
        if self._channel_axis is None:
            self.create_channel_axis(**channels)
        else:
            self._channel_axis.from_range(**channels)

    @construct_channel_axis.register
    def _(self, channels: None, *args: Any, **kwargs: Any) -> None:
        """Constructs the channel axis by creating a ChannelAxis from given args and kwargs.

        Args:
            channels: None
            *args: Arguments to supply ChannelAxis.
            **kwargs: Keyword arguments to supply ChannelAxis.
        """
        if self._channel_axis is None:
            self.create_channel_axis(**kwargs)
        else:
            self._channel_axis.from_range(**kwargs)
    
    @singlekwargdispatchmethod("samples")
    def construct_sample_axis(
        self,
        samples: SampleAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs the sample axis based on the arguments.

        Args:
            samples: An object to build the sample axis with.
            *args: Extra arguments to build the sample axis with.
            **kwargs: Extra keyword arguments to build the sample axis with.
        """
        raise TypeError(f"A {type(samples)} cannot be used to construct the sample axis.")

    @construct_sample_axis.register
    def _(self, samples: SampleAxis) -> None:
        """Constructs the sample axis by attaching SampleAxis object to this object.

        Args:
            samples: The SampleAxis to attach.
        """
        self.attach_sample_axis(samples)

    @construct_sample_axis.register(np.ndarray)
    @construct_sample_axis.register(Iterable)
    def _(self, samples: np.ndarray | Iterable[int], **kwargs) -> None:
        """Constructs the sample axis by creating a SampleAxis object from a numpy array.

        Args:
            samples: The numpy array to make into the sample axis.
        """
        if self._sample_axis is None:
            self.create_sample_axis(build=True, data=samples, **kwargs)
        else:
            self._sample_axis.set_data(samples, **kwargs)

    @construct_sample_axis.register(Mapping)
    def _(self, samples: Mapping[str, Any]) -> None:
        """Constructs the sample axis by creating a SampleAxis object from a Mapping of kwargs.

        Args:
            samples: The keyword arguments to create a SampleAxis from.
        """
        if self._sample_axis is None:
            self.create_sample_axis(**samples)
        else:
            self._sample_axis.from_range(**samples)

    @construct_sample_axis.register
    def _(self, samples: None, *args: Any, **kwargs: Any) -> None:
        """Constructs the sample axis by creating a SampleAxis from given args and kwargs.

        Args:
            samples: None
            *args: Arguments to supply SampleAxis.
            **kwargs: Keyword arguments to supply SampleAxis.
        """
        if self._sample_axis is None:
            self.create_sample_axis(**kwargs)
        else:
            self._sample_axis.from_range(**kwargs)
    
    @singlekwargdispatchmethod("timestamps")
    def construct_time_axis(
        self, 
        timestamps: TimeAxis | np.ndarray | Iterable[datetime.datetime] | Mapping[str, Any] | None,
        *args: Any, 
        **kwargs: Any,
    ) -> None:
        """Constructs the time axis based on the arguments.

        Args:
            timestamps: An object to build the time axis with.
            *args: Extra arguments to build the time axis with.
            **kwargs: Extra keyword arguments to build the time axis with.
        """
        raise TypeError(f"A {type(timestamps)} cannot be used to construct the time axis.")

    @construct_time_axis.register
    def _(self, timestamps: TimeAxis) -> None:
        """Constructs the time axis by attaching TimeAxis object to this object.
        
        Args:
            timestamps: The TimeAxis to attach.
        """
        self.attach_time_axis(timestamps)

    @construct_time_axis.register(np.ndarray)
    @construct_time_axis.register(Iterable)
    def _(self, timestamps: np.ndarray | Iterable[datetime.datetime], **kwargs) -> None:
        """Constructs the time axis by creating a TimeAxis object from a numpy array.

        Args:
            timestamps: The numpy array to make into the time axis.
        """
        if self._time_axis is None:
            self.create_time_axis(datetimes=timestamps, **kwargs)
        else:
            self._time_axis.from_datetimes(timestamps, **kwargs)

    @construct_time_axis.register(Mapping)
    def _(self, timestamps: Mapping[str, Any]) -> None:
        """Constructs the time axis by creating a TimeAxis object from a Mapping of kwargs.

        Args:
            timestamps: The keyword arguments to create a TimeAxis from.
        """
        if self._time_axis is None:
            self.create_time_axis(**timestamps)
        else:
            self._time_axis.from_range(**timestamps)

    @construct_time_axis.register
    def _(self, timestamps: None, *args: Any, **kwargs: Any) -> None:
        """Constructs the time axis by creating a TimeAxis from given args and kwargs.

        Args:
            timestamps: None
            *args: Arguments to supply TimeAxis.
            **kwargs: Keyword arguments to supply TimeAxis.
        """
        if self._time_axis is None:
            self.create_time_axis(**kwargs)
        else:
            self._time_axis.from_range(**kwargs)

    # File
    def create(
        self,
        data: np.ndarray | None = None,
        start: datetime.datetime | float | None = None,
        sample_rate: int | float | None = None,
        channels: ChannelAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        samples: SampleAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        timestamps: TimeAxis | np.ndarray | Iterable[datetime.datetime] | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "TimeSeriesDataset":
        """Creates the TimeSeriesDataset by creating the dataset and axes.

        Args:
            data: The data to fill in this timeseries.
            start: The start of the data as a timestamp.
            sample_rate: The sample rate of the data in Hz.
            channels: An object to build the channel axis from.
            samples: An object to build the sample axis from.
            timestamps: An object to build the time axis from.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        super().create(data=data, **kwargs)

        if sample_rate is not None:
            self._sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]

        if timestamps is None and start is not None:
            timestamps = {"start": start, "rate": self._sample_rate, "size": self.n_samples}

        self.construct_axes(channels=channels, samples=samples, timestamps=timestamps)
        self._time_axis.sample_rate = self._sample_rate

        return self

    def require(
        self,
        data: np.ndarray | None = None,
        start: datetime.datetime | float | None = None,
        sample_rate: int | float | None = None,
        channels: ChannelAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        samples: SampleAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        timestamps: TimeAxis | np.ndarray | Iterable[datetime.datetime] | Mapping[str, Any] | None = None,
        make_axes: bool = True,
        **kwargs: Any,
    ) -> "TimeSeriesDataset":
        """Creates the TimeSeriesDataset by creating the dataset and axes if it does not exist.

        Args:
            data: The data to fill in this timeseries.
            start: The start of the data as a timestamp.
            sample_rate: The sample rate of the data in Hz.
            channels: An object to build the channel axis from.
            samples: An object to build the sample axis from.
            timestamps: An object to build the time axis from.
            make_axes: Determines if the axes will be created.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        existed = self.exists
        super().require(data=data, **kwargs)

        if sample_rate is not None:
            self._sample_rate = sample_rate

        if data is not None:
            self.n_samples = data.shape[self.t_axis]

        if existed:
            self.load_axes()
            if channels is not None:
                self.construct_channel_axis(channels)
            if samples is not None:
                self.construct_sample_axis(samples)
            if timestamps is not None:
                self.construct_time_axis(timestamps)
            self._time_axis.sample_rate = self._sample_rate
        elif make_axes:
            if timestamps is None and start is not None:
                timestamps = {"start": start, "rate": self._sample_rate, "size": self.n_samples}

            self.construct_axes(channels=channels, samples=samples, timestamps=timestamps)
        return self

    def set_data(
        self,
        data: np.ndarray | None = None,
        sample_rate: int | float | None = None,
        channels: ChannelAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        samples: SampleAxis | np.ndarray | Iterable[int] | Mapping[str, Any] | None = None,
        timestamps: TimeAxis | np.ndarray | Iterable[datetime.datetime] | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Sets the data of the timeseries and creates it if it does not exist.

        Args:
            data: The data to fill in this timeseries.
            sample_rate: The sample rate of the data in Hz.
            channels: An object to build the channel axis from.
            samples: An object to build the sample axis from.
            timestamps: An object to build the time axis from.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        if not self.exists:
            self.require(data=data, sample_rate=sample_rate,
                         channels=channels, samples=samples, timestamps=timestamps, **kwargs)
        else:
            self.replace_data(data=data)

            if sample_rate is not None:
                self._sample_rate = sample_rate

            if data is not None:
                self.n_samples = data.shape[self.t_axis]

            self.construct_axes(channels=channels, samples=samples, timestamps=timestamps)
            self._time_axis.sample_rate = self._sample_rate

    def load(self) -> None:
        """Loads this time series dataset and its axes."""
        super().load()
        self.load_axes()

    def standardize_attributes(self) -> None:
        """Ensures that the attributes have the correct values based on the contained data."""
        self.attributes["n_samples"] = self.get_shape()[self.t_axis]
    
    def clear_all_caches(self, **kwargs: Any) -> None:
        """Clears all caches in this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the clear caches method.
        """
        self.attributes.clear_caches(**kwargs)
        self.clear_caches(**kwargs)
        if self._channel_axis is not None:
            self._channel_axis.clear_caches(**kwargs)
        if self._sample_axis is not None:
            self._sample_axis.clear_caches(**kwargs)
        if self._time_axis is not None:
            self._time_axis.clear_caches(**kwargs)
    
    def enable_all_caching(self, **kwargs: Any) -> None:
        """Enables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the enable caching method.
        """
        self.attributes.enable_caching(**kwargs)
        self.enable_caching(**kwargs)
        if self._channel_axis is not None:
            self._channel_axis.enable_caching(**kwargs)
        if self._sample_axis is not None:
            self._sample_axis.enable_caching(**kwargs)
        if self._time_axis is not None:
            self._time_axis.enable_caching(**kwargs)

    def disable_all_caching(self, **kwargs: Any) -> None:
        """Disables caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the disable caching method.
        """
        self.attributes.disable_caching(**kwargs)
        self.disable_caching(**kwargs)
        if self._channel_axis is not None:
            self._channel_axis.disable_caching(**kwargs)
        if self._sample_axis is not None:
            self._sample_axis.disable_caching(**kwargs)
        if self._time_axis is not None:
            self._time_axis.disable_caching(**kwargs)
        
    def timeless_all_caching(self, **kwargs: Any) -> None:
        """Allows timeless caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timeless caching method.
        """
        self.attributes.timeless_caching(**kwargs)
        self.timeless_caching(**kwargs)
        if self._channel_axis is not None:
            self._channel_axis.timeless_caching(**kwargs)
        if self._sample_axis is not None:
            self._sample_axis.timeless_caching(**kwargs)
        if self._time_axis is not None:
            self._time_axis.timeless_caching(**kwargs)
        
    def timed_all_caching(self, **kwargs: Any) -> None:
        """Allows timed caching on this object and all contained objects.

        Args:
            **kwargs: The keyword arguments for the timed caching method.
        """
        self.attributes.timed_caching(**kwargs)
        self.timed_caching(**kwargs)
        if self._channel_axis is not None:
            self._channel_axis.timed_caching(**kwargs)
        if self._sample_axis is not None:
            self._sample_axis.timed_caching(**kwargs)
        if self._time_axis is not None:
            self._time_axis.timed_caching(**kwargs)

    def set_all_lifetimes(self, lifetime: int | float | None, **kwargs: Any) -> None:
        """Sets the lifetimes on this object and all contained objects.

        Args:
            lifetime: The lifetime to set all the caches to.
            **kwargs: The keyword arguments for the lifetime caching method.
        """
        self.attributes.set_lifetimes(lifetime=lifetime, **kwargs)
        self.set_lifetimes(lifetime=lifetime, **kwargs)
        if self._channel_axis is not None:
            self.channel_axis.set_lifetimes(lifetime=lifetime, **kwargs)
        if self._sample_axis is not None:
            self.sample_axis.set_lifetimes(lifetime=lifetime, **kwargs)
        if self._time_axis is not None:
            self.time_axis.set_lifetimes(lifetime=lifetime, **kwargs)

    # Axes
    def create_channel_axis(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int | None = 1,
        rate: float | None = None,
        size: int | None = None,
        axis: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates a channel axis for this time series.

        Args:
            start: The start channel.
            stop: The end channel.
            step: The interval between each channel of the axis.
            rate: The frequency of the channels.
            size: The number of channels in the axis.
            axis: The axis the channel axis will be attached to.
            **kwargs: The keyword arguments for the ChannelAxis.
        """
        axis = self.attributes.get_attribute("c_axis", None) if axis is None else axis
        size = self.shape[axis] if size is None else size
        if "name" not in kwargs:
            kwargs["name"] = self._full_name + "_" + self.map.map_names["channel_axis"]
        
        self._channel_axis = self.map["channel_axis"].construct_object(
            start=start,
            stop=stop,
            step=step,
            rate=rate,
            size=size,
            s_name=self.channel_scale_name,
            build=True,
            file=self._file,
            **kwargs,
        )
        self.attach_axis(self._channel_axis, axis)

    def attach_channel_axis(self, dataset: h5py.Dataset | ChannelAxis, axis: int | None = None) -> None:
        """Attaches a channel axis to this time series.

        Args:
            dataset: The ChannelAxis to attach.
            axis: The axis to attach the ChannelAxis.
        """
        if axis is None:
            axis = self.c_axis
        self.attach_axis(dataset, axis)
        self._channel_axis = dataset
        self.channel_scale_name = getattr(dataset, "scale_name", None)

    def detach_channel_axis(self, axis: int | None = None) -> None:
        """Detaches a channel axis from this time series.

        Args:
            axis: The axis to attach the ChannelAxis from.
        """
        if axis is None:
            axis = self.c_axis
        self.detach_axis(self.channel_axis, axis)
        self._channel_axis = None

    def create_sample_axis(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int | None = 1,
        rate: float | None = None,
        size: int | None = None,
        axis: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates a sample axis for this time series.

        Args:
            start: The start sample.
            stop: The end sample.
            step: The interval between each sample of the axis.
            rate: The frequency of the samples.
            size: The number of samples in the axis.
            axis: The axis the sample axis will be attached to.
            **kwargs: The keyword arguments for the SampleAxis.
        """
        axis = self.attributes.get_attribute("t_axis", None) if axis is None else axis
        size = self.shape[axis] if size is None else size
        if "name" not in kwargs:
            kwargs["name"] = self._full_name + "_" + self.map.map_names["sample_axis"]

        self._sample_axis = self.map["sample_axis"].type(
            start=start,
            stop=stop,
            step=step,
            rate=rate,
            size=size,
            s_name=self.sample_scale_name,
            build=True,
            file=self._file,
            **kwargs,
        )
        self.attach_axis(self._sample_axis, axis)

    def attach_sample_axis(self, dataset: h5py.Dataset | SampleAxis, axis: int | None = None) -> None:
        """Attaches a sample axis to this time series.

        Args:
            dataset: The SampleAxis to attach.
            axis: The axis to attach the SampleAxis.
        """
        if axis is None:
            axis = self.t_axis
        self.attach_axis(dataset, axis)
        self._sample_axis = dataset
        self.sample_scale_name = getattr(dataset, "scale_name", None)

    def detach_sample_axis(self, axis: int | None = None) -> None:
        """Detaches a sample axis from this time series.

        Args:
            axis: The axis to attach the SampleAxis from.
        """
        if axis is None:
            axis = self.t_axis
        self.detach_axis(self._sample_axis, axis)
        self._sample_axis = None

    def create_time_axis(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        rate: int | float | None = None,
        size: int | None = None,
        axis: int | None = None,
        datetimes: Iterable[datetime.datetime | float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates a time axis for this time series.

        Args:
            start: The start of the time axis.
            stop: The end of the time axis.
            step: The interval between each time of the axis.
            rate: The frequency of the time of the axis.
            size: The number of times in the axis.
            axis: The axis the time axis will be attached to.
            datetimes: The datetimes to populate this axis.
            **kwargs: The keyword arguments for the TimeAxis.
        """
        axis = self.attributes.get_attribute("t_axis", None) if axis is None else axis
        size = self.shape[axis] if size is None else size
        rate = self._sample_rate if rate is None else rate
        if "name" not in kwargs:
            kwargs["name"] = self._full_name + "_" + self.map.map_names["time_axis"]

        self._time_axis = self.map["time_axis"].type(
            start=start,
            stop=stop,
            step=step,
            rate=rate,
            size=size,
            datetimes=datetimes,
            s_name=self.time_scale_name,
            build=True,
            file=self._file,
            **kwargs,
        )
        self.attach_axis(self._time_axis, axis)

    def attach_time_axis(self, dataset: h5py.Dataset | TimeAxis, axis: int | None = None) -> None:
        """Attaches a time axis to this time series.

        Args:
            dataset: The TimeAxis to attach.
            axis: The axis to attach the TimeAxis.
        """
        if axis is None:
            axis = self.t_axis
        self.attach_axis(dataset, axis)
        self._time_axis = dataset
        self.time_scale_name = getattr(dataset, "scale_name", None)

    def detach_time_axis(self, axis: int | None = None) -> None:
        """Detaches a time axis from this time series.

        Args:
            axis: The axis to attach the TimeAxis from.
        """
        if axis is None:
            axis = self.t_axis
        self.detach_axis(self._time_axis, axis)
        self._time_axis = None

    def load_axes(self) -> None:
        """Loads the axes from file."""
        with self:
            if self.channel_scale_name in self._dataset.dims[self.c_axis]:
                dataset = self._dataset.dims[self.c_axis][self.channel_scale_name]
                self._channel_axis = self.map["channel_axis"].type(
                    dataset=dataset,
                    s_name=self.channel_scale_name,
                    file=self._file
                )

            if self.sample_scale_name in self._dataset.dims[self.t_axis]:
                dataset = self._dataset.dims[self.t_axis][self.sample_scale_name]
                self._sample_axis = self.map["sample_axis"].type(
                    dataset=dataset,
                    s_name=self.sample_scale_name,
                    file=self._file
                )

            if self.time_scale_name in self._dataset.dims[self.t_axis]:
                dataset = self._dataset.dims[self.t_axis][self.time_scale_name]
                self._time_axis = self.map["time_axis"].type(
                    dataset=dataset,
                    s_name=self.time_scale_name,
                    file=self._file
                )

    # Axis Getters
    def get_datetime(self, index: int) -> datetime.datetime:
        """Get a datetime at an index.

        Args:
            index: The index to get the datetime from.

        Returns:
            The request datetime.
        """
        return self.time_axis.get_datetime(index)

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
        return self.time_axis.get_timestamp_range(start=start, stop=stop, step=step)

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
        return self.time_axis.get_datetime_range(start=start, stop=stop, step=step)

    def get_time_intervals(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Get the intervals between each sample of the axis.

        Args:
            start: The start index to get the intervals.
            stop: The last index to get the intervals.
            step: The step of the indices to the intervals.

        Returns:
            The intervals between each datum of the axis.
        """
        return self.time_axis.get_intervals(start=start, stop=stop, step=step)

    # Find Range
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
        return self.time_axis.find_datetime_range(start=start, stop=stop, step=step, approx=approx, tails=tails)

    def find_data_range_sample(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = False,
        tails: bool = False,
    ) -> FoundData:
        """Finds the range of data inbetween two samples, can give approximate values.

        Args:
            start: The first sample to find for the range.
            stop: The last sample to find for the range.
            step: The step between samples in the range.
            approx: Determines if an approximate indices will be given if the samples are not present.
            tails: Determines if the first or last indices will be give the requested samples are outside the axis.

        Returns:
            The data range on the axis and the start and stop indices.
        """
        axis, start_index, stop_index = self.sample_axis.find_range(
            start=start,
            stop=stop,
            step=step,
            approx=approx,
            tails=tails,
        )

        if axis is None:
            return FoundData(None, None, None, None, None, None)
        else:
            with self:
                data = self._dataset[slice(start=start_index, stop=stop_index, step=step)]
                return FoundData(data, axis, axis[0], axis[-1], start_index, stop_index)

    def find_data_range_timestamp(
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
        axis, start_index, stop_index = self.time_axis.find_timestamp_range(
            start=start, 
            stop=stop, 
            step=step, 
            approx=approx, 
            tails=tails,
        )

        if axis is None:
            return FoundData(None, None, None, None, None, None)
        else:
            with self:
                data = self._dataset[slice(start=start_index, stop=stop_index, step=step)]
                return FoundData(data, axis, axis[0], axis[-1], start_index, stop_index)

    def find_data_range_datetime(
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
            The data range on the axis and the time axis as datetimes.
        """
        axis, start_index, stop_index = self.time_axis.find_datetime_range(
            start=start,
            stop=stop,
            step=step,
            approx=approx,
            tails=tails,
        )

        if axis is None:
            return FoundData(None, None, None, None, None, None)
        else:
            with self:
                data = self._dataset[slice(start=start_index, stop=stop_index, step=step)]
                return FoundData(data, axis, axis[0], axis[-1], start_index, stop_index)

    # Todo: Add Fill Methods

    # Manipulation
    def append(self, data: np.ndarray, axis: int | None = 0, **kwargs: np.ndarray) -> None:
        """Append data to the dataset along a specified axis and the given axis object.

        Args:
            data: The data to append.
            axis: The axis to append the data along.
            **kwargs: The data to append to an axis. The kwarg name should be the name of the axis.
        """
        axis = self.t_axis if axis is None else axis
        super().append(data=data, axis=axis)
        for name, axis_data in kwargs.items():
            axis = getattr(self, name)
            axis.append(data=axis_data)

    def shift_samples(
        self,
        shift: int | float,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> None:
        """Shifts the samples over a range in the axis.

        Args:
            shift: The value to shift the samples by.
            start: The first value to shift.
            stop: The last value to shift.
            step: The interval to apply the shift across the range.
        """
        self.sample_axis.shift(shift=shift, start=start, stop=stop, step=step)

    def shift_timestamps(
        self,
        shift: int | float,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> None:
        """Shifts the timestamps over a range in the axis.

        Args:
            shift: The value to shift the timestamps by.
            start: The first value to shift.
            stop: The last value to shift.
            step: The interval to apply the shift across the range.
        """
        self.time_axis.shift(shift=shift, start=start, stop=stop, step=step)


# Assign Cyclic Definitions
TimeSeriesMap.default_type = TimeSeriesDataset
