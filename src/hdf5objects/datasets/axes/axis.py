""" axis.py
An HDF5 Dataset subclass whose purpose is to be an Axis.
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
from collections.abc import Mapping
from typing import Any

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache
from dspobjects.dataclasses import IndexValue, FoundRange
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Map, DatasetMap, HDF5Dataset


# Definitions #
# Classes #
class AxisMap(DatasetMap):
    """A map for the Axis object."""
    default_kwargs: dict[str, Any] = {}


class Axis(HDF5Dataset):
    """A HDF5Dataset whose primary role is to be an axis (scale).

    Class_Attributes:
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
        s_name: The name of the axis (scale).
        require: Determines if the axis should be created and filled.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the HDF5Dataset.
    """
    default_map: HDF5Map = AxisMap()
    default_scale_name: str | None = None  # Set this to the name of the axis

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self, 
        start: int | float | None = None, 
        stop: int | float | None = None, 
        step: int | float | None = None, 
        rate: float | None = None, 
        size: int | None = None,
        s_name: str | None = None, 
        require: bool | None = None,
        init: bool = True, 
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.default_kwargs: Mapping[str, Any] = self.default_map.kwargs

        self._scale_name = self.default_scale_name

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(
                start=start, 
                stop=stop, 
                step=step, 
                rate=rate, 
                size=size,
                s_name=s_name, 
                require=require,
                **kwargs,
            )

    @property
    def start(self) -> Any:
        """Get the first element of this axis."""
        try:
            return self.get_start.caching_call()
        except AttributeError:
            return self.get_start()

    @property
    def end(self) -> Any:
        """Get the last element of this axis."""
        try:
            return self.get_start.caching_call()
        except AttributeError:
            return self.get_start()

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        start: int | float | None = None,
        stop: int | float | None = None,
        step: int | float | None = None,
        rate: float | None = None,
        size: int | None = None,
        s_name: str | None = None,
        require: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.
        
        Args:
            start: The start of the axis.
            stop: The end of the axis.
            step: The interval between each datum of the axis.
            rate: The frequency of the data of the axis.
            size: The number of datum in the axis.
            s_name: The name of the axis (scale).
            require: Determines if the axis should be created and filled.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        if s_name is not None:
            self._scale_name = s_name

        # Construct the dataset and handle creation here unless data is present.
        super().construct(require=False, **(self.default_kwargs | kwargs))

        if require and "data" not in kwargs:
            if start is not None and size != 0:
                self.from_range(start, stop, step, rate, size)
            else:
                self.require()

    def from_range(
        self,
        start: int | float | None = None,
        stop: int | float | None = None,
        step: int | float | None = None,
        rate: float | None = None,
        size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates the axis from a range style of input.
        
        Args:
            start: The start of the axis.
            stop: The end of the axis.
            step: The interval between each datum of the axis.
            rate: The frequency of the data of the axis
            size: The number of datum in the axis.
            **kwargs: The keyword arguments for the HDF5Dataset.
        """
        if step is None and rate is not None:
            step = 1 / rate

        if start is None:
            start = stop - step * size

        if stop is None:
            stop = start + step * size

        if step is not None:
            self.set_data(data=np.arange(start, stop, step), **kwargs)
        else:
            self.set_data(data=np.linspace(start, stop, size), **kwargs)

    # File
    def refresh(self) -> None:
        """Reloads the axis and clears the caches."""
        super().refresh()
        self.get_start.clear_cache()
        self.get_end.clear_cache()

    # Getters/Setters
    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_start(self) -> Any:
        """Get the first element of this axis, using caching.
        
        Returns:
            The first element of this axis.
        """
        with self:
            return self._dataset[0]

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_end(self) -> Any:
        """Get the last element of this axis, using caching.

        Returns:
            The last element of this axis.
        """
        with self:
            return self._dataset[-1]

    def get_intervals(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> np.ndarray:
        """Get the intervals between each datum of the axis.

        Args:
            start: The start index to get the intervals.
            stop: The last index to get the intervals.
            step: The step of the indices to the intervals.

        Returns:
            The intervals between each datum of the axis.
        """
        return np.ediff1d(self.all_data[slice(start, stop, step)])

    # Find
    def find_index(self, item: int | float, approx: bool = False, tails: bool = False) -> IndexValue:
        """Finds the index with the given value, can give an approximate index if the value is not present.

        Args:
            item: The item to find within this axis.
            approx: Determines if an approximate index will be given if the value is not present.
            tails: Determines if the first or last index will be give the requested item is outside the axis.

        Returns:
            The requested closest index and the value at that index.
        """
        samples = self.shape[0]
        if item < self.start:
            if tails:
                return IndexValue(0, self.start)
        elif item > self.end:
            if tails:
                return IndexValue(samples - 1, self.end)
        else:
            item = int(np.searchsorted(self.all_data, item, side="right") - 1)
            if approx or item == self.all_data[item]:
                return IndexValue(item, self.all_data[item])
            else:
                return IndexValue(None, None)

    def find_range(
        self,
        start: int | float | None = None,
        stop: int | float | None = None,
        step: int | float | None = None,
        approx: bool = False,
        tails: bool = False,
    ) -> FoundRange:
        """Finds the range on the axis inbetween two values, can give approximate values.

        Args:
            start: The first value to find for the range.
            stop: The last value to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the value is not present.
            tails: Determines if the first or last indices will be give the requested item is outside the axis.

        Returns:
            The data range on the axis and the start and stop indices.
        """
        if start is None:
            start_index = 0
        else:
            start_index, _ = self.find_index(item=start, approx=approx, tails=tails)

        if stop is None:
            stop_index = self.shape[0] - 1
        else:
            stop_index, _ = self.find_index(item=stop, approx=approx, tails=tails)

        if start_index is None and stop_index is None:
            return FoundRange(None, None, None)
        else:
            data = self.all_data[slice(start=start_index, stop=stop_index, step=step)]

            if step is not None and step != 1:
                stop_index = int(data.shape[0] * step + start_index)

            return FoundRange(data, start_index, stop_index)

    # Manipulation
    def shift(
        self,
        shift: int | float,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> None:
        """Shifts values over a range in the axis.

        Args:
            shift: The value to shift the values by.
            start: The first value to shift.
            stop: The last value to shift.
            step: The interval to apply the shift across the range.
        """
        with self:
            self._dataset[start:stop:step] += shift
        self.refresh()


# Assign Cyclic Definitions
AxisMap.default_type = Axis
