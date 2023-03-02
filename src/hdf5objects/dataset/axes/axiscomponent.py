""" axiscomponent.py
A component and map for a HDF5Dataset which gives axis (scale) functionality.
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
from typing import Any

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache
from dspobjects.dataclasses import IndexValue, FoundRange
import numpy as np

# Local Packages #
from ...hdf5bases import DatasetMap
from ..basedatasetcomponent import BaseDatasetComponent


# Definitions #
# Classes #
class AxisComponent(BaseDatasetComponent):
    """A component for a HDF5Dataset which gives axis (scale) functionality.

    Args:
        composite: The object which this object is a component of.
        start: The start of the axis.
        stop: The end of the axis.
        step: The interval between each datum of the axis.
        rate: The frequency of the data of the axis.
        size: The number of datum in the axis.
        create: Determines if the axis should be created and filled.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        composite: Any = None,
        start: int | float | None = None,
        stop: int | float | None = None,
        step: int | float | None = None,
        rate: float | None = None,
        size: int | None = None,
        create: bool | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
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
                create=create,
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
        composite: Any = None,
        start: int | float | None = None,
        stop: int | float | None = None,
        step: int | float | None = None,
        rate: float | None = None,
        size: int | None = None,
        create: bool | None = None,
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
            create: Determines if the axis should be created and filled.
            **kwargs: Keyword arguments for inheritance.
        """
        super().construct(composite=composite, **kwargs)

        if create and start is not None and size != 0:
            self.from_range(start, stop, step, rate, size)

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
            self.composite.set_data(data=np.arange(start, stop, step), **kwargs)
        else:
            self.composite.set_data(data=np.linspace(start, stop, size), **kwargs)

    # File
    def refresh(self) -> None:
        """Reloads the axis and clears the caches."""
        self.get_start.clear_cache()
        self.get_end.clear_cache()

    # Getters/Setters
    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_start(self) -> Any:
        """Get the first element of this axis, using caching.

        Returns:
            The first element of this axis.
        """
        return self.composite[0]

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_end(self) -> Any:
        """Get the last element of this axis, using caching.

        Returns:
            The last element of this axis.
        """
        return self.composite[-1]

    def get_intervals(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> np.ndarray:
        """Get the intervals between each datum of the axis.

        Args:
            start: The start index to get the intervals.
            stop: The last index to get the intervals.
            step: The step of the indices to the intervals.

        Returns:
            The intervals between each datum of the axis.
        """
        return np.ediff1d(self.composite.all_data[slice(start, stop, step)])

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
        samples = self.composite.shape[0]
        if item < self.composite.start:
            if tails:
                return IndexValue(0, self.start)
        elif item > self.end:
            if tails:
                return IndexValue(samples - 1, self.end)
        else:
            item = int(np.searchsorted(self.composite.all_data, item, side="right") - 1)
            if approx or item == self.composite.all_data[item]:
                return IndexValue(item, self.composite.all_data[item])
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
            stop_index = self.composite.shape[0] - 1
        else:
            stop_index, _ = self.find_index(item=stop, approx=approx, tails=tails)

        if start_index is None and stop_index is None:
            return FoundRange(None, None, None)
        else:
            data = self.composite.all_data[slice(start=start_index, stop=stop_index, step=step)]

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
            self.composite[start:stop:step] += shift
        self.refresh()

class AxisMap(DatasetMap):
    """An outline which defines an HDF5Dataset as an Axis."""
    default_kwargs: dict[str, Any] = {}
    default_component_types = {
        "axis": (AxisComponent, {}),
    }
