"""geometrycomponent.py
A component for a HDF5Dataset which gives it spatial geometry functionality.
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
from typing import Any

# Third-Party Packages #
import h5py
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Dataset
from ..basedatasetcomponent import BaseDatasetComponent


# Definitions #
# Classes #
class GeometryComponent(BaseDatasetComponent):
    """A component for a HDF5Dataset which gives it spatial geometric functionality.

    Attributes:
        _label_axis: The channel labels axis object.
        _coordinate_axis: The channel coordinates axis object.
        _c_axis: The dim number of the channel axis.
        label_scale_name: The scale name of the label axis.
        coordinate_scale_name: The scale name of the coordinate axis.

    Args:
        composite: The object which this object is a component of.
        c_axis: The dim number of the channel axis.
        label_scale_name: The scale name of the label axis.
        coordinate_scale_name: The scale name of the coordinate axis.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        composite: Any = None,
        c_axis: int = 1,
        label_scale_name: str | None = None,
        coordinate_scale_name: str | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._label_axis: HDF5Dataset | None = None
        self._coordinate_axis: HDF5Dataset | None = None

        self._c_axis: int | None = None
        self.label_scale_name: str = "label_axis"
        self.coordinate_scale_name: str = "coordinate_axis"

        # Parent Attributes #
        super().__init__(self, init=False)

        # Object Construction #
        if init:
            self.construct(
                composite=composite,
                c_axis=c_axis,
                label_scale_name=label_scale_name,
                coordinate_scale_name=coordinate_scale_name,
                **kwargs,
            )

    @property
    def c_axis(self) -> int:
        """The axis which the channel axis is attached."""
        return self.get_c_axis()

    @property
    def label_axis(self) -> HDF5Dataset | None:
        """Loads and returns the label axis."""
        if self._label_axis is None:
            self._label_axis = self.composite.axes[self.c_axis][self.label_scale_name]
        return self._label_axis.components["axis"]

    @property
    def coordinate_axis(self) -> HDF5Dataset | None:
        """Loads and returns the coordinate axis."""
        if self._coordinate_axis is None:
            self._coordinate_axis = self.composite.axes[self.c_axis][self.coordinate_scale_name]
        return self._coordinate_axis.components["axis"]

    @label_axis.setter
    def label_axis(self, value: HDF5Dataset | None) -> None:
        self._label_axis = value

    @coordinate_axis.setter
    def coordinate_axis(self, value: HDF5Dataset | None) -> None:
        self._coordinate_axis = value

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        composite: Any = None,
        c_axis: int = 1,
        label_scale_name: str | None = None,
        coordinate_scale_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.
        Args:
            composite: The object which this object is a component of.
            c_axis: The dim number of the channel axis.
            label_scale_name: The scale name of the label axis.
            coordinate_scale_name: The scale name of the coordinate axis.
            **kwargs: Keyword arguments for inheritance.
        """
        if c_axis is not None:
            self._c_axis

        if label_scale_name is not None:
            self.label_scale_name = label_scale_name

        if coordinate_scale_name is not None:
            self.coordinate_scale_name = coordinate_scale_name

        super().construct(composite=composite, **kwargs)

    def get_c_axis(self) -> int:
        """Gets the dim number of the channel axis.

        Returns:
            The dim number of the channel axis.
        """
        if self._c_axis is not None:
            return self._c_axis
        else:
            return self.composite.attributes["c_axis"]

    def set_c_axis_local(self, value: int | None) -> None:
        """Sets the c axis to a value, but does not update the attribute in the file.

        Args:
            value: The dim number of the channel axis.
        """
        self._c_axis = value

    def set_c_axis_attribute(self, value: int | None) -> None:
        """Sets the c axis to a value and updates the attribute in the file.

        Args:
            value: The dim number of the channel axis.
        """
        self.composite.attributes.set_attribute("c_axis", value)
        self._c_axis = None

    def set_label_axis(self, c_axis: int | None = None, scale_name: str | None = None) -> None:
        """Sets the label axis for this component.

        Args:
            c_axis: The dim number of the channel axis.
            scale_name: The scale name of the label axis.
        """
        if scale_name is not None:
            self.label_scale_name = scale_name

        if c_axis is not None:
            self._c_axis = c_axis

        self._label_axis = self.composite.axes[self.c_axis][self.label_scale_name]

    def set_coordinate_axis(self, c_axis: int | None = None, scale_name: str | None = None) -> None:
        """Sets the coordinate axis for this component.

        Args:
            c_axis: The dim number of the channel axis.
            scale_name: The scale name of the coordinate axis.
        """
        if scale_name is not None:
            self.coordinate_scale_name = scale_name

        if c_axis is not None:
            self._c_axis = c_axis

        self._coordinate_axis = self.composite.axes[self.c_axis][self.coordinate_scale_name]

    #Axes
    def create_geometry_axes(
        self,
        c_axis: int | None = None,
        label_scale_name: str | None = None,
        coordinate_scale_name: str | None = None,
        label_data: np.ndarray | None = None,
        coordinate_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates and fills the time axis, gives an error if it already exists.

        Args:
            c_axis: The dim number of the channel axis.
            label_scale_name: The scale name of the label axis.
            coordinate_scale_name: The scale name of the coordinate axis.
            axis: The axis the time axis will be attached to.
            datetimes: The datetimes to populate this axis.
            **kwargs: The keyword arguments for the TimeAxis.
        """
        if c_axis is not None:
            self._c_axis = c_axis

        if label_scale_name is not None:
            self.label_scale_name = label_scale_name

        if coordinate_scale_name is not None:
            self.coordinate_scale_name = coordinate_scale_name

        self._label_axis = self.composite.create_axis(
            dim=self.c_axis,
            name=self.label_scale_name,
            scale_name=self.label_scale_name,
            data=label_data,
            require=True,
            file=self.file,
            **kwargs,
        )

        self._coordinate_axis = self.composite.create_axis(
            dim=self.c_axis,
            name=self.coordinate_scale_name,
            scale_name=self.coordinate_scale_name,
            data=coordinate_data,
            require=True,
            file=self.file,
            **kwargs,
        )

    def require_geometry_axes(
        self,
        c_axis: int | None = None,
        label_scale_name: str | None = None,
        coordinate_scale_name: str | None = None,
        label_data: str | None = None,
        coordinate_data: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates and fills the time axis if it does not exists.

        Args:
            c_axis: The dim number of the channel axis.
            label_scale_name: The scale name of the label axis.
            coordinate_scale_name: The scale name of the coordinate axis.
            axis: The axis the time axis will be attached to.
            datetimes: The datetimes to populate this axis.
            **kwargs: The keyword arguments for the TimeAxis.
        """
        if c_axis is not None:
            self._c_axis = c_axis

        if label_scale_name is not None:
            self.label_scale_name = label_scale_name

        if coordinate_scale_name is not None:
            self.coordinate_scale_name = coordinate_scale_name

        self._label_axis = self.composite.require_axis(
            dim=self.c_axis,
            name=self.label_scale_name,
            scale_name=self.label_scale_name,
            require=True,
            file=self.composite.file,
            component_kwargs={
                "axis": {
                    "data": label_data
                }
            },
            **kwargs,
        )

        self._coordinate_axis = self.composite.require_axis(
            dim=self.c_axis,
            name=self.coordinate_scale_name,
            scale_name=self.coordinate_scale_name,
            require=True,
            file=self.composite.file,
            component_kwargs={
                "axis": {
                    "data": coordinate_data
                }
            },
            **kwargs,
        )

    # Data
    def create_component(
        self,
        c_axis: int | None = None,
        label_scale_name: str | None = None,
        coordinate_scale_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates all the required parts of the dataset for this component, errors if a part already exists.

        Args:
            c_axis: The dim number of the time axis.
            label_scale_name: The scale name of the time axis.
            coordinate_scale_name: The scale name of the time axis.
            sample_rate: The sample rate of the data in Hz.
            **kwargs: The keyword to create the time axis.
        """
        if c_axis is not None:
            self._c_axis = c_axis

        if label_scale_name is not None:
            self.label_scale_name = label_scale_name

        if coordinate_scale_name is not None:
            self.coordinate_scale_name = coordinate_scale_name

        self.create_geometry_axes(**kwargs)

    def require_component(
        self,
        c_axis: int | None = None,
        label_scale_name: str | None = None,
        coordinate_scale_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates all the required parts of the dataset for this component if it does not exists.

        Args:
            c_axis: The dim number of the time axis.
            label_scale_name: The scale name of the time axis.
            coordinate_scale_name: The scale name of the time axis.
            **kwargs: The keyword to require the time axis.
        """
        if c_axis is not None:
            self._c_axis = c_axis

        if label_scale_name is not None:
            self.label_scale_name = label_scale_name

        if coordinate_scale_name is not None:
            self.coordinate_scale_name = coordinate_scale_name

        self.require_geometry_axes(**kwargs)
