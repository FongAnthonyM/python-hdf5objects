""" multitypedataset.py

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
from bidict import bidict
import numpy as np

# Local Packages #
from ...hdf5bases import HDF5Map, DatasetMap, HDF5Dataset, HDF5Caster


# Definitions #
# Classes #
class MultiTypeDataset(HDF5Dataset):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    caster = HDF5Caster
    default_types: tuple[tuple[str, type]] = ()
    default_casting_kwargs: list[dict[str, Any]] | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        data: np.ndarray | None = None,
        types: tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        load: bool = False,
        require: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._types: tuple[tuple[str, type]] = self.default_types
        self._types_dict: bidict = bidict({name: i for i, (name, _) in enumerate(self.default_types)})
        self._dtypes: tuple[tuple, ...] = tuple(
            (name, self.caster.map_type(type_)) for name, type_ in self.default_types
        )

        self.casting_kwargs: list[dict[str, Any]] | None = None

        # Object Construction #
        if init:
            self.construct(
                data=data,
                types=types,
                casting_kwargs=casting_kwargs,
                load=load,
                require=require,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        types: tuple[tuple[str, type]] | None = None,
        casting_kwargs: tuple[dict[str, Any]] | None = None,
        load: bool = False,
        require: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The data to fill in this timeseries.
            load: Determines if this object will load the timeseries from the file on construction.
            require: Determines if this object will create and fill the timeseries in the file on construction.
            **kwargs: The keyword arguments to construct the base HDF5 dataset.
        """
        if types is not None:
            self.set_types(types)

        if casting_kwargs is not None:
            self.casting_kwargs = casting_kwargs
        elif self.default_casting_kwargs is None:
            self.casting_kwargs = ([{}] * len(self._types))
        else:
            self.casting_kwargs = self.default_casting_kwargs.copy()

        kwargs["dtype"] = self._dtypes
        super().construct(data=data, require=require, **kwargs)

    def set_types(self, types: tuple[tuple[str, type]] | None = None):
        self._types = types
        self._types_dict.clear()
        self._types_dict.update({name: i for i, (name, _) in enumerate(types)})
        self._dtypes = tuple((name, self.caster.map_type(type_)) for name, type_ in self.default_types)

    def item_to_dict(self, item: Any) -> dict:
        types = zip(self._types, self.casting_kwargs)
        return {name: self.caster.cast_to(type_, item[i], **kwargs) for i, ((name, type_), kwargs) in enumerate(types)}

    def dict_to_item(self, dict_: dict) -> Any:
        return [self.caster.cast_from(dict_[name]) for i, (name, _) in enumerate(self._types)]

    def get_item_dict(self, index: int | tuple) -> dict:
        return self.item_to_dict(self[index])

    def set_item_dict(self, index: int | tuple, dict_: dict) -> None:
        self[index] = self.dict_to_item(dict_)

