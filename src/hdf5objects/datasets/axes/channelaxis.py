#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" channelaxis.py
Description:
"""
# Package Header #
from ...__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #

# Third-Party Packages #

# Local Packages #
from .axis import AxisMap, Axis


# Definitions #
# Classes #
class ChannelAxisMap(AxisMap):
    ...


class ChannelAxis(Axis):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Magic Methods #
    # Construction/Destruction
    def __init__(self, start: int = None, stop: int = None, step: int = None, rate: float = None, size: int = None,
                 s_name: str = None, build: bool = None, init: bool = True, **kwargs):
        super().__init__(init=False)
        self.default_kwargs = {"dtype": 'i', "maxshape": (None,)}
        self._scale_name = "channel axis"

        if init:
            self.construct(start=start, stop=stop, step=step, rate=rate, size=size,
                           s_name=s_name, build=build, **kwargs)

    @property
    def channels(self):
        try:
            return self.get_all_data.caching_call()
        except AttributeError:
            return self.get_all_data()

    def get_channels(self):
        return self.get_all_data()


# Assign Cyclic Definitions
ChannelAxisMap.default_type = ChannelAxis
ChannelAxis.default_map = ChannelAxisMap()
