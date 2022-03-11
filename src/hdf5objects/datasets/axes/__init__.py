""" __init__.py
Datasets that are designed to be Axes.
"""
# Package Header #
from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Local Packages #
from .axis import Axis, AxisMap
from .channelaxis import ChannelAxis, ChannelAxisMap
from .sampleaxis import SampleAxis, SampleAxisMap
from .timeaxis import TimeAxis, TimeAxisMap
