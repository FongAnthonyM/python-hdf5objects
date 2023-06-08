"""__init__.py
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
from .axiscomponent import AxisComponent, AxisMap
from .channelaxiscomponent import ChannelAxisComponent, ChannelAxisMap
from .sampleaxiscomponent import SampleAxisComponent, SampleAxisMap
from .timeaxiscomponent import TimeAxisComponent, TimeAxisMap
from .idaxiscomponent import IDAxisComponent, IDAxisMap
from .regionreferenceaxiscomponent import (
    RegionReferenceAxisComponent,
    RegionReferenceAxisMap,
)
