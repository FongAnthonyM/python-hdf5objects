"""__init__.py
Datasets that are designed to be Axes.
"""
# Header #
__package_name__ = "hdf5objects"

__author__ = "Anthony Fong"
__credits__ = ["Anthony Fong"]
__copyright__ = "Copyright 2021, Anthony Fong"
__license__ = "MIT"

__version__ = "0.6.0"
__maintainer__ = "Anthony Fong"
__email__ = """


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
from .labelaxiscomponent import LabelAxisComponent, LabelAxisMap
from .coordinateaxiscomponent import CoordinateAxisComponent, CoordinateAxisMap
