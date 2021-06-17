#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_xltekobjects.py
Description:
"""
__author__ = "Anthony Fong"
__copyright__ = "Copyright 2021, Anthony Fong"
__credits__ = ["Anthony Fong"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Anthony Fong"
__email__ = ""
__status__ = "Prototype"

# Default Libraries #
import datetime
import pathlib

# Downloaded Libraries #
import pytest

# Local Libraries #
from src.hdf5objects.xltek import *
from src.hdf5objects.xltek.ui import EEGScanner


# Definitions #
# Functions #
@pytest.fixture
def tmp_dir(tmpdir):
    """A pytest fixture that turn the tmpdir into a Path object."""
    return pathlib.Path(tmpdir)


# Classes #
class ClassTest:
    """Default class tests that all classes should pass."""
    class_ = None
    timeit_runs = 100
    speed_tolerance = 200

    def get_log_lines(self, tmp_dir, logger_name):
        path = tmp_dir.joinpath(f"{logger_name}.log")
        with path.open() as f_object:
            lines = f_object.readlines()
        return lines


class TestHDF5XLTEKstudy(ClassTest):
    class_ = HDF5XLTEKstudy
    studies_path = pathlib.Path("/Users/changlab/Documents/Projects/Epilepsy Spike Detection")

    def test_open_study(self):
        view_channels = list(range(1, 5))
        first = datetime.datetime(2020, 9, 21, 15, 00, 00)
        second = datetime.datetime(2020, 9, 22, 15, 00, 00)

        study = HDF5XLTEKstudy("EC228", spath=self.studies_path)

        d, f, g = study.data_range_time(first, second, frame=True)

        d[0:1, 0:2]

        all_viewer = EEGScanner(d[0], view_channels, ylim=2000, show=True)

        assert 1


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])
