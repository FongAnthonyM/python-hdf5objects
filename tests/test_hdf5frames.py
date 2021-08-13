#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_hdf5frames.py
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
import timeit

# Downloaded Libraries #
import pytest
import numpy as np

# Local Libraries #
from src.hdf5objects import *
from src.hdf5objects.dataframes import *


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
    timeit_runs = 2
    speed_tolerance = 200

    def get_log_lines(self, tmp_dir, logger_name):
        path = tmp_dir.joinpath(f"{logger_name}.log")
        with path.open() as f_object:
            lines = f_object.readlines()
        return lines


class TestXLTEKStudy(ClassTest):
    class_ = XLTEKStudyFrame
    studies_path = pathlib.Path("/Users/changlab/Documents/Projects/Epilepsy Spike Detection")
    load_path = pathlib.Path("/Users/changlab/PycharmProjects/python-hdf5objects/tests/EC228_2020-09-21_14~53~19.h5")
    save_path = pathlib.Path("/Users/changlab/PycharmProjects/python-hdf5objects/tests/")

    def test_load_study(self):
        s_id = "EC228"
        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        assert 1

    def test_get_data(self):
        s_id = "EC228"
        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        data = study_frame[slice(0, 1)]
        assert data is not None

    def test_get_study_time_range(self):
        s_id = "EC228"
        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)

        assert 1


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])

