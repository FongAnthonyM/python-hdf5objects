#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_hdf5objects.py
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


class TestHDF5XLTEK(ClassTest):
    class_ = HDF5XLTEK
    studies_path = pathlib.Path("/Users/changlab/Documents/Projects/Epilepsy Spike Detection")
    load_path = pathlib.Path("/Users/changlab/PycharmProjects/python-hdf5objects/tests/EC228_2020-09-21_14~53~19.h5")
    save_path = pathlib.Path("/Users/changlab/PycharmProjects/python-hdf5objects/tests/")

    @pytest.fixture
    def load_file(self):
        return HDF5XLTEK(file=self.load_path)

    def test_load_file(self):
        f_obj = HDF5XLTEK(file=self.load_path)
        assert 1

    def test_validate_file(self):
        assert HDF5XLTEK.validate_file_type(self.load_path)

    @pytest.mark.xfail
    def test_data_speed(self, load_file):
        def assignment():
            x = 10

        def get_data():
            x = load_file.eeg_data[:10000, :100]

        mean_new = timeit.timeit(get_data, number=self.timeit_runs) / self.timeit_runs * 1000000
        mean_old = timeit.timeit(assignment, number=self.timeit_runs) / self.timeit_runs * 1000000
        percent = (mean_new / mean_old) * 100

        print(f"\nNew speed {mean_new:.3f} μs took {percent:.3f}% of the time of the old function.")
        assert percent < self.speed_tolerance

    def test_create_file(self):
        start = datetime.datetime.now()
        f_obj = HDF5XLTEK(s_id="EC_test", s_dir=self.save_path, start=start)
        f_obj.create_eeg_dataset()
        assert 1


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])

