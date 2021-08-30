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
import cProfile
import io
import pstats
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
    studies_path = pathlib.Path("/common/subjects")
    mount_path = pathlib.Path("/mnt/changserver/data_store0/human/converted_clinical")
    load_path = pathlib.Path("/common/subjects/EC228/EC228_2020-09-21/EC228_2020-09-21_14~53~19.h5")
    save_path = pathlib.Path("~/Documents/Projects/Epilepsy Spike Detection")

    def test_load_study(self):
        s_id = "EC228"
        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        assert 1

    def test_load_study_profile(self):
        s_id = "EC228"
        pr = cProfile.Profile()
        pr.enable()

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        assert 1

    def test_load_study_mount(self):
        s_id = "EC228"
        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.mount_path)
        assert 1

    def test_get_data(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        data = study_frame[slice(0, 1)]

        assert data is not None

    def test_get_study_time(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        dt = study_frame.get_time(-1)

        assert dt == study_frame.end

    def test_get_time_range(self):
        s_id = "EC228"
        first = datetime.datetime(2020, 9, 22, 0, 00, 00)
        second = datetime.datetime(2020, 9, 22, 0, 10, 00)

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        data = study_frame.get_time_range(first, second, aprox=True)

        assert data is not None

    def test_date_time_range_mount(self):
        s_id = "EC228"
        first = datetime.datetime(2020, 9, 22, 0, 00, 00)
        second = datetime.datetime(2020, 9, 22, 1, 00, 00)
        pr = cProfile.Profile()
        pr.enable()

        with XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path) as study_frame:
            data, true_start, true_end = study_frame.get_time_range(first, second, aprox=True)
            print(data.shape)

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        assert data is not None

    def test_data_range_time(self):
        s_id = "EC228"
        first = datetime.datetime(2020, 9, 22, 0, 00, 00)
        second = datetime.datetime(2020, 9, 22, 0, 10, 00)

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        data, start, stop = study_frame.data_range_time(first, second, aprox=True)

        assert data is not None

    def test_date_rang_time_mount(self):
        s_id = "EC228"
        first = datetime.datetime(2020, 9, 22, 0, 00, 00)
        second = datetime.datetime(2020, 9, 22, 0, 20, 00)
        pr = cProfile.Profile()
        pr.enable()

        with XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path) as study_frame:
            data, true_start, true_end = study_frame.data_range_time(first, second, aprox=True)
            print(data.shape)

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_validate_shape(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        valid = study_frame.validate_shape()

        assert valid

    def test_shapes(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        shapes = study_frame.shapes

        assert shapes is not None

    def test_shape(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        shape = study_frame.shape

        assert shape is not None

    def test_validate_sample_rate(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        valid = study_frame.validate_sample_rate()

        assert valid

    def test_sample_rates(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        sample_rates = study_frame.sample_rates

        assert sample_rates

    def test_sample_rate(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        sample_rate = study_frame.sample_rate

        assert sample_rate

    def test_where_discontinuous(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        discontinuities = study_frame.where_discontinuous()

        assert discontinuities is not None

    def test_validate_continuous(self):
        s_id = "EC228"

        study_frame = XLTEKStudyFrame(s_id=s_id, studies_path=self.studies_path)
        valid = study_frame.validate_continuous()

        assert isinstance(valid, bool)


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])

