
""" test_hdf5objects.py
Description:
"""
# Package Header #
from src.hdf5objects.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import datetime
import pathlib
import timeit

# Third-Party Packages #
import pytest
import numpy as np

# Local Packages #
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


class TestHDF5File(ClassTest):
    class_ = HDF5File
    studies_path = pathlib.Path("/common/subjects")
    load_path = pathlib.Path.cwd().joinpath("pytest_cache/EC228_2020-09-21_14~53~19.h5")
    save_path = pathlib.Path.cwd().joinpath("pytest_cache/")

    @pytest.fixture
    def load_file(self):
        return self.class_(file=self.load_path)

    @pytest.mark.parametrize("mode", ['r', 'r+', 'a'])
    def test_new_object(self, mode):
        with self.class_(file=self.load_path, mode=mode) as f_obj:
            assert f_obj is not None
        assert True

    @pytest.mark.parametrize("mode", ['r', 'r+', 'a'])
    def test_load_whole_file(self, mode):
        with self.class_(file=self.load_path, mode=mode, load=True) as f_obj:
            assert f_obj is not None
        assert True

    def test_load_fragment(self):
        f_obj = self.class_(file=self.load_path)
        data = f_obj["data"]
        f_obj.close()
        assert data is not None

    def test_load_from_property(self):
        f_obj = self.class_(file=self.load_path)
        data = f_obj.eeg_data
        f_obj.close()
        assert data is not None

    def test_get_attribute(self):
        f_obj = self.class_(file=self.load_path)
        attribute = f_obj.attributes["start"]
        f_obj.close()
        assert attribute is not None

    def test_get_attribute_property(self):
        f_obj = self.class_(file=self.load_path)
        attribute = f_obj.start
        f_obj.close()
        assert attribute is not None

    def test_get_data(self):
        f_obj = self.class_(file=self.load_path)
        data = f_obj.eeg_data[0:1]
        f_obj.close()
        assert data.shape is not None

    def test_get_times(self):
        f_obj = self.class_(file=self.load_path)
        start = f_obj.time_axis.start_datetime
        f_obj.close()
        assert start is not None

    def test_validate_file(self):
        assert self.class_.validate_file_type(self.load_path)

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
        f_obj = self.class_(s_id="EC_test", s_dir=self.save_path, start=start)
        f_obj.create_eeg_dataset()
        assert 1


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])

