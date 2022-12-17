#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_regionreferencedataset.py
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
from collections.abc import Mapping
import datetime
import pathlib
import timeit
import uuid

# Third-Party Packages #
from classversioning import Version, TriNumberVersion
import h5py
from hdf5objects import HDF5Map
import pytest
import numpy as np

# Local Packages #
from src.hdf5objects import BaseHDF5, BaseHDF5Map, HDF5Dataset, DatasetMap
from src.hdf5objects.datasets import RegionReferenceDataset, RegionReferenceMap
from .test_dataset import DatasetTestMap


# Definitions #
# Classes #
# Module Implementation
class RegionReferenceTestMap(RegionReferenceMap):
    """Implementation of RegionReferenceMap."""
    default_attribute_names = RegionReferenceMap.default_attribute_names | {"test_attribute": "TestAttribute"}
    # default_attributes = RegionReferenceMap.default_attributes | {"test_attribute": h5py.Reference()}
    default_dtype = (
        ("ID", uuid.UUID),
        ("Text", str),
        ("multiple_object", h5py.ref_dtype),
        ("multiple_region", h5py.regionref_dtype),
        ("single_region", h5py.regionref_dtype),
    )
    default_single_reference_fields = {"test_single": ("test_attribute", "single_region")}
    default_multiple_reference_fields = {"test_multiple": ("multiple_object", "multiple_region")}
    default_primary_reference_field = "test_single"


class RegionReferenceDatasetTestFileMap(BaseHDF5Map):
    """The map for the file which implements RegionReferenceDataset."""
    default_map_names: Mapping[str, str] = {"test_dataset": "test_dataset"}
    default_maps: Mapping[str, HDF5Map] = {
        "main_dataset": RegionReferenceTestMap(shape=(1,), maxshape=(None,)),
        "secondary_dataset": DatasetMap(shape=(0, 0), maxshape=(None, None), dtype='f8'),
        "tertiary_dataset": DatasetTestMap(shape=(1,), maxshape=(None,)),
    }


class RegionReferenceDatasetTestHDF5(BaseHDF5):
    """The file object that implements the RegionReferenceDataset."""
    _registration: bool = True
    FILE_TYPE: str = "TestRegionReference"
    VERSION: Version = TriNumberVersion(0, 0, 0)
    default_map: HDF5Map = RegionReferenceDatasetTestFileMap()


# Module Test
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


class TestDataset(ClassTest):
    @pytest.fixture
    def load_file(self, tmp_path):
        return DatasetMap(file=tmp_path)

    def test_set_item_dict(self, tmp_path):
        test_data = np.random.rand(100, 100)
        test_tertiary_entry = {"First": 2.0, "Second": 3, "Third": "Random", "Fourth": uuid.uuid4()}
        test_ref_entry = {
            "ID": uuid.uuid4(),
            "Text": "Something",
            "multiple_object": h5py.Reference(),
            "multiple_region": h5py.RegionReference(),
            "single_region": h5py.RegionReference(),
        }
        with RegionReferenceDatasetTestHDF5(file=tmp_path / "test.h5", mode='a', create=True, require=True) as test_file:
            test_dataset = test_file["main_dataset"]
            normal_dataset = test_file["secondary_dataset"]
            multi_dataset = test_file["tertiary_dataset"]

            normal_dataset.set_data(test_data)
            multi_dataset.set_item_dict(0, test_tertiary_entry)

            test_dataset.attributes["test_attribute"] = normal_dataset.ref
            obj_1, region_1 = test_dataset.generate_region_reference(region=(slice(20, 30), slice(20, 30)))
            obj_2, region_2 = test_dataset.generate_region_reference(region=0, object_=multi_dataset, ref_name="test_multiple")

            test_ref_entry["single_region"] = region_1
            #test_ref_entry.update({"multiple_region": region_2, "multiple_object": obj_2})

            test_dataset.set_item_dict(0, test_ref_entry)
            grabbed_data = test_dataset.get_from_reference(0)

            test_ref_entry.update({"multiple_region": region_2, "multiple_object": obj_2})
            test_dataset.append_item_dict(test_ref_entry)
            grabbed_entry = test_dataset.get_from_reference_dict(1, ref_name="test_multiple")

        assert grabbed_data.shape == (10, 10)
        assert tuple(grabbed_entry) == tuple(test_tertiary_entry)

    def test_append_item_dict(self, tmp_path):

        test_data = {"First": 2.0, "Second": 3, "Third": "Random", "Fourth": uuid.uuid4()}
        with RegionReferenceDatasetTestHDF5(file=tmp_path / "test.h5", mode='a', create=True, require=True) as test_file:
            multi_dataset = test_file["test_dataset"]
            multi_dataset.append_item_dict(test_data)
            new_dict = multi_dataset.get_item_dict(-1)
            n_items = multi_dataset.shape[0]

        assert n_items == 2
        assert tuple(test_data) == tuple(new_dict)


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])

