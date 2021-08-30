#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hierarchialdataset.py
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
import collections

# Downloaded Libraries #
from baseobjects import BaseObject, DynamicWrapper
from bidict import bidict
import numpy as np

# Local Libraries #
from .linkeddataset import HDF5LinkedDatasets


# Definitions #
# Functions #
def merge_dict(dict1, dict2, copy_=True):
    if dict2 is not None:
        if copy_:
            dict1 = dict1.copy()
        dict1.update(dict2)
    return dict1


def recursive_looping(loops, func, previous=None, size=None, **kwargs):
    if previous is None:
        previous = []
    if size is None:
        size = len(loops)
    output = []
    loop = loops.pop(0)
    previous.append(None)
    for item in loop:
        previous[-1] = item
        if len(previous) == size:
            output.append(func(items=previous, **kwargs))
        else:
            output.append(recursive_looping(loops, func, previous=previous, size=size, **kwargs))
    return output


# Classes #
class HDF5hierarchicalDatasets(object):
    # Instantiation, Copy, Destruction
    def __init__(self, h5_container=None, dataset=None, name="", child_name="", link_name="", init=True, **kwargs):
        self.h5_container = None

        self.parent_name = None
        self.parent_dtype = None
        self.parent_dataset = None
        self.parent_link_name = None
        self.child_datasets = bidict()
        self.dataset_links = HDF5LinkedDatasets()

        self.child_name_field = None

        if init:
            self.construct(h5_container, dataset, name, child_name, link_name, **kwargs)

    # Container Magic Methods
    def __getitem__(self, item):
        return self.get_item(item)

    # Generic Methods #
    # Constructors Methods
    def construct(self, h5_container, dataset=None, name="", child_name="", link_name="", **kwargs):
        self.h5_container = h5_container
        self.child_name_field = child_name
        if dataset:
            self.set_parent_dataset(name=name, dataset=dataset, link_name=link_name)
        else:
            self.create_parent_dataset(name=name, link_name=link_name, **kwargs)

    # Dataset Getter and Setters
    def create_parent_dataset(self, name, link_name, **kwargs):
        if self.parent_name is not None:
            self.dataset_links.pop_dataset(self.parent_name)
        self.parent_name = name
        self.parent_dataset = self.h5_container.set_dataset(name=name, **kwargs)
        self.parent_dtype = self.parent_dataset.dtype.descr
        self.parent_link_name = link_name
        self.dataset_links.add_datset(name, self.parent_dataset, link_name)

    def set_parent_dataset(self, name, dataset, link_name):
        if self.parent_name is not None:
            self.dataset_links.pop_dataset(self.parent_name)
        self.parent_name = name
        self.parent_dataset = dataset
        self.parent_dtype = self.parent_dataset.dtype.descr
        self.parent_link_name = link_name
        self.dataset_links.add_datset(name, self.parent_dataset, link_name)
        self.load_parent_dataset()

    def remove_parent_dataset(self):
        self.dataset_links.pop_dataset(self.parent_name)
        self.parent_name = None
        self.parent_dtype = None
        self.parent_link_name = None

    def create_child_dataset(self, name, link_name=None, **kwargs):
        if link_name is None:
            link_name = self.parent_link_name
        dataset = self.h5_container.set_dataset(name=name, **kwargs)
        self.dataset_links.add_datset(name, dataset, link_name)
        self.child_datasets[name] = dataset

    def add_child_dataset(self, name, dataset, link_name=None):
        if link_name is None:
            link_name = self.parent_link_name
        self.dataset_links.add_datset(name, dataset, link_name)
        self.child_datasets[name] = dataset

    def clear_child_datasets(self):
        for child in self.child_datasets:
            self.dataset_links.pop_dataset(child)
        self.child_datasets.clear()

    def remove_child_dataset(self, name):
        self.dataset_links.pop_dataset(name)
        self.child_datasets.pop(name)

    # Parent Dataset
    def load_parent_dataset(self):
        self.clear_child_datasets()

        array = self.parent_dataset[self.child_name_field]

        if array.size > 0:
            for child in array.flatten().tolist():
                if child not in self.child_datasets:
                    self.add_child_dataset(child, self.h5_container[child])

    # Item Getter and Setters
    def get_dataset(self, name, id_info=False):
        data = self.parent_dataset
        child_names = data[self.child_name_field]
        if name != self.parent_name:
            data = data[child_names == name]
            child_names = data[self.child_name_field]
        link_ids = data[self.parent_link_name]
        if isinstance(data, np.void):
            return self.get_data(link_ids, child_names, id_info)
        else:
            shape = data.shape

            loops = [range(0, x) for x in shape]

            return recursive_looping(loops, self.arrays_to_items, names=child_names, ids=link_ids, id_info=id_info)

    def get_item(self, index, name=None, id_info=False):
        if name is None or name == self.parent_name:
            parent_index = index
            child_name = self.parent_dataset[parent_index][self.child_name_field]
        else:
            parent_index = self.dataset_links.get_linked_indices(name, index, self.parent_name)[self.parent_name]
            child_name = name
        return self.get_data(parent_index, child_name, id_info)

    def get_items(self, indices, id_info=False):
        data = self.parent_dataset[indices]
        child_names = data[self.child_name_field]
        link_ids = data[self.parent_link_name]
        if isinstance(data, np.void):
            return self.get_data(link_ids, child_names, id_info)
        else:
            shape = data.shape

            loops = [range(0, x) for x in shape]

            return recursive_looping(loops, self.arrays_to_items, names=child_names, ids=link_ids, id_info=id_info)

    def arrays_to_items(self, items, names, ids, id_info=False):
        child_name = names[items]
        if isinstance(child_name, np.ndarray):
            child_name = child_name.tolist()[0]
        parent_id = ids[items]
        if isinstance(parent_id, np.ndarray):
            parent_id = parent_id.tolist()[0]
        return self.get_data(parent_id, child_name, id_info)

    def get_data(self, parent_ref, child_name, id_info=False):
        parent_link = self.dataset_links.references[self.parent_name].reference_field
        child_link = self.dataset_links.references[child_name].reference_field

        data = self.dataset_links.get_linked_data(self.parent_name, parent_ref, child_name)

        parent = data[self.parent_name]
        child = data[child_name]
        result = merge_dict(parent, child)

        if not id_info:
            if parent_link in result:
                result.pop(parent_link)
            if child_link in result:
                result.pop(child_link)

        return result

    def add_item(self, item, index):
        parent_item = []
        for dtype in self.parent_dtype:
            field = dtype[0]
            parent_item.append(item.pop(field))

        self.parent_dataset[index] = tuple(parent_item)

    def append_item(self, item, children=None, axis=0):
        if isinstance(children, collections.abc.Mapping):
            children_names = children.keys()
            for child, kwargs in children.items():
                if child not in self.child_datasets:
                    self.create_child_dataset(child, **kwargs)
        elif isinstance(children, str):
            children_names = (children,)
        else:
            children_names = children

        self.dataset_links.append_linked_data(self.parent_name, item, children_names, axis=axis)
