#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" linkeddataset.py
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
import uuid

# Downloaded Libraries #
from baseobjects import BaseObject, DynamicWrapper

# Local Libraries #
from .referncedataset import HDF5ReferenceDataset


# Definitions #
# Classes #
class HDF5LinkedDatasets(object):
    # Instantiation, Copy, Destruction
    def __init__(self):
        self.references = {}

    def __copy__(self):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        return new

    # Container Magic Methods
    def __getitem__(self, items):
        if not isinstance(items, tuple):
            items = (items,)

        result = []
        for item in items:
            if isinstance(item, uuid.UUID):
                result.append(self.get_indices(item))
            elif isinstance(item, str):
                result.append(self.references[item])

        if len(result) > 1:
            result = tuple(result)
        else:
            result = result[0]
        return result

    # Copy Methods
    def copy(self):
        return self.__copy__()

    # Dataset Getters and Setters
    def add_datset(self, name, dataset, link_name):
        self.references[name] = HDF5ReferenceDataset(dataset, link_name)

    def pop_dataset(self, name):
        return self.references.pop(name)

    # Links and Indices
    def new_link(self, locations=None, axis=0, id_=None):
        if id_ is None:
            id_ = uuid.uuid4()

        if not locations:
            for references in self.references.values():
                references.new_reference(axis=axis, id_=id_)
        elif isinstance(locations, collections.abc.Mapping):
            for name, index in locations.items():
                self.references[name].new_reference(index=index, id_=id_)
        else:
            for name in locations:
                self.references[name].new_reference(axis=axis, id_=id_)

        return id_

    def get_index(self, name, id_):
        return self.references[name].get_references(id_)

    def get_indices(self, id_, datasets=None):
        result = {}
        if not datasets:
            for name, references in self.references.items():
                if id_ in references:
                    result[name] = references.get_references(id_)
        else:
            if isinstance(datasets, str):
                datasets = (datasets,)
            for name in datasets:
                result[name] = self.references[name].get_references(id_)

        return result

    def get_linked_indices(self, name, index, datasets=None):
        id_ = self.get_id(name, index)
        return self.get_indices(id_, datasets)

    def get_id(self, name, index):
        return self.references[name].get_references((index,))

    # Item Getter and Setters
    def get_linked_data(self, name, location, children=None, dict_=True):
        if isinstance(location, str):
            location = uuid.UUID(location)
        if isinstance(location, uuid.UUID):
            child_indices = self.get_indices(location)
        else:
            child_indices = self.get_linked_indices(name, location)

        valid_children = {}
        if children:
            if isinstance(children, str):
                children = [children]
            for child in children:
                valid_children[child] = child_indices[child]
        else:
            valid_children = child_indices
        if name in valid_children:
            valid_children.pop(name)

        if dict_:
            result = {name: self.references[name][location]}
            for name, child_index in valid_children.items():
                result[name] = self.references[name][child_index]
        else:
            result = [self.references[name].get_item(location, dict_=False)]
            for name, child_index in valid_children.items():
                result.append(self.references[name].get_item(child_index, dict_=False))
            result = tuple(result)

        return result

    def set_linked_data(self, name, location, item, children=None):
        if isinstance(location, str):
            location = uuid.UUID(location)
        if isinstance(location, uuid.UUID):
            id_ = location
            main_index = self.get_index(name, location)
        else:
            id_ = self.get_id(name, location)
            main_index = location

        child_indices = self.get_indices(id_, children)
        valid_children = {}
        if children:
            for child in children:
                valid_children[child] = child_indices[child]
        else:
            valid_children = child_indices
        if name in valid_children:
            valid_children.pop(name)

        data = item.copy()
        self.references[name].set_item(data, main_index, pop=True)
        for child_name, child_index in valid_children.items():
            self.references[child_name].set_item(data, child_index, pop=True)

        return data

    def append_linked_data(self, name, item, children=None, axis=0):
        datasets = {name} | set(children)
        location = self.new_link(datasets, axis=axis)
        self.set_linked_data(name, location, item, children)

