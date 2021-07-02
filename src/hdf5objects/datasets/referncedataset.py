#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" referncedataset.py
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
import uuid
from warnings import warn

# Downloaded Libraries #
from baseobjects import BaseObject, DynamicWrapper
from bidict import bidict
import numpy as np

# Local Libraries #


# Definitions #
# Functions #
def np_to_dict(array):
    result = {}
    for t in array.dtype.descr:
        name = t[0]
        result[name] = array[name]
    return result


def item_to_np(item):
    if isinstance(item, int) or isinstance(item, float) or isinstance(item, str):
        return item
    elif isinstance(item, datetime.datetime):
        return item.timestamp()
    elif isinstance(item, datetime.timedelta):
        return item.total_seconds()
    elif isinstance(item, uuid.UUID):
        return str(item)
    else:
        return item


def dict_to_np(item, descr, pop=False):
    array = []
    for dtype in descr:
        field = dtype[0]
        if pop:
            data = item.pop(field)
        else:
            data = item[field]
        data = item_to_np(data)
        array.append(data)
    return array


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
class HDF5ReferenceDataset(object):
    # Instantiation, Copy, Destruction
    def __init__(self, dataset, reference_field="", dtype=None, init=True):
        self.dataset = None

        self.reference_field = ""
        self.dtype = None
        self.fields = bidict()
        self.references = bidict()

        self._reference_array = None

        if init:
            self.construct(dataset, reference_field, dtype)

    def __copy__(self):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        return new

    @property
    def reference_array(self):
        try:
            self._reference_array = self.dataset[self.reference_field]
        except Exception as e:
            warn(e)
        finally:
            return self._reference_array

    # Container Magic Methods
    def __getitem__(self, items):
        return self.get_item(items)

    def __setitem__(self, key, value):
        self.set_item(value, key)

    def __contains__(self, item):
        if isinstance(item, str):
            item = uuid.UUID(item)
        if isinstance(item, uuid.UUID):
            result = self.find_index(item)
        elif isinstance(item, tuple):
            result = self.find_id(item)
        else:
            raise KeyError

        if result is None:
            return False
        else:
            return True

    # Generic Methods #
    # Constructors Methods
    def construct(self, dataset, reference_field, dtype=None):
        self.dataset = dataset

        if dtype is None:
            self.dtype = self.dataset.dtype
        else:
            self.dtype = dtype

        for i, field in enumerate(self.dtype.descr):
            self.fields[field[0]] = i

        if reference_field in self.fields:
            self.reference_field = reference_field
        else:
            raise NameError

        self._reference_array = self.dataset[self.reference_field]

    # Copy Methods
    def copy(self):
        return self.__copy__()

    # Reference Getters and Setters
    def new_reference(self, index=None, axis=0, id_=None):
        shape = self.dataset.shape
        if index is None:
            shape = list(shape)
            shape[axis] += 1
            new_shape = shape[axis]
            index = [a - 1 for a in shape]
        elif len(index) == len(shape):
            new_shape = []
            axis = None
            for s, i in zip(shape, index):
                new_shape.append(max((s, i)))
        else:
            raise IndexError

        with self.dataset:
            self.dataset.resize(new_shape, axis)

        if id_ is None:
            id_ = uuid.uuid4()
        self.set_reference(index, id_)
        return id_

    def get_references(self, items):
        if not isinstance(items, tuple):
            items = (items,)

        result = []
        for item in items:
            if isinstance(item, str):
                item = uuid.UUID(item)
            if isinstance(item, uuid.UUID):
                result.append(self.get_index(item))
            elif isinstance(item, tuple) or isinstance(item, int):
                result.append(self.get_id(item))

        if len(result) > 1:
            result = tuple(result)
        else:
            result = result[0]
        return result

    def set_reference(self, index, id_):
        index = tuple(index)
        if isinstance(id_, str):
            id_ = uuid.UUID(id_)
        item = self.dataset[index]
        item[self.reference_field] = str(id_)
        self.dataset[index] = item
        self.references[id_] = index

    def get_index(self, id_):
        try:
            index = self.find_index(id_)
            if index:
                self.references[id_] = index
            else:
                raise KeyError
        except Exception as e:
            warn(e)
        finally:
            return self.references[id_]

    def find_index(self, id_):
        indices = np.where(self.reference_array == str(id_))
        if len(indices) == 0:
            index = None
        elif len(indices[0]) > 1:
            raise KeyError
        else:
            temp = []
            for axis in indices:
                if axis.size > 0:
                    temp.append(axis[0])
                else:
                    return None
            index = tuple(temp)
        return index

    def get_id(self, index):
        if isinstance(index, int):
            shape = self.dataset.shape
            if index < 0:
                index = shape[0] - 1
            index = tuple([index] + [0]*(len(shape)-1))
        try:
            self.references.inverse[index] = self.find_id(index)
        except Exception as e:
            warn(e)
        finally:
            return self.references.inverse[index]

    def find_id(self, index):
        try:
            array = self.reference_array[index]
            result = uuid.UUID(self.reference_array[index])
        except ValueError:
            result = None
        finally:
            return result

    # Item Getters and Setters
    def get_item(self, location, dict_=True, id_=True):
        # Polymorphic Get Item as a Dictionary
        if isinstance(location, str):
            location = uuid.UUID(location)
            index = self.get_index(location)
        elif isinstance(location, uuid.UUID):
            index = self.get_index(location)
        elif isinstance(location, tuple):
            index = location

        result = self.dataset[index]
        if dict_:
            result = np_to_dict(result)
            if not id_:
                result.pop(self.reference_field)

        return result

    def set_item(self, item, location, pop=False):
        # Polymorphic Assignment
        if isinstance(location, uuid.UUID):
            id_ = location
            index = self.get_index(location)
        else:
            id_ = self.get_id(location)
            if id_ is None:
                id_ = uuid.uuid4()
            index = location

        # Add/Assign Reference ID to Item
        item[self.reference_field] = id_

        # Build Array from Item
        array = dict_to_np(item, self.dtype.descr, pop=pop)

        # Assign Array to Dataset
        self.dataset[index] = tuple(array)
        self.references[id_] = index


