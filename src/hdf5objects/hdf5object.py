#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" hdf5object.py
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
import copy
import datetime
import pathlib
import time
import uuid
from warnings import warn

# Downloaded Libraries #
from baseobjects import BaseObject, DynamicWrapper, StaticWrapper
from bidict import bidict
from classversioning import VersionedClass, VersionType, TriNumberVersion
import h5py
import numpy as np

# Local Libraries #


# Definitions #
# Functions #
def merge_dict(dict1, dict2, copy_=True):
    if dict2 is not None:
        if copy_:
            dict1 = dict1.copy()
        dict1.update(dict2)
    return dict1


def np_to_dict(array):
    result = {}
    for t in array.dtype.descr:
        name = t[0]
        result[name] = array[name]
    return result


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
class HDF5Object(BaseObject):
    # Instantiation, Copy, Destruction
    def __init__(self, path=None, update=True, init=False):
        self._file_attrs = set()
        self._datasets = set()
        self._path = None
        self.is_updating = True

        self.cargs = {"compression": "gzip", "compression_opts": 4}
        self.default_datasets_parameters = self.cargs.copy()
        self.default_file_attributes = {}
        self.default_datasets = {}

        self.h5_fobj = None

        if init:
            self.construct(path, update)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    @property
    def is_open(self):
        return bool(self.h5_fobj)

    @property
    def file_attribute_names(self):
        if not self._file_attrs or self.is_updating:
            return self.get_file_attribute_names()
        else:
            return self._file_attrs

    @property
    def dataset_names(self):
        if not self._datasets or self.is_updating:
            return self.get_dataset_names()
        else:
            return self._datasets

    def __deepcopy__(self, memo={}):
        # Todo: Rethink how a deep copy should be made
        new = type(self)(path=self.path.as_posix)
        new._file_attrs = copy.deepcopy(self._file_attrs, memo=memo)
        new._datasets = copy.deepcopy(self._datasets, memo=memo)
        if self.is_open:
            new.open()
        new.is_updating = self.is_updating
        new.cargs = copy.deepcopy(self.cargs, memo=memo)
        new.default_datasets_parameters = copy.deepcopy(self.cargs, memo=memo)
        return new

    def __del__(self):
        self.close()

    # Pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        name = state["path"].as_posix()
        open_state = state["is_open"]
        if self.is_open:
            fobj = state["h5_fobj"]
            fobj.flush()
            fobj.close()
        state["h5_fobj"] = (name, open_state)
        return state

    def __setstate__(self, state):
        name, open_state = state["h5_fobj"]
        state["h5_fobj"] = h5py.File(name, "r+")
        if not open_state:
            state["h5_fobj"].close()
        self.__dict__.update(state)

    # Attribute Access
    def __getattr__(self, name):
        """Overrides the getattr magic method to get an attribute or dataset from the h5 file.

        Args:
            name (str): The name of the attribute to get.

        Returns:
            obj: Whatever the attribute contains.
        """
        if name in self.file_attribute_names:
            return self.get_file_attribute(name)
        elif name in self.dataset_names:
            return self.get_dataset(name)
        else:
            raise AttributeError(f"{type(self)} does not have \"{name}\" as an attribute")

    def __setattr__(self, name, value):
        """Overrides the setattr magic method to set the attribute of the h5 file.

        Args:
            name (str): The name of the attribute to set.
            value: Whatever the attribute will contain.
        """
        if name in ("_file_attrs", "_datasets"):
            super().__setattr__(name, value)
        elif name in self.file_attribute_names:
            self.set_file_attribute(name, value)
        elif name in self.dataset_names:
            self.set_dataset(name, value)
        else:
            super().__setattr__(name, value)

    # Container Magic Methods
    def __len__(self):
        op = self.is_open
        self.open()
        length = len(self.h5_fobj)
        if not op:
            self.close()
        return length

    def __getitem__(self, item):
        if item in self.dataset_names:
            data = self.get_dataset(item)
        else:
            raise KeyError(item)
        return data

    def __setitem__(self, key, value):
        self.set_dataset(key, value)

    # Context Managers
    def __enter__(self):
        if self.h5_fobj is None:
            self.construct(open_=True)
        else:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    # Constructors
    def construct(self, path=None, update=None, open_=False, **kwargs):
        if path is not None:
            self.path = path
        if update is not None:
            self.is_updating = update

        if self.path.is_file():
            self.open(validate=True, **kwargs)
            if not open_:
                self.close()
        else:
            self.create_file(open_=open_)

    # File Creation
    def create_file(self, open_=False):
        self.open()
        if self.default_attrs:
            self.add_file_attributes(self.default_attrs)
        if self.default_datasets:
            self.add_file_datasets(self.default_datasets)
        if open_:
            return self.h5_fobj
        else:
            self.close()
            return None

    def construct_file_attributes(self, value=""):
        if len(self.file_attribute_names.intersection(self._datasets)) > 0:
            warn("Attribute name already exists", stacklevel=2)

        op = self.is_open
        self.open()
        for key in self.file_attrs_names:
            _key = "_" + key
            try:
                self.h5_fobj.attrs[key] = value
            except Exception as e:
                warn("Could not set attribute due to error: " + str(e), stacklevel=2)
            else:
                super().__setattr__(_key, value)
        if not op:
            self.close()

    def construct_file_datasets(self, **kwargs):
        if len(self.dataset_names.intersection(self._file_attrs)) > 0:
            warn("Dataset name already exists", stacklevel=2)

        op = self.is_open
        self.open()
        for key in self._datasets:
            _key = "_" + key
            try:
                args = merge_dict(self.default_datasets_parameters, kwargs)
                self.h5_fobj.require_dataset(key, **args)
            except Exception as e:
                warn("Could not set datasets due to error: " + str(e), stacklevel=2)
            else:
                super().__setattr__(_key, type(self)(self.h5_fobj[key], self))
        if not op:
            self.close()

    # Getters and Setters
    def get(self, item):
        if item in self.file_attribute_names:
            return self.get_file_attribute(item)
        elif item in self.dataset_names:
            return self.get_dataset(item)
        else:
            warn("No attribute or dataset " + item, stacklevel=2)
            return None

    # File Attributes
    def get_file_attribute_names(self):
        for key, value in self.h5_fobj.attrs.items():
            self._file_attrs.update((key,))
        return self._file_attrs

    def get_file_attribute(self, item):
        _item = "_" + item
        op = self.is_open
        self.open()

        try:
            if item in self.h5_fobj.attrs and (self._item is None or self.is_updating):
                setattr(self, _item, self.h5_fobj.attrs[item])
        except Exception as e:
            warn("Could not update attribute due to error: "+str(e), stacklevel=2)

        if not op:
            self.close()
        return getattr(self, _item)

    def get_file_attributes(self):
        op = self.is_open
        self.open()

        attr = dict(self.h5_fobj.attrs.items())

        if not op:
            self.close()
        return attr

    def set_file_attribute(self, key, value):
        op = self.is_open
        self.open()
        _item = "_" + key

        try:
            if key not in self._file_attrs:
                self._file_attrs.update((key,))
            self.h5_fobj.attrs[key] = value
        except Exception as e:
            warn("Could not set attribute due to error: " + str(e), stacklevel=2)
        else:
            super().__setattr__(_item, value)

        if not op:
            self.close()

    def add_file_attributes(self, items):
        names = set(items.keys())
        if len(names.intersection(self.dataset_names)) > 0:
            warn("Attribute name already exists", stacklevel=2)

        op = self.is_open
        self.open()
        for key, value in items.items():
            _key = "_" + key
            try:
                if key not in self._file_attrs:
                    self._file_attrs.update((key,))
                self.h5_fobj.attrs[key] = value
            except Exception as e:
                warn("Could not set attribute due to error: " + str(e), stacklevel=2)
            else:
                super().__setattr__(_key, value)
        if not op:
            self.close()

    def clear_file_attributes(self):
        for key in self._file_attrs:
            self.__delattr__("_" + key)
        self._file_attrs.clear()

    def load_file_attributes(self):
        for key, value in self.h5_fobj.attrs.items():
            _item = "_" + key
            self._file_attrs.update((key,))
            super().__setattr__(_item, value)

    def list_file_attributes(self):
        return list(self.file_attrs_names)

    # Datasets
    def get_dataset_names(self):
        for name, value in self.h5_fobj.items():
            self._datasets.update((name,))

    def create_dataset(self, name, data=None, **kwargs):
        self.set_dataset(name=name, data=data, **kwargs)
        return self.get_dataset(name)

    def get_dataset(self, item):
        _item = "_" + item
        op = self.is_open
        self.open()

        try:
            if item in self.h5_fobj and self.is_updating:
                super().__setattr__(_item, HDF5dataset(self.h5_fobj[item], self))
        except Exception as e:
            warn("Could not update datasets due to error: " + str(e), stacklevel=2)

        if not op:
            self.close()
        return super().__getattribute__(_item)

    def set_dataset(self, name, data=None, **kwargs):
        op = self.is_open
        self.open()
        _key = "_" + name

        try:
            if name in self._datasets:
                self.h5_fobj[name][...] = data
            else:
                self._datasets.update((name,))
                args = merge_dict(self.default_datasets_parameters, kwargs)
                args["data"] = data
                self.h5_fobj.require_dataset(name, **args)
        except Exception as e:
            warn("Could not set datasets due to error: " + str(e), stacklevel=2)
        else:
            super().__setattr__(_key, HDF5dataset(self.h5_fobj[name], self))

        if not op:
            self.close()

    def add_file_datasets(self, items):
        names = set(items.keys())
        if len(names.intersection(self._file_attrs)) > 0:
            warn("Dataset name already exists", stacklevel=2)

        op = self.is_open
        self.open()
        for name, kwargs in items.items():
            _key = "_" + name
            try:
                self._datasets.update((name,))
                args = merge_dict(self.default_datasets_parameters, kwargs)
                self.h5_fobj.require_dataset(name, **args)
            except Exception as e:
                warn("Could not set datasets due to error: " + str(e), stacklevel=2)
            else:
                super().__setattr__(_key, HDF5dataset(self.h5_fobj[name], self))
        if not op:
            self.close()

    def clear_datasets(self):
        for name in self._datasets:
            self.__delattr__("_" + name)
        self._datasets.clear()

    def load_datasets(self):
        self.clear_datasets()
        for name, value in self.h5_fobj.items():
            _item = "_" + name
            self._datasets.update((name,))
            super().__setattr__(_item, value)

    def list_dataset_names(self):
        return list(self.datasets_names)

    #  Mapping Items Methods
    def items(self):
        return self.items_file_attributes() + self.items_datasets()

    def items_file_attributes(self):
        result = []
        for key in self.file_attribute_names:
            result.append((key, self.get_file_attribute(key)))
        return result

    def items_datasets(self):
        result = []
        for key in self.dataset_names:
            result.append((key, self.get_dataset(key)))
        return result

    # Mapping Keys Methods
    def keys(self):
        return self.keys_file_attributes() + self.keys_datasets()

    def keys_file_attributes(self):
        return list(self.file_attribute_names)

    def keys_datasets(self):
        return list(self.dataset_names)

    # Mapping Pop Methods
    def pop(self, key):
        if key in self._file_attrs:
            return self.pop_file_attribute(key)
        elif key in self._datasets:
            return self.pop_dataset(key)
        else:
            warn("No attribute or dataset " + key, stacklevel=2)
            return None

    def pop_file_attribute(self, key):
        value = self.get_file_attribute(key)
        del self.h5_fobj.attrs[key]
        return value

    def pop_dataset(self, key):
        value = self.get_dataset(key)[...]
        del self.h5_fobj[key]
        return value

    # Mapping Update Methods
    def update_file_attributes(self, **kwargs):
        if len(self.dataset_names.intersection(kwargs.keys())) > 0:
            warn("Dataset name already exists", stacklevel=2)
        if len(self.file_attribute_names.intersection(kwargs.keys())) > 0:
            warn("Attribute name already exists", stacklevel=2)

        op = self.is_open
        self.open()
        self._file_attrs.update(kwargs.keys())
        for key, value in kwargs:
            _key = "_" + key
            try:
                self.h5_fobj.attrs[key] = value
            except Exception as e:
                warn("Could not set attribute due to error: " + str(e), stacklevel=2)
            else:
                super().__setattr__(_key, value)
        if not op:
            self.close()

    def update_datasets(self, **kwargs):
        if len(self.dataset_names.intersection(kwargs.keys())) > 0:
            warn("Dataset name already exists", stacklevel=2)
        if len(self.file_attribute_names.intersection(kwargs.keys())) > 0:
            warn("Attribute name already exists", stacklevel=2)

        op = self.is_open
        self.open()
        self._datasets.update(kwargs.keys())
        for key, value in kwargs.items():
            _key = "_" + key
            try:
                args = merge_dict(self.default_datasets_parameters, value)
                self.h5_fobj.require_dataset(key, **args)
            except Exception as e:
                warn("Could not set datasets due to error: " + str(e), stacklevel=2)
            else:
                super().__setattr__(_key, type(self)(self.h5_fobj[key], self))
        if not op:
            self.close()

    # File Methods
    def open(self, mode="a", exc=False):
        if not self.is_open:
            try:
                self.h5_fobj = h5py.File(self.path.as_posix(), mode=mode)
            except Exception as e:
                if exc:
                    warn("Could not open" + self.path.as_posix() + "due to error: " + str(e), stacklevel=2)
                    self.h5_fobj = None
                    return None
                else:
                    raise e
            else:
                self.load_attributes()
                self.load_datasets()
                return self.h5_fobj

    def close(self):
        if self.is_open:
            self.h5_fobj.flush()
            self.h5_fobj.close()
        return not self.is_open

    # General Methods
    def append_to_dataset(self, name, data, axis=0):
        dataset = self.get_dataset(name)
        s_shape = dataset.shape
        d_shape = data.shape
        f_shape = list(s_shape)
        f_shape[axis] = s_shape[axis] + d_shape[axis]
        slicing = tuple(slice(s_shape[ax]) for ax in range(0, axis)) + (-d_shape[axis], ...)

        with dataset:
            dataset.resize(*f_shape)
            dataset[slicing] = data


class HDF5dataset(StaticWrapper):
    _wrapped_types = [h5py.Dataset]
    _wrap_attributes = ["_dataset"]

    # Static Methods
    @staticmethod
    def _create_callback_functions(call_name, name):
        """A factory for creating property modification functions which accesses an embedded objects attributes.

        Args:
            call_name (str): The name attribute the object to call is stored.
            name (str): The name of the attribute that this property will mask.

        Returns:
            get_: The get function for a property object.
            set_: The wet function for a property object.
            del_: The del function for a property object.
        """
        store_name = "_" + call_name

        def get_(obj):
            """Gets the wrapped object's attribute and check the temporary attribute if not."""
            try:
                # Todo: In StaticWrapper consider having this as a separate method to make it easier to use.
                with obj:
                    return getattr(getattr(obj, store_name), name)
            except AttributeError as error:
                try:
                    return getattr(obj, "__" + name)
                except AttributeError:
                    raise error

        def set_(obj, value):
            """Sets the wrapped object's attribute or saves it to a temporary attribute if wrapped object."""
            try:
                with obj:
                    setattr(getattr(obj, store_name), name, value)
            except AttributeError as error:
                if not hasattr(obj, store_name) or getattr(obj, store_name) is None:
                    setattr(obj, "__" + name, value)
                else:
                    raise error

        def del_(obj):
            """Deletes the wrapped object's attribute."""
            with obj:
                delattr(getattr(obj, store_name), name)

        return get_, set_, del_

    # Instantiation, Copy, Destruction
    def __init__(self, dataset=None, file=None, init=True):
        self._dataset = None
        self._name = None

        self._file = None
        self._file_was_open = None

        if init:
            self.construct(dataset=dataset, file=file)

    @property
    def dataset(self):
        return self._dataset

    # Container Magic Methods
    def __getitem__(self, item):
        with self:
            return self._dataset[item]

    def __setitem__(self, key, value):
        with self:
            self._dataset[self._name][key] = value

    # Context Managers
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    # Generic Methods #
    # Constructors Methods
    def construct(self, dataset=None, file=None):
        if isinstance(dataset, HDF5dataset):
            self._dataset = dataset._dataset
            self._name = dataset._name
            if file is None:
                self._file = dataset._file
            else:
                self._file = file
        else:
            if not dataset and file is None:
                raise ValueError("Dataset need to be open or a file must be given")
            elif file is None:
                self._file = HDF5Object(dataset.file)
            elif isinstance(file, HDF5Object):
                self._file = file
            else:
                self._file = HDF5Object(dataset.file.filename)

            self._dataset = dataset
            self._name = dataset.name

    def open(self, mode='a', **kwargs):
        self._file_was_open = self._file.is_open
        if not self._dataset:
            self._file.open(mode=mode, **kwargs)
            self._dataset = self._file[self._name]

        return self

    def close(self):
        if not self._file_was_open:
            self._file.close()
