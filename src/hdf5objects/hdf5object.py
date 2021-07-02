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
from warnings import warn

# Downloaded Libraries #
from baseobjects import BaseObject, StaticWrapper
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


# Classes #
class HDF5Object(BaseObject):
    # Magic Methods
    # Construction/Destruction
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

    # Container Methods
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

    # Instance Methods
    # Constructors/Destructors
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
                super().__setattr__(_item, HDF5Dataset(self.h5_fobj[item], self))
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
            super().__setattr__(_key, HDF5Dataset(self.h5_fobj[name], self))

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
                super().__setattr__(_key, HDF5Dataset(self.h5_fobj[name], self))
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

    #  Mapping Items
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

    # Mapping Keys
    def keys(self):
        return self.keys_file_attributes() + self.keys_datasets()

    def keys_file_attributes(self):
        return list(self.file_attribute_names)

    def keys_datasets(self):
        return list(self.dataset_names)

    # Mapping Pop
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

    # Mapping Update
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

    # File
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


class HDF5Dataset(StaticWrapper):
    """A wrapper object which wraps a HDF5 dataset and gives more functionality.

    Attributes:
        _name (str): The name of this dataset.
        _dataset: The HDF5 dataset to wrap.
        _file_was_open (bool): Determines if the file object was open when this dataset was accessed.
        _file: The file this dataset is from.

    Args:
        dataset: The HDF5 dataset to build this dataset around.
        file: The file which the dataset originates from.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.Dataset]
    _wrap_attributes = ["_dataset"]

    # Class Methods
    # Wrapped Attribute Callback Functions
    @classmethod
    def _get_attribute(cls, obj, wrap_name, attr_name):
        """Gets an attribute from a wrapped dataset.

        Args:
            obj (Any): The target object to get the wrapped object from.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to get from the wrapped object.

        Returns:
            (Any): The wrapped object.
        """
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
            super()._get_attribute(obj, wrap_name, attr_name)

    @classmethod
    def _set_attribute(cls, obj, wrap_name, attr_name, value):
        """Sets an attribute in a wrapped dataset.

        Args:
            obj (Any): The target object to set.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to set from the wrapped object.
            value (Any): The object to set the wrapped objects attribute to.
        """
        with obj: # Ensures the hdf5 dataset is open when accessing attributes
            super()._set_attribute(obj, wrap_name, attr_name, value)

    @classmethod
    def _del_attribute(cls, obj, wrap_name, attr_name):
        """Deletes an attribute in a wrapped dataset.

        Args:
            obj (Any): The target object to set.
            wrap_name (str): The attribute name of the wrapped object.
            attr_name (str): The attribute name of the attribute to delete from the wrapped object.
        """
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
            super()._del_attribute(obj, wrap_name, attr_name)

    # Magic Methods
    # Constructors/Destructors
    def __init__(self, dataset=None, file=None, init=True):
        self._name = None
        self._dataset = None

        self._file_was_open = None
        self._file = None

        if init:
            self.construct(dataset=dataset, file=file)

    # Container Methods
    def __getitem__(self, key):
        """Allows slicing access to the data in the dataset.

        Args:
            key: A slice used to get data from the dataset.

        Returns:
            Data from the dataset base on the slice.
        """
        with self:
            return self._dataset[key]

    def __setitem__(self, key, value):
        """Allows slicing assignment of the data in the dataset.

        Args:
            key: A slice used to get data from the dataset.
            value: Data to assign to the slice in the dataset.
        """
        with self:
            self._dataset[self._name][key] = value

    # Context Managers
    def __enter__(self):
        """The enter context which opens the file to make this dataset usable"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """The exit context which close the file."""
        return self.close()

    # Instance Methods
    # Constructors/Destructors
    def construct(self, dataset=None, file=None):
        """Constructs this object from the provided arguments.

        Args:
            dataset: The HDF5 dataset to build this dataset around.
            file: The file which the dataset originates from.
        """
        if isinstance(dataset, HDF5Dataset):
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

    # File
    def open(self, mode='a', **kwargs):
        """Opens the file to make this dataset usable.

        Args:
            mode (str, optional): The file mode to open the file with.
            **kwargs (dict, optional): The additional keyword arguments to open the file with.

        Returns:
            This object.
        """
        self._file_was_open = self._file.is_open
        if not self._dataset:
            self._file.open(mode=mode, **kwargs)
            self._dataset = self._file[self._name]

        return self

    def close(self):
        """Closes the file of this dataset."""
        if not self._file_was_open:
            self._file.close()

    # Data Modification
    def append(self, data, axis=0):
        """Append data to the dataset along a specified axis.

        Args:
            data: The data to append.
            axis (int): The axis to append the data along.
        """
        with self:
            # Get the shapes of the dataset and the new data to be added
            s_shape = self._dataset.shape
            d_shape = data.shape
            # Determine the new shape of the dataset
            new_shape = list(s_shape)
            new_shape[axis] = s_shape[axis] + d_shape[axis]
            # Determine the location where the new data should be assigned
            slicing = tuple(slice(s_shape[ax]) for ax in range(0, axis)) + (-d_shape[axis], ...)

            # Assign Data
            self._dataset.resize(new_shape)  # Reshape for new data
            self._dataset[slicing] = data    # Assign data to the new location
