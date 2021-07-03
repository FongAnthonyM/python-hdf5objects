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
from contextlib import contextmanager
import pathlib
from warnings import warn

# Downloaded Libraries #
from baseobjects import BaseObject, StaticWrapper
import h5py

# Local Libraries #


# Definitions #
# Classes #
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


class HDF5Object(BaseObject):
    """A class which wraps a HDF5 File and gives more functionality.

    Attributes:
        _file_attrs (set): The names of the attributes in the HDF5 file.
        _datasets (set): The names of the datasets in the HDF5 file.
        _path (obj:`Path`): The path to were the HDF5 file exists.
        is_updating (bool): Determines if this object should constantly open the file for updating attributes.

        c_kwargs: The keyword arguments for the data compression.
        default_dataset_kwargs: The default keyword arguments for datasets when they are created.
        default_file_attributes (dict): The default file attributes the HDF5 file should have.
        default_datasets (dict): The default datasets the HDF5 file should have.

        hf_fobj: The HDF5 File object this object wraps.

    Args:
        path: The path to the HDF5 file.
        update (bool): Determines if this object should constantly open the file for updating attributes.
        open_ (bool): Determines if this object will remain open after construction.
        init (bool): Determines if this object should initialize.
        **kwargs: The keyword arguments for the open method.
    """
    # Todo: Rethink about how Errors and Warnings are handled in this object.
    dataset_type = HDF5Dataset

    # Magic Methods
    # Construction/Destruction
    def __init__(self, path=None, update=True, open_=False, init=False, **kwargs):
        self._file_attrs = set()
        self._datasets = set()
        self._path = None
        self.is_updating = True

        self.c_kwargs = {"compression": "gzip", "compression_opts": 4}
        self.default_dataset_kwargs = self.c_kwargs.copy()
        self.default_file_attributes = {}
        self.default_datasets = {}

        self.h5_fobj = None

        if init:
            self.construct(path, update, open_, **kwargs)

    @property
    def path(self):
        """:obj:`Path`: The path to the file.

        The setter casts objects that are not Path to path before setting
        """
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    @property
    def is_open(self):
        """bool: Determines if the hdf5 file is open."""
        try:
            return bool(self.h5_fobj)
        except:
            return False

    @property
    def file_attribute_names(self):
        """:obj:`set` of `str`: The names of the file attributes in the HDF5 file."""
        if not self._file_attrs or self.is_updating:
            return self.get_file_attribute_names()
        else:
            return self._file_attrs

    @property
    def dataset_names(self):
        """:obj:`set` of `str`: The names of the datasets in the HDF5 file."""
        if not self._datasets or self.is_updating:
            return self.get_dataset_names()
        else:
            return self._datasets

    def __del__(self):
        """Closes the file when this object is deleted."""
        self.close()

    # Pickling
    def __getstate__(self):
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            dict: A dictionary of this object's attributes.
        """
        state = self.__dict__.copy()
        state["open_state"] = self.is_open
        return state

    def __setstate__(self, state):
        """Builds this object based on a dictionary of corresponding attributes.

        Args:
            state (dict): The attributes to build this object from.
        """
        state["h5_fobj"] = h5py.File(state["path"].as_posix(), "r+")
        if not state.pop("open_state"):
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
        """Returns the length of the HDF5 file.

        Returns:
            int: The length of HDF5 file.
        """
        with self._temp_open():
            length = len(self.h5_fobj)
        return length

    def __getitem__(self, item):
        """Gets a dataset from the HDF5 file based on the arguments.

        Args:
            item (str): The name of the dataset to get.

        Returns:
            The dataset requested.
        """
        if item in self.dataset_names:
            data = self.get_dataset(item)
        else:
            raise KeyError(item)
        return data

    def __setitem__(self, key, value):
        """Sets a dataset in the HDF5 file.

        Args:
            key (str): The name of the dataset to set in the HDF5 file.
            value: The new dataset to assign in the HDF5 file.
        """
        self.set_dataset(key, value)

    # Context Managers
    def __enter__(self):
        """The context enter which opens the HDF5 file.

        Returns:
            This object.
        """
        if self.h5_fobj is None:
            self.construct(open_=True)
        else:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """The context exit which closes the file."""
        return self.close()

    # Type Conversion
    def __bool__(self):
        """When cast as a bool, this object True if open and False if closed.

        Returns:
            bool: If this object is open or not.
        """
        return self.is_open

    # Instance Methods
    # Constructors/Destructors
    def construct(self, path=None, update=None, open_=False, **kwargs):
        """Constructs this object.

        Args:
            path: The path to the HDF5 file.
            update (bool): Determines if this object should constantly open the file for updating attributes.
            open_ (bool): Determines if this object will remain open after construction.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if path is not None:
            self.path = path
        if update is not None:
            self.is_updating = update

        if self.path.is_file():
            self.open(**kwargs)
            if not open_:
                self.close()
        else:
            self.create_file(open_=open_)

        return self

    # File Creation
    def create_file(self, attr={}, data={}, construct=True, open_=False):
        """Creates the HDF5 file.

        Args:
            attr (dict, optional): File attributes to set when the file is created.
            data (dict, optional): Datasets to assign when the file is created.
            construct (bool, optional): Determines if the file will be constructed.
            open_ (bool, optional): Determines if this object will remain open after construction.

        Returns:
            This object.
        """
        self.open()
        if construct:
            self.construct_file(attr, data)

        elif not open_:
            self.close()

        return self

    def construct_file(self, attr={}, data={}):
        """Constructs the file with file attributes and datasets.

        Args:
            attr (dict, optional): File attributes to set when the file is created.
            data (dict, optional): Datasets to assign when the file is created.
        """
        self.construct_file_attributes(**attr)
        self.construct_file_datasets(**data)

    def construct_file_attributes(self, **kwargs):
        """Sets the file attributes based on the default and given attributes.

        Args:
            **kwargs: File attributes to set when the file is created.
        """
        a_kwargs = self.default_file_attributes.copy()
        a_kwargs.update(kwargs)
        self.update_file_attributes(**a_kwargs)

    def construct_file_datasets(self, **kwargs):
        """Assigns the file's datasets based on the default and given datasets.

        Args:
            **kwargs: Datasets to assign when the file is created.
        """
        d_kwargs = self.default_dataset_kwargs.copy()
        d_kwargs.update(kwargs)
        self.update_datasets(**d_kwargs)

    # def copy_file(self, path):  # Todo: Implement this.
    #     pass

    # File
    def open(self, mode='a', exc=False, **kwargs):
        """Opens the HDF5 file.

        Args:
            mode (str): The mode which this file should be opened in.
            exc (bool): Determines if an error should be excepted as warning or not.
            kwargs: The keyword arguments for opening the HDF5 file.

        Returns:
            This object.
        """
        if not self.is_open:
            try:
                self.h5_fobj = h5py.File(self.path.as_posix(), mode=mode, **kwargs)
                self.load_attributes()
                self.load_datasets()
                return self
            except Exception as error:
                if exc:
                    warn("Could not open" + self.path.as_posix() + "due to error: " + str(error), stacklevel=2)
                    self.h5_fobj = None
                    return self
                else:
                    raise error

    @contextmanager
    def _temp_open(self, **kwargs):
        """Temporarily opens the file if it is not already open.

        Args:
            **kwargs: The keyword arguments for opening the HDF5 file.

        Returns:
            This object.
        """
        was_open = self.is_open
        self.open(**kwargs)
        try:
            yield self
        finally:
            if not was_open:
                self.close()

    def close(self):
        """Closes the HDF5 file.

        Returns:
            bool: If the file was successfully closed.
        """
        if self.is_open:
            self.h5_fobj.flush()
            self.h5_fobj.close()
        return not self.is_open

    # File Attributes
    # Names
    def get_file_attribute_names(self):
        """Gets the file attributes names from the HDF5 file.

        Returns:
            set: All the file attribute names in the HDF5 file.
        """
        self.get_file_attributes()
        return self._file_attrs

    def list_file_attributes(self):
        """Gets the file attributes names from the HDF5 file.

        Returns:
            list: All the file attribute names in the HDF5 file.
        """
        return list(self.get_file_attribute_names())

    # Getters/Setters
    def get_file_attribute(self, name):
        """Gets an attribute from the HDF5 file.

        Args:
            name (str): The name of the file attribute to get.

        Returns:
            The attribute requested.
        """
        __name = "__" + name

        if self.__name not in self._file_attrs or self.is_updating:
            try:
                with self._temp_open():
                    setattr(self, __name, self.h5_fobj.attrs[name])
            except Exception as e:
                warn("Could not update attribute due to error: "+str(e), stacklevel=2)

        return getattr(self, __name)

    def get_file_attributes(self):
        """Gets all file attributes from the HDF5 file.

        Returns:
            dict: The file attributes.
        """
        with self._temp_open():
            attrs = dict(self.h5_fobj.attrs.items())

        for name, value in attrs.items():
            self._file_attrs.update((name,))
            setattr(self, "__" + name, value)

        return attrs

    def set_file_attribute(self, name, value):
        """Sets a file attribute for the HDF5 file.

        Args:
            name (str): The name of the file attribute to set.
            value (Any): The object to set the file attribute to.
        """
        __name = "__" + name

        try:
            with self._temp_open():
                self.h5_fobj.attrs[name] = value
                if name not in self._file_attrs:
                    self._file_attrs.update((name,))
                setattr(self, __name, value)
        except Exception as e:
            warn("Could not set attribute due to error: " + str(e), stacklevel=2)

    def del_file_attribute(self, name):
        """Deletes an attribute from the HDF5 file.

        Args:
            name (str): The name of the file attribute to delete.

        """
        __name = "__" + name
        try:
            with self._temp_open():
                del self.h5_fobj.attrs[name]  # Todo: Check if this works.
                delattr(self, __name)
        except Exception as e:
            warn("Could not update attribute due to error: " + str(e), stacklevel=2)

    def update_file_attributes(self, **items):
        """Updates the file attributes based on the dictionary update scheme.

        Args:
            **items: The keyword arguments which are the attributes an their values.
        """
        with self._temp_open():
            for name, value in items.items():
                __name = "__" + name
                try:
                    self.h5_fobj.attrs[name] = value
                    if name not in self._file_attrs:
                        self._file_attrs.update((name,))
                    setattr(self, __name, value)
                except Exception as e:
                    warn("Could not set attribute due to error: " + str(e), stacklevel=2)

    def clear_file_attributes(self):
        """Clears set tracking the file attributes and the local object attributes."""
        for key in self._file_attrs:
            self.__delattr__("__" + key)
        self._file_attrs.clear()

    def load_file_attributes(self):
        """Loads all the file attributes from the HDF5 file in this object."""
        self.get_file_attributes()

    # Datasets
    # Names
    def get_dataset_names(self):
        """Gets the dataset names from the HDF5 file.

        Returns:
            set: All the dataset names in the HDF5 file.
        """
        with self._temp_open():
            for name, value in self.h5_fobj.items():
                self._datasets.update((name,))
                setattr(self, "__d_" + name, HDF5Dataset(value, self))

        return self._datasets

    def list_dataset_names(self):
        """Gets the dataset names from the HDF5 file.

        Returns:
            list: All the dataset names in the HDF5 file.
        """
        return list(self.get_datasets_names())

    # Getters/Setters
    def get_dataset(self, name):
        """Gets a dataset from the HDF5 file.

        Args:
            name (str): The name of the dataset to get.

        Returns:
            The dataset requested.
        """
        __name = "__d_" + name

        if __name not in self._datasets or self.is_updating:
            try:
                with self._temp_open():
                    setattr(self, __name, self.dataset_type(self.h5_fobj[name], self))
            except Exception as e:
                warn("Could not update datasets due to error: " + str(e), stacklevel=2)

        return getattr(self, __name)

    def get_datasets(self):
        """Gets all the datasets from the HDF5 file.

        Returns:
            dict: The datasets.
        """
        with self._temp_open():
            datasets = {}
            for name, value in self.h5_fobj.items():
                dataset = self.dataset_type(value, self)
                datasets[name] = dataset
                self._datasets.update((name,))
                setattr(self, "__d_" + name, dataset)

        return datasets

    def set_dataset(self, name, data=None, **kwargs):
        """Sets a dataset with the HDF5 file.

        Args:
            name (str): The name of the dataset in the HDF5 file.
            data: The data to add to the dataset.
            **kwargs: The keyword arguments for the new dataset.
        """
        __name = "__d_" + name

        try:
            with self._temp_open():
                if name in self._datasets:
                    self.h5_fobj[name].resize(data.shape)
                    self.h5_fobj[name][...] = data
                else:
                    d_kwargs = self.default_dataset_kwargs.copy()
                    d_kwargs.update(kwargs)
                    d_kwargs["data"] = data

                    self.h5_fobj.require_dataset(name, **d_kwargs)
                    self._datasets.update((name,))
                    setattr(self, __name, self.dataset_type(self.h5_fobj[name], self))
        except Exception as e:
            warn("Could not set datasets due to error: " + str(e), stacklevel=2)

    def del_dataset(self, name):
        """Deletes a dataset from the HDF5 file.

        Args:
            name (str): The name of the dataset to delete.
        """
        __name = "__d_" + name

        try:
            with self._temp_open():
                del self.h5_fobj[name]  # Todo: Check if this works.
                delattr(self, __name)
        except Exception as e:
            warn("Could not update datasets due to error: " + str(e), stacklevel=2)

    def update_file_datasets(self, **items):
        """Updates the datasets based on the dictionary update scheme.

        Args:
            **items: The keyword arguments which are the attributes an their values.
        """
        with self._temp_open():
            for name, kwargs in items.items():
                __name = "__d_" + name
                try:
                    if name in self._datasets:
                        self.h5_fobj[name].resize(kwargs["data"].shape)
                        self.h5_fobj[name][...] = kwargs["data"]
                    else:
                        self._datasets.update((name,))
                        d_kwargs = self.default_dataset_kwargs.copy()
                        d_kwargs.update(kwargs)

                        self.h5_fobj.require_dataset(name, **d_kwargs)
                        self._datasets.update((name,))
                        setattr(self, __name, self.dataset_type(self.h5_fobj[name], self))
                except Exception as e:
                    warn("Could not set datasets due to error: " + str(e), stacklevel=2)

    def clear_datasets(self):
        """Clears set tracking the datasets and the local object datasets."""
        for name in self._datasets:
            self.__delattr__("_" + name)
        self._datasets.clear()

    def load_datasets(self):
        """Loads all the datasets from the HDF5 file in this object."""
        self.get_dataset_names()

    # General
    def create_dataset(self, name, data=None, **kwargs):
        """Creates a dataset in the HDF5 file.

         Args:
            name (str): The name of the dataset in the HDF5 file.
            data: The data to add to the dataset.
            **kwargs: The keyword arguments for the new dataset.

        Returns:
            The new dataset that was created.
        """
        self.set_dataset(name, data, **kwargs)
        return self.get_dataset(name)

    def append_to_dataset(self, name, data, axis=0):
        """Append data to the dataset along a specified axis.

        Args:
            name (str): The name of the dataset to append the data to.
            data: The data to append.
            axis (int): The axis to append the data along.
        """
        self.get_dataset(name).append(data, axis)

    # Getters and Setters
    def get(self, name):
        """A general get which return either an attribute or a dataset.

        Args:
            name (str): The name of the item to get.

        Returns:
            Either the attribute or dataset requested.
        """
        if name in self.file_attribute_names:
            return self.get_file_attribute(name)
        elif name in self.dataset_names:
            return self.get_dataset(name)
        else:
            warn("No attribute or dataset " + name, stacklevel=2)
            return None

    #  Mapping Items
    def items(self):
        """All attributes and datasets as a list of items, keys and values.

        Returns:
            list: All keys and values.
        """
        return self.items_file_attributes() + self.items_datasets()

    def items_file_attributes(self):
        """All attributes as list a of items, keys and values.

        Returns:
            list: All keys and values.
        """
        result = []
        for key in self.file_attribute_names:
            result.append((key, self.get_file_attribute(key)))
        return result

    def items_datasets(self):
        """All datasets as a list of items, keys and values.

        Returns:
            list: All keys and values.
        """
        result = []
        for key in self.dataset_names:
            result.append((key, self.get_dataset(key)))
        return result

    # Mapping Keys
    def keys(self):
        """All attribute and dataset names as a list of keys.

        Returns:
            list: All keys.
        """
        return self.keys_file_attributes() + self.keys_datasets()

    def keys_file_attributes(self):
        """All attribute names as a list of keys.

        Returns:
            list: All keys.
        """
        return list(self.file_attribute_names)

    def keys_datasets(self):
        """All dataset names as a list of keys.

        Returns:
            list: All keys.
        """
        return list(self.dataset_names)

    # Mapping Pop
    def pop(self, name):
        """Get either an attribute or dataset then deletes it in the HDF5 file.

        Args:
            name (str): The name of the item to pop.

        Returns:
            Either the attribute or dataset requested.
        """
        if name in self._file_attrs:
            return self.pop_file_attribute(name)
        elif name in self._datasets:
            return self.pop_dataset(name)
        else:
            warn("No attribute or dataset " + name, stacklevel=2)
            return None

    def pop_file_attribute(self, name):
        """Gets an attribute then deletes it in the HDF5 file.

        Args:
            name (str): The name of the attribute to pop.

        Returns:
            The attribute requested.
        """
        value = self.get_file_attribute(name)
        self.del_file_attribute(name)
        return value

    def pop_dataset(self, name):
        """Gets a dataset then deletes it in the HDF5 file.

        Args:
            name (str): The name of the dataset to pop.

        Returns:
            The dataset requested.
        """
        value = self.get_dataset(name)[...]
        self.del_dataset(name)
        return value
