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
from abc import abstractmethod
from contextlib import contextmanager
import pathlib
from warnings import warn

# Downloaded Libraries #
from baseobjects import BaseObject, StaticWrapper
import h5py
from multipledispatch import dispatch

# Local Libraries #


# Definitions #
# Classes #
class HDF5BaseWrapper(StaticWrapper):
    """An abstract wrapper which wraps object from an HDF5 file and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this dataset was accessed.
        _file: The file this dataset is from.
        _name (str): The name of the object from the HDF5 file.

    Args:
        file: The file which the dataset originates from.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.Group]
    _wrap_attributes = ["_group"]

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
        with obj:  # Ensures the hdf5 dataset is open when accessing attributes
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
    def __init__(self, file=None, init=True):
        self._file_was_open = None
        self._file = None

        self._name = None

        if init:
            self.construct(file=file)

    # Container Methods
    def __getitem__(self, key):
        """Ensures HDF5 object is open for getitem"""
        with self:
            return getattr(self, self._wrap_attributes[0])[key]

    def __setitem__(self, key, value):
        """Ensures HDF5 object is open for setitem"""
        with self:
            getattr(self, self._wrap_attributes[0])[key] = value

    def __delitem__(self, key):
        """Ensures HDF5 object is open for delitem"""
        with self:
            del getattr(self, self._wrap_attributes[0])[key]

    # Context Managers
    def __enter__(self):
        """The enter context which opens the file to make this dataset usable"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """The exit context which close the file."""
        self.close()

    # Type Conversion
    def __bool__(self):
        """When cast as a bool, this object True if valid and False if not.

        Returns:
            bool: If this object is open or not.
        """
        return bool(getattr(self, self._wrap_attributes[0]))

    # Instance Methods
    # Constructors/Destructors
    def construct(self, file=None):
        """Constructs this object from the provided arguments.

        Args:
            file: The file which the dataset originates from.
        """
        if file is not None:
            self.set_file(file)

    # Getters/Setters
    @dispatch(object)
    def set_file(self, file):
        """Sets the file for this object to an HDF5Object.

        Args:
            file: An object to set the file to.
        """
        if isinstance(file, HDF5Object):
            self._file = file
        else:
            raise ValueError("file must be a path, File, or HDF5Object")

    @dispatch((str, pathlib.Path, h5py.File))
    def set_file(self, file):
        """Sets the file for this object to an HDF5Object.

        Args:
            file: An object to set the file to.
        """
        self._file = HDF5Object(file)

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
        if not getattr(self, self._wrap_attributes[0]):
            self._file.open(mode=mode, **kwargs)
            setattr(self, self._wrap_attributes[0], self._file[self._name])

        return self

    def close(self):
        """Closes the file of this dataset."""
        if not self._file_was_open:
            self._file.close()


class HDF5Attributes(HDF5BaseWrapper):
    """A wrapper object which wraps a HDF5 attribute manager and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this attribute_manager was accessed.
        _file: The file this attribute_manager is from.
        _name (str): The name of this attribute_manager.
        _attribute_manager: The HDF5 attribute_manager to wrap.
        _attribute_names: The names of the attributes.
        is_updating (bool): Determines if this object should constantly open the file for updating attributes.

    Args:
        attributes: The HDF5 attribute_manager to build this attribute_manager around.
        name (str): The location of the attributes in the HDF5 file.
        file: The file which the attribute_manager originates from.
        update (bool): Determines if this object should constantly open the file for updating attributes.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.AttributeManager]
    _wrap_attributes = ["_attribute_manager"]

    # Magic Methods
    # Constructors/Destructors
    def __init__(self, attributes=None, name=None, file=None, update=None, init=True):
        super().__init__(file=file, init=False)
        self._attribute_manager = None

        self._attribute_names = set()
        self.is_updating = True

        if init:
            self.construct(attributes, name, file, update)

    # Container Methods
    def __getitem__(self, name):
        """Ensures HDF5 object is open for getitem"""
        return self.get_attribute(name)

    def __setitem__(self, name, value):
        """Ensures HDF5 object is open for setitem"""
        self.set_attribute(name, value)

    def __delitem__(self, name):
        """Ensures HDF5 object is open for delitem"""
        self.del_attribute(name)

    def __iter__(self):
        """Ensures HDF5 object is open for iter"""
        with self:
            return self._attribute_manager.__iter__()

    def __contains__(self, item):
        """Ensures HDF5 object is open for contains"""
        with self:
            return item in self._attribute_manager

    # Instance Methods
    # Constructors/Destructors
    def construct(self, attributes=None, name=None, file=None, update=None):
        """Constructs this object from the provided arguments.

        Args:
            attributes: The HDF5 attribute_manager to build this attribute_manager around.
            name (str): The location of the attributes in the HDF5 file.
            file: The file which the attribute_manager originates from.
            update (bool): Determines if this object should constantly open the file for updating attributes.
        """
        if name is None and not (isinstance(attributes, str) and isinstance(attributes, HDF5BaseWrapper)):
            raise ValueError("name must be given if or an object with the name.")
        if file is None and (isinstance(attributes, str) or attributes is None):
            raise ValueError("A file must be given if giving attribute manager name")

        if attributes is not None:
            self.set_attribute_manager(attributes)

        if name is not None:
            self._name = name

        if file is not None:
            self.set_file(file)

        if update is not None:
            self.is_updating = update

        self.load_attributes()

    # Getters/Setters
    @dispatch(object)
    def set_attribute_manager(self, attributes):
        """Sets the wrapped attribute_manager.

        Args:
            attributes: The attribute_manager this object will wrap.
        """
        if isinstance(attributes, HDF5BaseWrapper):
            self._file = attributes._file
            self._name = attributes._name
            if isinstance(attributes, HDF5Attributes):
                self._attribute_manager = attributes._attribute_manager
        else:
            raise ValueError("attribute_manager must be a Dataset or HDF5Dataset")

    @dispatch(str)
    def set_attribute_manager(self, attributes):
        """Sets the wrapped attribute_manager base on a str.

        Args:
            attributes (str): The name of the attributes
        """
        self._name = attributes

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
        if not self._attribute_manager:
            self._file.open(mode=mode, **kwargs)
            self._attribute_manager = self._file.h5_fobj[self._name].attr

    # Names
    def get_attribute_names(self):
        """Gets the file attributes names from the HDF5 file.

        Returns:
            set: All the file attribute names in the HDF5 file.
        """
        self.get_attributes()
        return self._attribute_names

    def list_attributes(self):
        """Gets the file attributes names from the HDF5 file.

        Returns:
            list: All the file attribute names in the HDF5 file.
        """
        return list(self.get_attribute_names())

    # Getters/Setters
    def get_attribute(self, name):
        """Gets an attribute from the HDF5 file.

        Args:
            name (str): The name of the file attribute to get.

        Returns:
            The attribute requested.
        """
        __name = "__" + name

        if __name not in self._attribute_names or self.is_updating:
            try:
                with self:
                    setattr(self, __name, self._attribute_manager[name])
            except Exception as e:
                warn("Could not update attribute due to error: " + str(e), stacklevel=2)

        return getattr(self, __name)

    def get_attributes(self):
        """Gets all file attributes from the HDF5 file.

        Returns:
            dict: The file attributes.
        """
        with self:
            attrs = dict(self._attribute_manager.items())

        for name, value in attrs.items():
            self._attribute_names.update((name,))
            setattr(self, "__" + name, value)

        return attrs

    def set_attribute(self, name, value):
        """Sets a file attribute for the HDF5 file.

        Args:
            name (str): The name of the file attribute to set.
            value (Any): The object to set the file attribute to.
        """
        __name = "__" + name

        try:
            with self:
                self._attribute_manager[name] = value
                if name not in self._attribute_names:
                    self._attribute_names.update((name,))
                setattr(self, __name, value)
        except Exception as e:
            warn("Could not set attribute due to error: " + str(e), stacklevel=2)

    def del_attribute(self, name):
        """Deletes an attribute from the HDF5 file.

        Args:
            name (str): The name of the file attribute to delete.

        """
        __name = "__" + name
        try:
            with self:
                del self._attribute_manager[name]  # Todo: Check if this works.
                delattr(self, __name)
        except Exception as e:
            warn("Could not update attribute due to error: " + str(e), stacklevel=2)

    def update_attributes(self, **items):
        """Updates the file attributes based on the dictionary update scheme.

        Args:
            **items: The keyword arguments which are the attributes an their values.
        """
        with self:
            for name, value in items.items():
                __name = "__" + name
                try:
                    self._attribute_manager[name] = value
                    if name not in self._attribute_names:
                        self._attribute_names.update((name,))
                    setattr(self, __name, value)
                except Exception as e:
                    warn("Could not set attribute due to error: " + str(e), stacklevel=2)

    def clear_attributes(self):
        """Clears set tracking the file attributes and the local object attributes."""
        for key in self._attribute_names:
            self.__delattr__("__" + key)
        self._attribute_names.clear()

    def load_attributes(self):
        """Loads all the file attributes from the HDF5 file in this object."""
        self.get_attributes()


class HDF5Group(HDF5BaseWrapper):
    """A wrapper object which wraps a HDF5 group and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this dataset was accessed.
        _file: The file this dataset is from.
        _name (str): The name of this dataset.
        _group: The HDF5 group to wrap.
        attributes (:obj:`HDF5Attributes`): The attributes of this group.

    Args:
        group: The HDF5 dataset to build this dataset around.
        file: The file which the dataset originates from.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.Group]
    _wrap_attributes = ["_group"]

    # Magic Methods
    # Constructors/Destructors
    def __init__(self, group=None, file=None, init=True):
        super().__init__(file=file, init=False)
        self._group = None
        self.attributes = None

        if init:
            self.construct(group=group, file=file)

    # Container Methods
    def __getitem__(self, key):
        """Ensures HDF5 object is open for getitem"""
        with self:
            h_object = self._group[key]
            if isinstance(h_object, h5py.Dataset):
                return HDF5Dataset(h_object, self._file)
            else:
                return type(self)(h_object, self._file)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, group=None, file=None):
        """Constructs this object from the provided arguments.

        Args:
            group: The HDF5 group to build this group around.
            file: The file which the group originates from.
        """
        if file is None and isinstance(group, str):
            raise ValueError("A file must be given if giving a group name")

        if group is not None:
            self.set_group(group)

        if file is not None:
            self.set_file(file)

        self.attributes = HDF5Attributes(name=self._name, file=self._file)

    # Getters/Setters
    @dispatch(object)
    def set_group(self, group):
        """Sets the wrapped group.

        Args:
            group: The group this object will wrap.
        """
        if isinstance(group, HDF5Group):
            self._file = group._file
            self._name = group._name
            self._group = group._group
        else:
            raise ValueError("group must be a Dataset or HDF5Dataset")

    @dispatch(h5py.Group)
    def set_group(self, group):
        """Sets the wrapped group.

        Args:
            group (:obj:`Group`): The group this object will wrap.
        """
        if not group:
            raise ValueError("Dataset needs to be open")
        self._file = HDF5Object(group.file)
        self._name = group.name
        self._group = group

    @dispatch(str)
    def set_group(self, group):
        """Sets the wrapped group base on a str.

        Args:
            group (str): The name of the group.
        """
        self._name = group


class HDF5Dataset(HDF5BaseWrapper):
    """A wrapper object which wraps a HDF5 dataset and gives more functionality.

    Attributes:
        _file_was_open (bool): Determines if the file object was open when this dataset was accessed.
        _file: The file this dataset is from.
        _name (str): The name of this dataset.
        _dataset: The HDF5 dataset to wrap.
        attributes (:obj:`HDF5Attributes`): The attributes of this dataset.

    Args:
        dataset: The HDF5 dataset to build this dataset around.
        file: The file which the dataset originates from.
        init (bool): Determines if this object should initialize.
    """
    _wrapped_types = [h5py.Dataset]
    _wrap_attributes = ["_dataset"]

    # Magic Methods
    # Constructors/Destructors
    def __init__(self, dataset=None, file=None, init=True):
        super().__init__(file=file, init=False)
        self._dataset = None
        self.attributes = None

        if init:
            self.construct(dataset=dataset, file=file)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, dataset=None, file=None):
        """Constructs this object from the provided arguments.

        Args:
            dataset: The HDF5 dataset to build this dataset around.
            file: The file which the dataset originates from.
        """
        if file is None and isinstance(dataset, str):
            raise ValueError("A file must be given if giving dataset name")

        if dataset is not None:
            self.set_dataset(dataset)

        if file is not None:
            self.set_file(file)

        self.attributes = HDF5Attributes(name=self._name, file=self._file)

    # Getters/Setters
    @dispatch(object)
    def set_dataset(self, dataset):
        """Sets the wrapped dataset.

        Args:
            dataset: The dataset this object will wrap.
        """
        if isinstance(dataset, HDF5Dataset):
            self._file = dataset._file
            self._name = dataset._name
            self._dataset = dataset._dataset
        else:
            raise ValueError("dataset must be a Dataset or HDF5Dataset")

    @dispatch(h5py.Dataset)
    def set_dataset(self, dataset):
        """Sets the wrapped dataset.

        Args:
            dataset (:obj:`Dataset`): The dataset this object will wrap.
        """
        if not dataset:
            raise ValueError("Dataset needs to be open")
        self._file = HDF5Object(dataset.file)
        self._name = dataset.name
        self._dataset = dataset

    @dispatch(str)
    def set_dataset(self, dataset):
        """Sets the wrapped dataset base on a str.

        Args:
            dataset (str): The name of the dataset.
        """
        self._name = dataset

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

    Class Attributes:
        attribute_type: The class to cast the HDF5 attribute manager as.
        group_type: The class to cast the HDF5 group as.
        dataset_type: The class to cast the HDF5 dataset as.

    Attributes:
        _file_attrs (set): The names of the attributes in the HDF5 file.
        _containers (set): The names of the containers in the HDF5 file.
        _path (obj:`Path`): The path to were the HDF5 file exists.
        is_updating (bool): Determines if this object should constantly open the file for updating attributes.

        c_kwargs: The keyword arguments for the data compression.
        default_dataset_kwargs: The default keyword arguments for datasets when they are created.
        default_file_attributes (dict): The default file attributes the HDF5 file should have.
        default_datasets (dict): The default datasets the HDF5 file should have.

        hf_fobj: The HDF5 File object this object wraps.

    Args:
        obj: An object to build this object from. It can be the path to the file or a File objects.
        update (bool): Determines if this object should constantly open the file for updating attributes.
        open_ (bool): Determines if this object will remain open after construction.
        init (bool): Determines if this object should initialize.
        **kwargs: The keyword arguments for the open method.
    """
    # Todo: Rethink about how Errors and Warnings are handled in this object.
    attribute_type = HDF5Attributes
    group_type = HDF5Group
    dataset_type = HDF5Dataset

    # Magic Methods
    # Construction/Destruction
    def __init__(self, obj=None, update=True, open_=False, init=False, **kwargs):
        self._containers = set()
        self._path = None
        self.is_updating = True

        self.attributes = None

        self.c_kwargs = {"compression": "gzip", "compression_opts": 4}
        self.default_dataset_kwargs = self.c_kwargs.copy()
        self.default_file_attributes = {}
        self.default_datasets = {}

        self.h5_fobj = None

        if init:
            self.construct(obj, update, open_, **kwargs)

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
    def file_attributes(self):
        """Returns the attributes from the file."""
        return self.attributes

    @property
    def container_names(self):
        """:obj:`set` of `str`: The names of the containers in the HDF5 file."""
        if not self._containers or self.is_updating:
            return self.get_container_names()
        else:
            return self._containers

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
        """Gets a container from the HDF5 file based on the arguments.

        Args:
            item (str): The name of the container to get.

        Returns:
            The container requested.
        """
        return self.get_container(item)

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
    def construct(self, obj=None, update=None, open_=False, **kwargs):
        """Constructs this object.

        Args:
            obj: An object to build this object from. It can be the path to the file or a File object.
            update (bool): Determines if this object should constantly open the file for updating attributes.
            open_ (bool): Determines if this object will remain open after construction.
            **kwargs: The keyword arguments for the open method.

        Returns:
            This object.
        """
        if obj is not None:
            self._set_path(obj)

        if update is not None:
            self.is_updating = update

        if self.path.is_file():
            self.open(**kwargs)
            if not open_:
                self.close()
        else:
            self.create_file(open_=open_)

        self._construct_attributes()

        return self

    @dispatch(object)
    def _set_path(self, obj):
        if isinstance(obj, HDF5Object):
            self.path = obj.path

    @dispatch((str, pathlib.Path))
    def _set_path(self, obj):
        """Constructs the path attribute of this object.

        Args:
            obj: The path to the file to build this object around.
        """
        self.path = obj

    @dispatch(h5py.File)
    def _set_path(self, obj):
        """Constructs the path attribute of this object.

        Args:
            obj (obj:`File`): A HDF5 file to build this object around.
        """
        if obj:
            self.path = obj.filename
        else:
            raise ValueError("The supplied HDF5 File must be open.")

    def _construct_attributes(self):
        self.attributes = self.attribute_type(name="/", file=self)

    # File Creation/Construction
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
        """Constructs the file with file attributes and containers.

        Args:
            attr (dict, optional): File attributes to set when the file is created.
            data (dict, optional): Datasets to assign when the file is created.
        """
        with self._temp_open():
            self.construct_file_attributes(**attr)
            self.construct_file_datasets(**data)

    def construct_file_attributes(self, **kwargs):
        """Sets the file attributes based on the default and given attributes.

        Args:
            **kwargs: File attributes to set when the file is created.
        """
        a_kwargs = self.default_file_attributes.copy()
        a_kwargs.update(kwargs)
        self.attributes.update_attributes(**a_kwargs)

    def construct_file_datasets(self, **kwargs):
        """Assigns the file's datasets based on the default and given datasetss.

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
                self.load_containers()
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

    # Containers
    # Names
    def get_container_names(self):
        """Gets the container names from the HDF5 file.

        Returns:
            set: All the container names in the HDF5 file.
        """
        with self._temp_open():
            for name, value in self.h5_fobj.items():
                self._containers.update((name,))
                if isinstance(value, h5py.Group):
                    container = self.group_type(value, self)
                elif isinstance(value, h5py.Dataset):
                    container = self.dataset_type(value, self)
                setattr(self, "__c_" + name, container)

        return self._containers

    def list_container_names(self):
        """Gets the container names from the HDF5 file.

        Returns:
            list: All the container names in the HDF5 file.
        """
        return list(self.get_container_names())

    # Getters/Setters
    def get_container(self, name):
        """Gets a container from the HDF5 file.

        Args:
            name (str): The name of the container to get.

        Returns:
            The container requested.
        """
        __name = "__c_" + name

        if __name not in self._containers or self.is_updating:
            try:
                with self._temp_open():
                    h_obj = self.h5_fobj[name]
                    if isinstance(h_obj, h5py.Group):
                        container = self.group_type(h_obj, self)
                    elif isinstance(h_obj, h5py.Dataset):
                        container = self.dataset_type(h_obj, self)
                    setattr(self, __name, container)
            except Exception as e:
                warn("Could not update containers due to error: " + str(e), stacklevel=2)

        return getattr(self, __name)

    def get_containers(self):
        """Gets all the containers from the HDF5 file.

        Returns:
            dict: The containers.
        """
        with self._temp_open():
            containers = {}
            for name, value in self.h5_fobj.items():
                self._containers.update((name,))
                if isinstance(value, h5py.Group):
                    container = self.group_type(value, self)
                elif isinstance(value, h5py.Dataset):
                    container = self.dataset_type(value, self)
                containers[name] = container
                setattr(self, "__c_" + name, container)

        return containers

    def del_container(self, name):
        """Deletes a container from the HDF5 file.

        Args:
            name (str): The name of the container to delete.
        """
        __name = "__c_" + name

        try:
            with self._temp_open():
                del self.h5_fobj[name]
                delattr(self, __name)
        except Exception as e:
            warn("Could not update containers due to error: " + str(e), stacklevel=2)

    def clear_containers(self):
        """Clears set tracking the containers and the local object containers."""
        for name in self._containers:
            self.__delattr__("__c_" + name)
        self._containers.clear()

    def load_containers(self):
        """Loads all the containers from the HDF5 file in this object."""
        self.get_container_names()

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
        __name = "__c_" + name

        try:
            with self._temp_open():
                if name in self._containers:
                    self.h5_fobj[name].resize(data.shape)
                    self.h5_fobj[name][...] = data
                else:
                    d_kwargs = self.default_dataset_kwargs.copy()
                    d_kwargs.update(kwargs)
                    d_kwargs["data"] = data

                    self.h5_fobj.require_container(name, **d_kwargs)
                    self._containers.update((name,))
                    setattr(self, __name, self.dataset_type(self.h5_fobj[name], self))
        except Exception as e:
            warn("Could not set dataset due to error: " + str(e), stacklevel=2)

        return getattr(self, __name)

    def update_datasets(self, **items):
        """Updates the datasets based on the dictionary update scheme.

        Args:
            **items: The keyword arguments which are the attributes an their values.
        """
        with self._temp_open():
            for name, kwargs in items.items():
                __name = "__c_" + name
                try:
                    if name in self._containers:
                        self.h5_fobj[name].resize(kwargs["data"].shape)
                        self.h5_fobj[name][...] = kwargs["data"]
                    else:
                        self._containers.update((name,))
                        d_kwargs = self.default_dataset_kwargs.copy()
                        d_kwargs.update(kwargs)

                        self.h5_fobj.require_container(name, **d_kwargs)
                        self._containers.update((name,))
                        setattr(self, __name, self.dataset_type(self.h5_fobj[name], self))
                except Exception as e:
                    warn("Could not set containers due to error: " + str(e), stacklevel=2)

    def append_to_dataset(self, name, data, axis=0):
        """Append data to the dataset along a specified axis.

        Args:
            name (str): The name of the dataset to append the data to.
            data: The data to append.
            axis (int): The axis to append the data along.
        """
        self.get_container(name).append(data, axis)

    #  Mapping
    def items(self):
        """All containers as a list of items, keys and values.

        Returns:
            list: All keys and values.
        """
        result = []
        for key in self.container_names:
            result.append((key, self.get_container(key)))
        return result

    def keys(self):
        """All container names as a list of keys.

        Returns:
            list: All keys.
        """
        return list(self.container_names)

    def pop(self, name):
        """Gets a container then deletes it in the HDF5 file.

        Args:
            name (str): The name of the container to pop.

        Returns:
            The container requested.
        """
        value = self.get_container(name)[...]
        self.del_container(name)
        return value
