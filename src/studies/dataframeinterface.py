#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" dataframeinterface.py
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

# Downloaded Libraries #
from baseobjects import BaseObject

# Local Libraries #


# Definitions #
# Classes #
class DataFrameInterface(BaseObject):
    # Magic Methods #
    # Construction/Destruction
    def __init__(self, init=False):
        self.editable_method = self.default_editable_method

    # Container Methods
    def __len__(self):
        return self.get_length()

    def __getitem__(self, item):
        return self.get_item(item)

    # Instance Methods #
    # Constructors/Destructors
    def editable_copy(self, **kwargs):
        return self.editable_method(**kwargs)

    # Getters
    @abstractmethod
    def get_length(self):
        pass

    @abstractmethod
    def get_item(self, item):
        pass

    # Editable Copy Methods
    def default_editable_method(self, **kwargs):
        raise NotImplemented

    # Setters
    def set_editable_method(self, obj):
        self.editable_method = obj

    # Shape
    @abstractmethod
    def validate_shape(self):
        pass

    @abstractmethod
    def change_size(self, shape=None, **kwargs):
        pass

    # Get Frame within by Index
    @abstractmethod
    def get_index(self, indices, reverse=False, frame=True):
        pass

