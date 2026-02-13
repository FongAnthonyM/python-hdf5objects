Installation
============

PyPI (pip) is the recomended way to install Hdf5objects, but GitHub can also be used. If you want to run the examples
and Jupyter tutorials included in this repository, you should clone and install from GitHub.


PyPI
----
You can install hdf5objects using pip:

.. code-block:: bash

   pip install hdf5objects


GitHub
------

Install the latest code from the main branch without cloning:

.. code-block:: bash

   pip install "git+https://github.com/AnthonyTechnologies/python-hdf5objects.git@main"


GitHub Clone
------------

Installing a github clone can be useful for either exploring the examples and tutorials and/or contributing
hdf5objects.

For only exlporing examples and tutorials:

.. code-block:: bash

   git clone https://github.com/AnthonyTechnologies/python-hdf5objects.git
   cd python-hdf5objects
   pip install .[jupyter]

For contributing/developing hdf5objects:

.. code-block:: bash

   git clone https://github.com/AnthonyTechnologies/python-hdf5objects.git
   cd python-hdf5objects
   pip install -e .[dev]
