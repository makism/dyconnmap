Prerequisites
=============

The following software is required for `dyconnmap` to work properly:

1. Python 3.6+
2. NumPy
3. SciPy
4. Matplotlib
5. NetworkX
6. Brain Connictivity Toolbox for Python (BctPy)
7. nose (optional)

    This is required for executing the tests.

8. Sphinx (optional)

    This is required for building the documentation from the source code.

Consult the accompanying `requirement.txt` file for specific versions.

Installation
============

##### Using pip
The easiest way to install `dyconnmap` is through `pip`.
From a terminal just type:
> $ pip search dyconnmap

> dyconnmap (1.0.0)  - A dynamic connectome mapping module.

and to install it:
> $ pip install dyconnmap

You may want to check also the _testing_ repository:
> $ pip install --index-url https://test.pypi.org/simple dyconnmap


##### in an Anaconda distribution
Activate your installation and give:
> $ pip install --extra-index-url https://pypi.anaconda.org/makism/simple dyconnmap

##### From source code

First clone the github repository, navigate into the directory and run:
> $ python setup.py install

If you prefer to install `dyconnmap` locally, instead run:
> $ python setupy.py install --prefix=$HOME/.local

If you opt for the later option, you will have to set the environmental
variable $PYTHONPATH from the terminal as follows:
> $ export PYTHONPATH="$HOME/.local/dyconnmap/lib/python3.6/site-packages/"


Documentation
=============

Once you have installed `dyconnmap`, navigate into the source directory and run:
> $ sphinx-apidoc -f -o docs/ dyconnmap/

> $ cd docs

> $ make html


Testing
=======

Once you have installed `dyconnmap`, navigate into the `tests` directory and run:
> $ nosetests -svd .
