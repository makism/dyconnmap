Prerequisites
=============

The following software is required for `dyfunconn` to work properly:

1. Python 3.6.x or newer (for Python 3)
2. NumPy 1.11.x or newer
3. SciPy 0.18.x or newer
4. Matplotlib 1.5.x or newer
5. NetworkX 1.11.x or newer
6. nose (optional)

    This is required for executing the tests.

7. Sphinx (optional)

    This is required for building the documentation from the source code.

Installation
============

##### Using pip
The easiest way to install `dyfunconn` is through `pip`.
From a terminal just type:
> $ pip search dyfunconn

> dyfunconn (1.0.0b3)  - A dynamic functional connectivity module in Python

and to install it:
> $ pip install dyfunconn

You may want to check also the _testing_ repository:
> $ pip install --index-url https://test.pypi.org/simple dyfunconn

##### From source code

First clone the github repository, navigate into the directory and run:
> python setup.py install

If you prefer to install `dyfunconn` locally, instead run:
> python setupy.py install --prefix=$HOME/.local

If you opt for the later option, you will have to set the environmental
variable $PYTHONPATH from the terminal as follows:
> export PYTHONPATH="$HOME/.local/dyunconn/lib/python3.6/site-packages/"


Documentation
=============

Once you have installed `dyfunconn`, navigate into the `tests` directory and run:
> sphinx-apidoc -f -o docs/ dyfunconn/
> cd docs
> make html


Testing
=======

Once you have installed `dyfunconn`, navigate into the `tests` directory and run:
> nosetests -svd .
