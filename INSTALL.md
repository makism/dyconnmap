Prerequisites
=============

The following software is required for `dyfunconn` to work properly:

1. Python 2.7.x or newer (for Python 2)
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

To install `dyfunconn` simple run:
> python setup.py install

If you prefer to install `dyfunconn` locally, instead run:
> python setupy.py install --prefix=$HOME/.local

If you opt for the later option, you will have to set the environmental
variable $PYTONPATH from the terminal as follows:
> export PYTOHNPATH="$HOME/.local/dyunconn/lib/python2.7/site-packages/"


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
