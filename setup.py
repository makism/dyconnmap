#!/usr/bin/env python

# based on:
# https://github.com/marcindulak/python-mycli/blob/master/setup.py#L34

import os
from distutils.core import setup

name = "dyfunconn"
rootdir = os.path.abspath(os.path.dirname(__file__))

packages = []
for dirname, dirnames, filenames in os.walk(name):
    if '__init__.py' in filenames:
        packages.append(dirname.replace('/', '.'))

data_files = []
for extra_dirs in ("docs", "examples", "tests"):
    for dirname, dirnames, filenames in os.walk(extra_dirs):
        fileslist = []
        for filename in filenames:
            fullname = os.path.join(dirname, filename)
            fileslist.append(fullname)
        data_files.append(('share/' + name + '/' + dirname, fileslist))

setup(name='dyfunconn',
      version='1.0',
      description='dynamic functional connectivity',
      author='Avraam Marimpis, Stavros Dimitriadis',
      author_email='Avraam.Marimpis@gmail.com, STIDimitriadis@gmail.com',
      license='',
      keywords='',
      url='',
      packages=packages,
      package_dir={'dyfunconn': 'dyfunconn'},
      data_files=data_files,
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: New BSD License',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering'
                   ],
      )
