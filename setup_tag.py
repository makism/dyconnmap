#!/usr/bin/env python

# based on:
# * https://github.com/marcindulak/python-mycli/blob/master/setup.py#L34
# * https://gist.github.com/jpmens/6248478


import os
from setuptools import setup


def git_describe():
    from subprocess import Popen, PIPE
    try:
        p = Popen(['git', 'describe'], stdout=PIPE, stderr=PIPE)
        p.stderr.close()
        line = p.stdout.readlines()[0]

        return line.decode().strip()

    except:
        return None


name = "dyconnmap"
rootdir = os.path.abspath(os.path.dirname(__file__))

packages = []
for dirname, dirnames, filenames in os.walk(name):
    if "__init__.py" in filenames:
        packages.append(dirname.replace("/", "."))

data_files = []
for extra_dirs in ("docs", "examples", "tests"):
    for dirname, dirnames, filenames in os.walk(extra_dirs):
        fileslist = []
        for filename in filenames:
            fullname = os.path.join(dirname, filename)
            fileslist.append(fullname)
        data_files.append(("share/" + name + "/" + dirname, fileslist))

setup(
    name="dyconnmap",
    version=git_describe(),
    description="A dynamic connectome mapping module in Python",
    author="Avraam Marimpis, Stavros Dimitriadis",
    author_email="Avraam.Marimpis@gmail.com, STIDimitriadis@gmail.com",
    license="BSD",
    keywords="eeg fMRI meg connectivity graphs neuroimage brain",
    url="https://github.com/makism/dyconnmap",
    python_requires="~=3.6",
    packages=packages,
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "matplotlib",
        "statsmodels",
        "scikit-learn",
        "bctpy",
    ],
    package_dir={"dyconnmap": "dyconnmap"},
    data_files=data_files,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
)
