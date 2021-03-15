# Dry build

As a reminder, before doing anything make sure that the tests pass, and the distribution make be built with `python3 setup.py sdist bdist_wheel` using the relevant Python environment.

# Prepare version

1. Modify the version parameter in `setup.py`.

2. Push in `@master`, create a new tag and push on remote.

# Publish

## on PyPi

```
python3 setup.py sdist bdist_wheel

python3 -m twine upload dist/*                          
```

## on Anaconda Cloud

First, activate the Anaconda environment.

```
anaconda login
anaconda upload dist/*.tar.gz
```

