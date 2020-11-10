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

```
anaconda login
anaconda upload dist/*.tar.gz
```

