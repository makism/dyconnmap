
1. Modify the version parameter in `setup.cfg`.

2. Push in `@master`, create a new tag and push on remote.

3. Publish on pypi with:

```
python3 setup.py sdist bdist_wheel

python3 -m twine upload dist/*                          
```
