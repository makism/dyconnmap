# About

Currently, we use _pytest_ for the tests. Each test contains a some ground truth results (organized under `groundtruth/`), and a set of data that each function/method will use (in `sample_data/`).

# Run

You may run the tests using:
```
$ pytest .
```

# Advanced Run

You can also use our provided tox setup to test across Python 3.6, 3.8 and 3.9.
It is recommended to first activate an Anaconda environment, then from the project's parent directory run:
```
$ tox
```
