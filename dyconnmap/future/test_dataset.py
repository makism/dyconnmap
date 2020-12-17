import numpy as np

import os.path
import sys

sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/")
sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/dyconnmap/future/")

from .dataset import Dataset, Modality


def test_dataset():
    """Test Dataset's access to properties."""
    rng = np.random.RandomState(0)

    n_subjects = 1
    n_rois = 4
    n_samples = 1000
    fs = 1500.0
    labels = ["a", "b", "c", "d"]

    data = rng.rand(n_subjects, n_rois, n_samples)

    ds = Dataset(data, labels=labels, modality=Modality.Raw, fs=fs)
    ds.comments = "Test"

    assert ds.subjects == n_subjects
    assert ds.samples == n_samples
    assert ds.fs == fs
    assert ds.rois == n_rois
    assert ds.modality == Modality.Raw
    assert set(ds.labels) == set(labels)
    assert ds.comments == "Test"


def test_dataset_add_data():
    """Test adding/appending new data to a Dataset object."""
    rng = np.random.RandomState(0)

    n_subjects = 1
    n_rois = 4
    n_samples = 1000
    fs = 1500.0
    labels = ["a", "b", "c", "d"]

    data = rng.rand(n_subjects, n_rois, n_samples)

    ds = Dataset(data, labels=labels, modality=Modality.Raw, fs=fs)

    new_data = rng.rand(n_rois, n_samples)
    ds += new_data

    assert ds.subjects == 2


def test_dataset_access_data():
    """Test accessing subject data from a Dataset object."""
    rng = np.random.RandomState(0)

    n_subjects = 10
    n_rois = 4
    n_samples = 1000
    fs = 1500.0
    labels = ["a", "b", "c", "d"]

    data = rng.rand(n_subjects, n_rois, n_samples)

    ds = Dataset(data, labels=labels, modality=Modality.Raw, fs=fs)

    subject_data = ds[5]

    np.testing.assert_array_equal(data[5], subject_data)


def test_dataset_write():
    """Test Dataset dumping to the disk."""
    rng = np.random.RandomState(0)

    n_subjects = 2
    n_rois = 4
    n_samples = 1000
    fs = 1500.0
    labels = ["a", "b", "c", "d"]

    data = rng.rand(n_subjects, n_rois, n_samples)

    ds = Dataset(data, labels=labels, modality=Modality.Raw, fs=fs)

    ds.write("/tmp/test_ds_write")

    assert os.path.exists("/tmp/test_ds_write/dataset.json") == True
    assert os.path.exists("/tmp/test_ds_write/data_subject0.csv") == True
    assert os.path.exists("/tmp/test_ds_write/data_subject1.csv") == True
