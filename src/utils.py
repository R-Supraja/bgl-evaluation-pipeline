import json
from datetime import datetime
from typing import Any, Optional

import h5py
import numpy as np


def get_group(
    h5file: h5py.File,
    key: str,
    verbose: bool = True,
) -> Optional[h5py.Group]:
    """Return an HDF5 group for the given key, or None if it is missing or invalid."""
    obj = h5file.get(key)
    if not isinstance(obj, h5py.Group):
        if verbose:
            print(f"{key} is not a Group (type={type(obj)}), skipping.")
        return None
    return obj


def get_dataset(
    group: h5py.Group,
    dataset_name: str,
    verbose: bool = True,
) -> Optional[h5py.Dataset]:
    """Return a dataset from a group, or None if the dataset does not exist."""
    ds = group.get(dataset_name)
    if not isinstance(ds, h5py.Dataset):
        if verbose:
            print(f"{group.name} has no valid dataset '{dataset_name}', skipping.")
        return None
    return ds


def load_dataset_array(
    group: h5py.Group,
    dataset_name: str,
    verbose: bool = True,
    dtype: Optional[np.dtype] = None,
) -> Optional[np.ndarray]:
    """Load a dataset into a NumPy array and optionally cast it to a target dtype."""

    ds = get_dataset(group, dataset_name, verbose=verbose)
    if ds is None:
        return None

    arr = ds[:]
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr
