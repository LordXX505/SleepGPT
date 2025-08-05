import h5py
import numpy as np
def create_or_append_to_hdf5(file_path, dataset_name, data, maxshape=(None,), dtype='float32'):
    """
    Create a new HDF5 file or append data to an existing dataset.

    Parameters:
    file_path (str): Path to the HDF5 file.
    dataset_name (str): Name of the dataset to create or append to.
    data (numpy array): Data to append.
    maxshape (tuple): Maximum shape of the dataset (None for unlimited along the axis to append).
    dtype (str): Data type of the dataset.
    """
    with h5py.File(file_path, 'a') as h5file:
        if dataset_name in h5file:
            # Dataset exists, append data to it
            dset = h5file[dataset_name]
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
            dset[-data.shape[0]:] = data
        else:
            # Create a new dataset
            dset = h5file.create_dataset(dataset_name, data=data, maxshape=maxshape, dtype=dtype)

