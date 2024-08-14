import numpy as np
import h5py
import glob
import os
from tqdm import tqdm
def calculate_mean(file_path, dataset_name='signal', chunk_size=1000):
    """
    Calculate the mean of a dataset in an HDF5 file.

    Parameters:
    file_path (str): Path to the HDF5 file.
    dataset_name (str): Name of the dataset.
    chunk_size (int): Number of rows to read at a time.

    Returns:
    float: Mean of the dataset.
    """
    with h5py.File(file_path, 'r') as h5file:
        dataset = h5file[dataset_name]
        total_sum = np.zeros(7)
        total_val = np.zeros(7)
        total_count = 0

        for i in range(0, dataset.shape[0], chunk_size):
            try:
                chunk = dataset[i:i + chunk_size]
                total_sum += np.sum(np.mean(chunk, axis=-1), axis=0)
                total_val +=  np.sum(np.var(chunk, axis=-1), axis=0)
                total_count += chunk.shape[0]
            except OSError as e:
                print(f'{file_path} is wrong')
                continue

    return total_sum, total_count

Root_path ='/mnt/myvol/data/shhs'
data_path_list = np.load(os.path.join(Root_path, 'train11.npz'), allow_pickle=True)['names']
all_value = np.zeros(7)
all_val = np.zeros(7)
all_count = 0
#'/mnt/myvol/data/MGH/sub-S0001117111225/ses-1'
for items in tqdm(data_path_list):
    base_name = os.path.basename(items)
    data_path = os.path.join(Root_path, base_name, 'data.h5')
    if os.path.isfile(data_path):
        mean_value, count = calculate_mean(data_path)
        all_value += mean_value
        all_count += count
np.save(os.path.join(Root_path, f'mean_var_shhs1'),
        arr={'mean': all_value, "count": all_count, "var": all_val})
