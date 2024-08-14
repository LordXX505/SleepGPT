import numpy as np
import h5py
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool, Manager

def calculate_mean(file_path, dataset_name='signal', chunk_size=1000):
    """
    Calculate the mean of a dataset in an HDF5 file.

    Parameters:
    file_path (str): Path to the HDF5 file.
    dataset_name (str): Name of the dataset.
    chunk_size (int): Number of rows to read at a time.

    Returns:
    tuple: Total sum, total variance, and total count of the dataset.
    """
    with h5py.File(file_path, 'r') as h5file:
        dataset = h5file[dataset_name]
        total_sum = np.zeros(9)
        total_var = np.zeros(9)
        total_count = 0

        for i in range(0, dataset.shape[0], chunk_size):
            try:
                ed = min(dataset.shape[0], i + chunk_size)
                chunk = dataset[i:ed, :, :]
                total_sum += np.mean(np.mean(chunk, axis=-1), axis=0)
                total_var += np.mean(np.mean((chunk - np.mean(chunk, axis=-1, keepdims=True))**2, axis=-1), axis=0)
                total_count += 1
            except OSError as e:
                print(f'{file_path} is wrong: {e}')
                continue

    return total_sum/total_count, total_var/total_count, total_count

def process_file(file_path):
    mean_value, var_value, count = calculate_mean(file_path)
    return mean_value, var_value, count

def main():
    Root_path = '/mnt/myvol/data/MGH'
    data_path_list = np.load(os.path.join(Root_path, 'pre_train.npy'), allow_pickle=True).item()['names']

    manager = Manager()
    all_value = manager.list(np.zeros(9))
    all_var = manager.list(np.zeros(9))
    all_count = manager.Value('i', 0)

    num_processes = 64  # 指定进程数量
    name = [os.path.join(items, 'data.h5') for items in data_path_list if os.path.isfile(os.path.join(items, 'data.h5'))]
    process_file(name[0])
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_file, [os.path.join(items, 'data.h5') for items in data_path_list if os.path.isfile(os.path.join(items, 'data.h5'))]), total=len(data_path_list)))
    for mean_value, var_value, count in results:
        all_value += mean_value
        all_var += var_value
        all_count.value += count

    np.save(os.path.join(Root_path, 'mean_var'), arr={'mean': list(all_value), "count": all_count.value, "var": list(all_var)})

if __name__ == '__main__':
    main()


