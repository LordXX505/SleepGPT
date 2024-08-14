import numpy as np
import argparse
import os
import glob
import logging
import h5py
Root_path ='/mnt/myvol/data/MGH'
from tqdm import tqdm
def load_hdf5_data(file_path):
    try:
        with h5py.File(file_path, 'r') as hf:
                data_len = hf['signal'].shape[0]
    except Exception as e:
        raise RuntimeError(f"Error reading HDF5 file {file_path}: {e}")
    return data_len
def has_key(h5file_path, key):
    """
    Check if the HDF5 file contains the specified key.

    Parameters:
    h5file_path (str): Path to the HDF5 file.
    key (str): The key to check for.

    Returns:
    bool: True if the key exists, False otherwise.
    """
    try:
        with h5py.File(h5file_path, 'r') as h5file:
            return key in h5file
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return False
def main():
    output_dir = Root_path
    data_path_list = sorted(glob.glob(os.path.join(Root_path, '*', '*')))
    names_anno = []
    nums_anno = []
    names = []
    nums = []
    val_exclude_name = np.load("/root/Sleep/main/preprocessing/bdsp/val_error.npy")
    test_exclude_name = np.load("/root/Sleep/main/preprocessing/bdsp/test_error.npy")
    train_exclude_name = np.load("/root/Sleep/main/preprocessing/bdsp/train_error.npy")
    exclude_name = np.concatenate([train_exclude_name, val_exclude_name, test_exclude_name])
    print(f'shape exclude: {exclude_name.shape}')

    exclude_name = [os.path.split(n)[0] for n in exclude_name]
    for items in tqdm(data_path_list):
        try:
            path = os.path.join(items, 'data.h5')
            if os.path.isfile(path) and items not in exclude_name:
                lens = load_hdf5_data(path)
                if has_key(path, 'stage'):
                    names_anno.append(items)
                    nums_anno.append(lens)
                else:
                    names.append(items)
                    nums.append(lens)
        except Exception as e:
            print(f'Exception: {e}. idx: {items}')
    print(f'anno len : {len(names_anno)}, num len :{len(nums_anno)}')
    names_anno = np.array(names_anno)
    nums_anno = np.array(nums_anno)
    names = np.array(names)
    nums = np.array(nums)

    stage_len = len(names_anno)
    test_len = int (stage_len*0.3)
    val_len = int (stage_len*0.1)
    train_len = stage_len - test_len - val_len
    idx = np.arange(0, stage_len)
    np.random.shuffle(idx)
    train_names = names_anno[idx[:train_len]]
    train_nums = np.ones(train_len, dtype=int)
    val_names = names_anno[idx[train_len:train_len+val_len]]
    val_nums = np.ones(val_len, dtype=int)
    test_names = names_anno[idx[train_len+val_len:]]
    test_nums =  np.ones(test_len, dtype=int)
    assert len(test_names) == test_len
    res = {}
    res['names'] = train_names
    res['nums'] = train_nums
    np.save(os.path.join(output_dir, 'train_test'), arr=res, allow_pickle=True)

    res = {}
    res['names'] = val_names
    res['nums'] = val_nums
    np.save(os.path.join(output_dir, 'val_test'), arr=res, allow_pickle=True)

    res = {}
    res['names'] = test_names
    res['nums'] = test_nums
    np.save(os.path.join(output_dir, 'test_test'), arr=res, allow_pickle=True)

if __name__ == '__main__':
    main()
