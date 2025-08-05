import numpy as np
import argparse
import os
import glob
import logging
import h5py
Root_path ='/mnt/myvol/data/shhs'

def load_hdf5_data(file_path):
    try:
        with h5py.File(file_path, 'r') as hf:
                data_len = hf['signal'].shape[0]
    except Exception as e:
        raise RuntimeError(f"Error reading HDF5 file {file_path}: {e}")
    return data_len
def main():
    output_dir = Root_path
    data_path_list = sorted(glob.glob(os.path.join(Root_path, 'shhs2*')))
    names = []
    nums = []
    for items in data_path_list:
        try:
            path = os.path.join(items, 'data.h5')
            if os.path.isfile(path):
                lens = load_hdf5_data(path)
                names.append(items)
                nums.append(lens)
        except Exception as e:
            print(f'Exception: {e}. idx: {items}')
    res = {}
    res['names'] = names
    res['nums'] = nums
    np.save(os.path.join(output_dir, 'train'), arr=res, allow_pickle=True)
    print(f'len names: {len(names)}, nums: {len(nums)}')
if __name__ == '__main__':
    main()
