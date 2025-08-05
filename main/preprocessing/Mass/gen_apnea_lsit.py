import pyarrow.parquet as pq
import pyarrow as pa
import h5py
import numpy as np
import argparse
import os
import glob

import torch
from sklearn.model_selection import train_test_split
import re
import pyarrow.feather as feather
import pyarrow.ipc as ipc


def load_hdf5_data(file_path):
    print(f'filepath: {file_path}')
    data_dict = {}
    key = 'apnea'
    try:
        with h5py.File(file_path, 'r') as hf:
            if key in hf:
                # 打印数据集信息
                print(f"Dataset {key} found, shape: {hf[key].shape}")
                data_dict[key] = hf[key][:]
            else:
                print(f"Key '{key}' not found in the HDF5 file.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return data_dict
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        # default="/Volumes/T7/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
                        default="/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_1_Apnea",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="SS2",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    data_dir = args.data_dir
    names = []
    pos_idx = {}
    neg_idx = {}
    for sub in glob.glob(data_dir + '/*'):
        if os.path.isdir(sub):
            names.append(sub)
    for name in names:
        label = load_hdf5_data(os.path.join(name, 'data.h5'))
        print('label')
        pos_idx[name] = []
        neg_idx[name] = []
        for _, item in enumerate(label['apnea']):
            tensor = torch.from_numpy(item)
            if torch.sum(tensor) > 0:
                pos_idx[name].append(_)
            else:
                neg_idx[name].append(_)
    res_pos = {}
    res_neg = {}
    res_test = {}
    test = {}
    N = len(names)
    train_N = int(N*0.7)
    train_name = np.random.choice(names, train_N, replace=False)
    test_name = np.setdiff1d(names, train_name)
    print(f'train_name: {train_name}, test_name: {test_name}')
    cnt_pos = 0
    for k in range(0, 1):
        res_pos[f'train_{k}'] = {}
        res_pos[f'train_{k}']['names'] = []
        res_neg[f'train_{k}'] = {}
        res_neg[f'train_{k}']['names'] = []
        res_test[f'test_{k}'] = {}
        res_test[f'test_{k}']['names'] = []
        for tn in train_name[5:]:
            print(f'name: {tn}, pos lable: {pos_idx[tn]}')
            for pidx in pos_idx[tn]:
                res_pos[f'train_{k}']['names'].append(os.path.join(tn, f'index={pidx}_data.h5'))
                cnt_pos += 1
            for nidx in neg_idx[tn]:
                res_neg[f'train_{k}']['names'].append(os.path.join(tn, f'index={nidx}_data.h5'))
        for testn in test_name:
            for pidx in pos_idx[testn]:
                res_test[f'test_{k}']['names'].append(os.path.join(testn, f'index={pidx}_data.h5'))
            for nidx in neg_idx[testn]:
                res_test[f'test_{k}']['names'].append(os.path.join(testn, f'index={nidx}_data.h5'))
    file_name = data_dir
    print(f'cnt: {cnt_pos}')
    np.save(f'{file_name}/pos', arr=res_pos, allow_pickle=True)
    np.save(f'{file_name}/neg', arr=res_neg, allow_pickle=True)
    np.save(f'{file_name}/test', arr=res_test, allow_pickle=True)

if __name__ == '__main__':
    main()

