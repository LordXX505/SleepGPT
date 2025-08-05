import numpy as np

import os
import glob
import argparse
import pyarrow as pa
import torch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="./",
                        help="File path to the Sleep-MASS dataset.")
    parser.add_argument("--output_dir", type=str, default="processed",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    args.output_dir = os.path.join(args.data_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, args.log_file)
    outputdir = args.output_dir + '/*'

    file = {}
    for SSNum in range(1, 6):
        file[f'SS{SSNum}'] = np.load(os.path.join(args.data_dir, f'split_k_20_SS{SSNum}.npy'), allow_pickle=True).item()

    all_test_name = []
    for i in range(20):
        res_train_name = []
        res_test_name = []
        res_val_name = []
        for SSNum in range(1, 6):
            train_name = file[f'SS{SSNum}'][f'train_{i}']['names']
            test_name = file[f'SS{SSNum}'][f'test_{i}']['names']
            val_name = file[f'SS{SSNum}'][f'val_{i}']['names']
            res_train_name.append(train_name)
            res_test_name.append(test_name)
            res_val_name.append(val_name)
        res_train_name = np.concatenate(res_train_name)
        res_test_name = np.concatenate(res_test_name)
        res_val_name = np.concatenate(res_val_name)
        for vn in res_val_name:
            if vn in res_train_name or vn in res_test_name:
                raise NotImplemented
        for tn in res_test_name:
            if tn in res_train_name or tn in res_val_name:
                raise NotImplemented
        all = np.unique(np.concatenate([res_train_name, res_test_name, res_val_name]))
        print(np.sort(all), len(all))
        assert len(all) == 200
        all_test_name.append(res_test_name)
    all_test_name = np.concatenate(all_test_name)
    print(np.sort(all_test_name), len(all_test_name))

def test():
    file = {}

    file[f'SS3'] = np.load(f'split_k_20_SS3.npy', allow_pickle=True).item()
    for i in range(20):
        for idx, name in enumerate(file[f'SS3'][f'train_{i}']['names']):
            if name == '/home/cuizaixu_lab/huangweixuan/data/data/MASS_Processed/SS3/01-03-0058':
                print(file[f'SS3'][f'train_{i}']['nums'][idx])

def get_epochs(data):

    try:
        x = np.array(data.as_py())
    except:
        x = np.array(data.to_pylist())
    # rank_zero_info(f'settings: {self.settings}')
            # rank_zero_info(f'max: {np.max(x)}, min: {np.min(x)}')
    x = torch.from_numpy(x).float()
    return {'x': x}
def check_length():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/cuizaixu_lab/huangweixuan/data/data/MASS_Processed/",
                        help="File path to the Sleep-MASS dataset.")
    parser.add_argument("--output_dir", type=str, default="processed",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    for ssnum in range(2, 5):
        orig_path = os.path.join(args.data_dir, f'SS{ssnum}')
        names = glob.glob(orig_path + '/*')
        for name in names:
            items = glob.glob(name + '/*')
            for item in items:
                tables = pa.ipc.RecordBatchFileReader(
                    pa.memory_map(item, "r")
                ).read_all()
                x = get_epochs(tables['x'][0])['x']
                if x.shape[1] != 3000:
                    print(f'path: {item}')


if __name__ == '__main__':
    check_length()