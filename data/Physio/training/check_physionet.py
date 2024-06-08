import numpy as np

import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="./",
                        help="File path to the Sleep-EDF dataset.")
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

    file = np.load(os.path.join(args.data_dir, 'physio_downstream_usleep.npy'), allow_pickle=True).item()
    all_test_name = []
    for i in range(1):
        train_name = file[f'train_{i}']['names']
        test_name = file[f'test_{i}']['names']
        val_name = file[f'val_{i}']['names']
        print(np.sort(val_name))
        for vn in val_name:
            if vn in train_name or vn in test_name:
                raise NotImplemented
        for tn in test_name:
            if tn in train_name or tn in val_name:
                raise NotImplemented
        all = np.unique(np.concatenate([train_name, test_name, val_name]))
        # print(np.sort(all), len(all))
        assert len(test_name) == 100
        all_test_name.append(test_name)
    all_test_name = np.concatenate(all_test_name)
    # print(np.sort(all_test_name), len(all_test_name))

def pre():
    names = np.load('Physio.npy', allow_pickle=True)
    print(names)
def check_val_names():
    file = np.load(os.path.join('./', 'split_k_5.npy'), allow_pickle=True).item()
    file2 = np.load(os.path.join('./', 'split_k_5.npy'), allow_pickle=True).item()
    for i in range(2):
        test_name = file[f'test_{i}']['names']
        test_name2 = file2[f'test_{i}']['names']
        test_name = np.sort(test_name)
        test_name2 = np.sort(test_name2)
        print(np.setdiff1d(test_name, test_name2))
if __name__ == '__main__':
    main()
    # pre()
    # check_val_names()