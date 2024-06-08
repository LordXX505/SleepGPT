import argparse
import re
import sys

import numpy as np
import os
import glob

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

    file = np.load(os.path.join(args.output_dir, 'ISRUC_S1_split_k_10.npy'), allow_pickle=True).item()
    for i in range(10):
        train_name = file[f'train_{i}']['names']
        test_name = file[f'test_{i}']['names']
        val_name = file[f'val_{i}']['names']

        train_real_names = []
        for tn in train_name:
            base_name = os.path.basename(tn)
            subject_nums = base_name
            train_real_names.append(subject_nums)
        test_real_names = []
        for tn in test_name:
            base_name = os.path.basename(tn)
            subject_nums = base_name
            test_real_names.append(subject_nums)
        val_real_names = []
        for tn in val_name:
            base_name = os.path.basename(tn)
            subject_nums = base_name
            val_real_names.append(subject_nums)
        train_real_names = np.unique(np.array(train_real_names))
        test_real_names = np.unique(np.array(test_real_names))
        val_real_names = np.unique(np.array(val_real_names))
        assert len(train_real_names)+len(test_real_names)+len(val_real_names) == 100
        print(train_real_names, test_real_names, val_real_names)

if __name__ == '__main__':
    main()
