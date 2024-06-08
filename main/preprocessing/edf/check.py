import argparse
import re
import sys

import numpy as np
import os
import glob

def check():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        # default="/home/cuizaixu_lab/huangweixuan/data/data/sleep-cassette",
                        default="/Volumes/T7/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
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
    pretrain_file = np.load(os.path.join(args.output_dir, "edf_pretrain.npy"), allow_pickle=True).item()
    downstream_file = np.load(os.path.join(args.output_dir, "edf_downstream_9_5_5.npy"), allow_pickle=True).item()
    for random_seed in range(5):
        pre_names = []
        for item in pretrain_file[f'pretrain_{random_seed}']:
            base_name = os.path.split(item)[0]
            if base_name == '/home/cuizaixu_lab/huangweixuan/data/data/sleep-cassette/processed/processed':
                sys.exit(0)
            pre_names.append(base_name)
        pre_names = np.array(pre_names)
        print(f'before unique {pre_names.shape}')
        pre_names = np.unique(pre_names)
        print(f'after unique {pre_names.shape}')
        down_stream_train_res = downstream_file[f'train_{random_seed}']['names']
        down_stream_test_res = downstream_file[f'test_{random_seed}']['names']
        print(f'down_stream_train_res:{len(down_stream_train_res)}')
        print(f'down_stream_test_res:{len(down_stream_test_res)}')

        for testn in down_stream_test_res:
            assert testn not in down_stream_train_res
            assert testn not in pre_names
        for trn in down_stream_train_res:
            assert trn not in pre_names
            assert trn not in down_stream_test_res

        all_name = np.concatenate([down_stream_train_res, down_stream_test_res, pre_names])
        all_name = np.unique(all_name)
        print(all_name)
        assert all_name.shape[0] == 153, f'randomseed: {random_seed}, shape: {all_name.shape[0]}'
        print(len(all_name))
if __name__ == '__main__':
    check()