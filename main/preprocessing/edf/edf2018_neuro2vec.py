import argparse
import re

import numpy as np
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/cuizaixu_lab/huangweixuan/data/data/sleep-cassette",
                        # default="/Volumes/T7 Shield/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
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
    names = []
    nums = []
    for sub in glob.glob(outputdir):
        if os.path.isdir(sub):
            names.append(sub)
    for name in names:
        print(f'------{name}-------')
        tmp = 0
        for item in os.listdir(name):
            if os.path.isfile(os.path.join(str(name), str(item))):
                tmp += 1
        print(f'num: {tmp}')
        nums.append(tmp)
    pretrain_res = {}
    down_stream_res = {}
    nums = np.array(nums)
    names = np.array(names)
    n = len(names)
    for random_seed in range(10):
        idx = np.arange(n)
        np.random.seed(random_seed)
        np.random.shuffle(idx)
        test_nums = int(n*0.2)
        valid_nums = int((n - test_nums)*0.2)
        test_idx = idx[:test_nums]
        valid_idx = idx[test_nums:test_nums+valid_nums]
        train_idx = idx[test_nums+valid_nums:]
        pretrain_res[f'train_{random_seed}'] = []
        pretrain_res[f'val_{random_seed}'] = []
        for tn in names[train_idx]:
            for arr in os.listdir(tn):
                pretrain_res[f'train_{random_seed}'].append(os.path.join(tn, str(arr, encoding='utf-8')))
        for tn in names[valid_idx]:
            for arr in os.listdir(tn):
                pretrain_res[f'val_{random_seed}'].append(os.path.join(tn, str(arr, encoding='utf-8')))
        down_stream_res[f'train_{random_seed}'] = {}
        down_stream_res[f'train_{random_seed}']['names'] = names[train_idx]
        down_stream_res[f'train_{random_seed}']['nums'] = nums[train_idx]
        down_stream_res[f'val_{random_seed}'] = {}
        down_stream_res[f'val_{random_seed}']['names'] = names[valid_idx]
        down_stream_res[f'val_{random_seed}']['nums'] = nums[valid_idx]
        down_stream_res[f'test_{random_seed}'] = {}
        down_stream_res[f'test_{random_seed}']['names'] = names[test_idx]
        down_stream_res[f'test_{random_seed}']['nums'] = nums[test_idx]
    np.save(os.path.join(args.output_dir, f'edf_pretrain_n2v'), arr=pretrain_res, allow_pickle=True)
    np.save(os.path.join(args.output_dir, f'edf_downstream_n2v'), arr=down_stream_res, allow_pickle=True)
    # print(f'len: names: {len(names)}')




if __name__ == '__main__':
    main()
