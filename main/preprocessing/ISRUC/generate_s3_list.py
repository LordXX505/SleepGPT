import argparse
import re

import numpy as np
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Volumes/T7/data/ISRUC_s3/processed",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    os.makedirs(args.data_dir, exist_ok=True)
    args.log_file = os.path.join(args.data_dir, args.log_file)
    outputdir = args.data_dir + '/*'

    names = []
    nums = []
    name_index = 0
    for _, sub in enumerate(glob.glob(outputdir)):
        if os.path.isdir(sub):
            names.append(sub)
            name_index += 1
    print(f'The sum of subjects is : {len(names)}')
    for name in names:
        print(f'------{name}-------')
        tmp = 0
        for item in os.listdir(name):
            if os.path.isfile(os.path.join(str(name), str(item))):
                tmp += 1
        print(f'num: {tmp}')
        nums.append(tmp)

    nums = np.array(nums)
    names = np.array(names)
    n = len(names)
    idx = np.arange(n)
    print(f'n: {n}')
    k_split = 1
    res = {}
    for i in range(10):
        st = i * k_split
        ed = (i + 1) * k_split
        if i == 9:
            ed = n
        idx_split = idx[st:ed]
        print(f'st: {st}, ed: {ed}')
        idx_train = np.setdiff1d(idx, idx_split)
        np.random.shuffle(idx_train)
        res[f'train_{i}'] = {}
        res[f'train_{i}']['names'] = []
        res[f'train_{i}']['nums'] = []
        for ds_idx in idx_train:
            res[f'train_{i}']['names'].append(names[ds_idx])
            res[f'train_{i}']['nums'].append(nums[ds_idx])
        res[f'val_{i}'] = {}
        res[f'val_{i}']['names'] = []
        res[f'val_{i}']['nums'] = []
        for ds_idx in idx_split:
            res[f'val_{i}']['names'].append(names[ds_idx])
            res[f'val_{i}']['nums'].append(nums[ds_idx])
        res[f'test_{i}'] = {}
        res[f'test_{i}']['names'] = []
        res[f'test_{i}']['nums'] = []
        for ds_idx in idx_split:
            res[f'test_{i}']['names'].append(names[ds_idx])
            res[f'test_{i}']['nums'].append(nums[ds_idx])
        print(idx, st, ed, idx_train)
        print(f'train name : {res[f"train_{i}"]["names"]}')
        print(f'test name : {res[f"test_{i}"]["names"]}')
        print(f'val name : {res[f"val_{i}"]["names"]}')
    np.save(os.path.join(args.data_dir, f'ISRUC_S3_split_k_10_no_val'), arr=res, allow_pickle=True)
    print(f'len: names: {len(names)}')


if __name__ == '__main__':
    main()