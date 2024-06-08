import os
import torch
import numpy as np
import argparse
import re

import numpy as np
import os
import glob
from sklearn.model_selection import KFold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Volumes/T7/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
                        # default="/home/cuizaixu_lab/huangweixuan/data/data/sleep-cassette",
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
    files_dict = {}
    name_index = 0
    for _, sub in enumerate(glob.glob(outputdir)):
        if os.path.isdir(sub):
            base_name = os.path.basename(sub)
            subject_nums = base_name[3:5]
            if int(subject_nums) not in files_dict:
                files_dict[int(subject_nums)] = [name_index]
            else:
                files_dict[int(subject_nums)].append(name_index)
            names.append(sub)
            name_index += 1
    print(f'The sum of subjects is : {len(files_dict)}')
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

    n = len(files_dict)
    res = {}
    train_subs = [48, 72, 24, 30, 34, 50, 38, 15, 60, 12]
    train_segs = [3937, 2161, 3448, 1783, 3083, 2429, 3647, 2714, 3392, 2029]

    """
    val_subs : list of subjects for validation; this list depends on the seed set during the preprocessing

    val_segs : list of number of epochs of EEG data present in each subject of the val_subs list
    """

    val_subs = [23, 26, 37, 44, 49, 51, 54, 59, 73, 82]
    val_segs = [2633, 2577, 2427, 2287, 2141, 2041, 2864, 3071, 4985, 3070]

    edf_permutation = np.array(train_subs+val_subs)  # to have the same results as in the paper

    idxs = np.arange(len(val_subs+train_subs))
    kfold = KFold(n_splits=5, shuffle=False)

    for i, (train_idx, val_idx) in enumerate(kfold.split(idxs)):
        res[f'train_{i}'] = {}
        res[f'train_{i}']['names'] = []
        res[f'train_{i}']['nums'] = []
        for ds_idx in train_idx:
            for tn_idx in files_dict[edf_permutation[ds_idx]]:
                res[f'train_{i}']['names'].append(names[tn_idx])
                res[f'train_{i}']['nums'].append(nums[tn_idx])
        res[f'val_{i}'] = {}
        res[f'val_{i}']['names'] = []
        res[f'val_{i}']['nums'] = []
        for ds_idx in val_idx:
            for tn_idx in files_dict[edf_permutation[ds_idx]]:
                res[f'val_{i}']['names'].append(names[tn_idx])
                res[f'val_{i}']['nums'].append(nums[tn_idx])
        res[f'test_{i}'] = {}
        res[f'test_{i}']['names'] = []
        res[f'test_{i}']['nums'] = []
        for ds_idx in val_idx:
            for tn_idx in files_dict[edf_permutation[ds_idx]]:
                res[f'test_{i}']['names'].append(names[tn_idx])
                res[f'test_{i}']['nums'].append(nums[tn_idx])
        print(f'train name : {res[f"train_{i}"]["names"]}')
        print(f'test name : {res[f"test_{i}"]["names"]}')
        print(f'val name : {res[f"val_{i}"]["names"]}')
    np.save(os.path.join(args.output_dir, f'edf_downstream_mul'), arr=res, allow_pickle=True)
    print(f'len: names: {len(names)}')

if __name__ == '__main__':
    main()

