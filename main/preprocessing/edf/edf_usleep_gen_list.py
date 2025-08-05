import argparse
import re

import numpy as np
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
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
    names = []
    nums = []
    files_dict = {}
    name_index = 0
    for _, sub in enumerate(glob.glob(outputdir)):
        if os.path.isdir(sub):
            base_name = os.path.basename(sub)
            subject_nums = base_name
            if subject_nums not in files_dict:
                files_dict[subject_nums] = [name_index]
            else:
                files_dict[subject_nums].append(name_index)
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
    idx = np.arange(n)
    res = {}
    idx_train = glob.glob(os.path.join(args.output_dir, 'views/fixed_split/train/*'))
    idx_test = glob.glob(os.path.join(args.output_dir, 'views/fixed_split/test/*'))
    idx_val = glob.glob(os.path.join(args.output_dir, 'views/fixed_split/val/*'))
    for i in range(1):

        res[f'train_{i}'] = {}
        res[f'train_{i}']['names'] = []
        res[f'train_{i}']['nums'] = []
        for ds_idx in idx_train:
            ds_idx = os.path.basename(ds_idx)
            for tn_idx in files_dict[ds_idx]:
                res[f'train_{i}']['names'].append(names[tn_idx])
                res[f'train_{i}']['nums'].append(nums[tn_idx])
        res[f'val_{i}'] = {}
        res[f'val_{i}']['names'] = []
        res[f'val_{i}']['nums'] = []
        for ds_idx in idx_val:
            ds_idx = os.path.basename(ds_idx)
            for tn_idx in files_dict[ds_idx]:
                res[f'val_{i}']['names'].append(names[tn_idx])
                res[f'val_{i}']['nums'].append(nums[tn_idx])
        res[f'test_{i}'] = {}
        res[f'test_{i}']['names'] = []
        res[f'test_{i}']['nums'] = []
        for ds_idx in idx_test:
            ds_idx = os.path.basename(ds_idx)
            for tn_idx in files_dict[ds_idx]:
                res[f'test_{i}']['names'].append(names[tn_idx])
                res[f'test_{i}']['nums'].append(nums[tn_idx])
        print(f'train name : {res[f"train_{i}"]["names"]}')
        print(f'test name : {res[f"test_{i}"]["names"]}')
        print(f'val name : {res[f"val_{i}"]["names"]}')
    np.save(os.path.join(args.output_dir, f'edf_downstream_usleep'), arr=res, allow_pickle=True)
    print(f'len: names: {len(names)}')

if __name__ == '__main__':
    main()