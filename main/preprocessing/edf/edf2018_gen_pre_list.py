import argparse
import re
import sys

import numpy as np
import os
import glob


def main():
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
    names = []
    nums = []
    files_dict = {}
    name_index = 0
    for _, sub in enumerate(glob.glob(outputdir)):
        if os.path.isdir(sub):
            base_name = os.path.basename(sub)
            subject_nums = base_name[3:5]
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
    pretrain_res = {}
    down_stream_res = {}
    nums = np.array(nums)
    names = np.array(names)
    files_dict_value = list(files_dict.values())
    files_dict_keys = list(files_dict.keys())
    files_dict = list(files_dict.values())
    n = len(files_dict_value)
    print(f'n: {n}')
    test_nums = int(np.ceil(n * 0.05))
    down_stream = int(np.ceil(n * 0.05))
    print(f'test_nums, down_stream: {test_nums, down_stream}')
    res = {}

    for random_seed in range(0, 50000):
        idx = np.arange(n)
        np.random.seed(random_seed)
        np.random.shuffle(idx)
        test_idx = idx[:test_nums]
        down_stream_idx = idx[-down_stream:]
        test_2013_nums = 0
        down_stream_nums = 0
        test_name_res = []
        down_stream_idx_name_res = []
        for i, j in zip(test_idx, down_stream_idx):
            test_name = files_dict_keys[i]
            test_name_res.append(test_name)
            down_stream_idx_name = files_dict_keys[j]
            down_stream_idx_name_res.append(down_stream_idx_name)
            if int(i) <= 20:
                test_2013_nums += 1
            if int(j) <= 20:
                down_stream_nums += 1

        if test_2013_nums >= 3:
            res[random_seed] = [[test_name_res, test_2013_nums],
                                [down_stream_idx_name_res, down_stream_nums]]
    for item in res.items():
        print(f'random: {item[0]}, items: {item[1]}')
    sys.exit(0)
    adapted_random_seed = [52, 92, 114, 139, 154, 218, 226, 277, 312, 375,
                           427, 473, 730, 973, 1011, 1172, 1232, 1312, 1371, 1511,
                           1735, 1769, 1865, 1870, 1929, 2017, 2424, 2667, 2919, 3322]
    for random_seed in range(0, 30):
        idx = np.arange(n)
        np.random.seed(adapted_random_seed[random_seed])
        np.random.shuffle(idx)
        test_idx = idx[:test_nums]
        down_stream_idx = idx[-down_stream:]
        train_idx = idx[test_nums:-down_stream]
        pretrain_res[f'train_{random_seed}'] = []
        pretrain_res[f'test_{random_seed}'] = []
        for t_idx in train_idx:
            for tn_idx in files_dict[t_idx]:
                for arr in os.listdir(names[tn_idx]):
                    pretrain_res[f'train_{random_seed}'].append(os.path.join(names[tn_idx], str(arr, encoding='utf-8')))
        for t_idx in test_idx:
            for tn_idx in files_dict[t_idx]:
                for arr in os.listdir(names[tn_idx]):
                    pretrain_res[f'train_{random_seed}'].append(os.path.join(names[tn_idx], str(arr, encoding='utf-8')))
        down_stream_res[f'train_{random_seed}'] = {}
        down_stream_res[f'train_{random_seed}']['names'] = []
        down_stream_res[f'train_{random_seed}']['nums'] = []
        for ds_idx in down_stream_idx:
            for tn_idx in files_dict[ds_idx]:
                down_stream_res[f'train_{random_seed}']['names'].append(names[tn_idx])
                down_stream_res[f'train_{random_seed}']['nums'].append(nums[tn_idx])

        down_stream_res[f'test_{random_seed}'] = {}
        down_stream_res[f'test_{random_seed}']['names'] = []
        down_stream_res[f'test_{random_seed}']['nums'] = []
        all_test_idx = np.concatenate([test_idx, train_idx])
        for ds_idx in all_test_idx:
            for tn_idx in files_dict[ds_idx]:
                down_stream_res[f'test_{random_seed}']['names'].append(names[tn_idx])
                down_stream_res[f'test_{random_seed}']['nums'].append(nums[tn_idx])

        down_stream_res[f'val_{random_seed}'] = {}
        down_stream_res[f'val_{random_seed}']['names'] = []
        down_stream_res[f'val_{random_seed}']['nums'] = []
        for ds_idx in test_idx:
            for tn_idx in files_dict[ds_idx]:
                down_stream_res[f'val_{random_seed}']['names'].append(names[tn_idx])
                down_stream_res[f'val_{random_seed}']['nums'].append(nums[tn_idx])
    np.save(os.path.join(args.output_dir, f'edf_pretrain'), arr=pretrain_res, allow_pickle=True)
    np.save(os.path.join(args.output_dir, f'edf_downstream_9_5_5'), arr=down_stream_res, allow_pickle=True)
    # print(f'len: names: {len(names)}')


if __name__ == '__main__':
    main()
