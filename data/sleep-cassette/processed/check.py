import argparse
import re
import sys

import numpy as np
import os
import glob
def check_mul(train, test):
    train_subs = [48, 72, 24, 30, 34, 50, 38, 15, 60, 12]
    train_segs = [3937, 2161, 3448, 1783, 3083, 2429, 3647, 2714, 3392, 2029]

    """
    val_subs : list of subjects for validation; this list depends on the seed set during the preprocessing

    val_segs : list of number of epochs of EEG data present in each subject of the val_subs list
    """

    val_subs = [23, 26, 37, 44, 49, 51, 54, 59, 73, 82]
    val_segs = [2633, 2577, 2427, 2287, 2141, 2041, 2864, 3071, 4985, 3070]

    edf_permutation = np.array(train_subs + val_subs)  # to have the same results as in the paper
    for train_items, test_items in zip(train, test):
        base_name = os.path.basename(train_items)
        number = re.findall(r'\d+', base_name)[0][1:3]
        assert int(number) in edf_permutation
        base_name = os.path.basename(test_items)
        number = re.findall(r'\d+', base_name)[0][1:3]
        assert int(number) in edf_permutation


def check_usleep(train, val, test):
    print(train, val, test)
    assert len(test) == 23
def check(mode='mul'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="./",
                        # default="/home/cuizaixu_lab/huangweixuan/data/data/sleep-cassette",
                        # default="/Volumes/T7 Shield/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    args.output_dir = os.path.join(args.data_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, args.log_file)
    outputdir = args.output_dir + '/*'
    # pretrain_file = np.load(os.path.join(args.output_dir, "edf_pretrain_n2v.npy"), allow_pickle=True).item()
    # downstream_file = np.load(os.path.join(args.output_dir, "edf_downstream_n2v.npy"), allow_pickle=True).item()
    downstream_file = np.load(os.path.join(args.data_dir, f"edf_downstream_{mode}.npy"), allow_pickle=True).item()
    for random_seed in range(0, 1):
        print(f'randomseed : {random_seed}')
        down_stream_train_res = downstream_file[f'train_{random_seed}']['names']
        down_stream_test_res = downstream_file[f'test_{random_seed}']['names']
        down_stream_val_res = downstream_file[f'val_{random_seed}']['names']
        print(f'down_stream_train_res:{len(down_stream_train_res)}')
        print(f'down_stream_val_res:{len(down_stream_val_res)}')
        print(f'down_stream_test_res:{len(down_stream_test_res)}')

        test_2013_nums = 0
        if mode == 'mul':
            check_mul(down_stream_train_res, down_stream_test_res)
        elif mode == 'usleep':
            check_usleep(down_stream_train_res, down_stream_val_res, down_stream_test_res)
        print('test')
        for testn in down_stream_test_res:
            base_name = os.path.basename(testn)
            number = re.findall(r'\d+', base_name)[0][1:3]
            print(f'number: {number}')
            if int(number) <= 19:
                test_2013_nums += 1
            assert testn not in down_stream_train_res
        valn_2013_nums = 0
        for valn in down_stream_val_res:
            base_name = os.path.basename(valn)
            number = re.findall(r'\d+', base_name)[0][1:3]
            print(f'number: {number}')
            if int(number) <= 19:
                valn_2013_nums += 1
            assert valn not in down_stream_train_res
        train_2013_nums = 0
        print('train')
        for trn in down_stream_train_res:
            assert trn not in down_stream_test_res
            assert trn not in down_stream_val_res

            base_name = os.path.basename(trn)
            number = re.findall(r'\d+', base_name)[0][1:3]
            print(f'number: {number}')
            if int(number) <= 19:
                train_2013_nums += 1
        print(train_2013_nums, test_2013_nums, valn_2013_nums)

        all_name = np.concatenate([down_stream_val_res, down_stream_test_res, down_stream_train_res])
        all_name = np.unique(all_name)
        # print(all_name)
        # assert all_name.shape[0] == 153, f'randomseed: {random_seed}, shape: {all_name.shape[0]}'
        # print(len(all_name))

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

    file = np.load(os.path.join(args.data_dir, 'm_new_split_k_20.npy'), allow_pickle=True).item()
    for i in range(20):
        train_name = file[f'train_{i}']['names']
        test_name = file[f'test_{i}']['names']
        val_name = file[f'val_{i}']['names']
        for vn in val_name:
            if vn in train_name:
                    raise NotImplemented
        # for vn in val_name:
        #     if vn in train_name or vn in test_name:
        #         raise NotImplemented
        # for tn in test_name:
        #     if tn in train_name or tn in val_name:
        #         raise NotImplemented
        train_real_names = []
        for tn in train_name:
            base_name = os.path.basename(tn)
            subject_nums = base_name[3:5]
            train_real_names.append(subject_nums)
        test_real_names = []
        for tn in test_name:
            base_name = os.path.basename(tn)
            subject_nums = base_name[3:5]
            test_real_names.append(subject_nums)
        val_real_names = []
        for tn in val_name:
            base_name = os.path.basename(tn)
            subject_nums = base_name[3:5]
            val_real_names.append(subject_nums)
        train_real_names = np.unique(np.array(train_real_names))
        test_real_names = np.unique(np.array(test_real_names))
        val_real_names = np.unique(np.array(val_real_names))
        print(f'test_real_names: {test_real_names}')
        assert train_real_names.shape[0] == 15
        assert test_real_names.shape[0] == 1 or test_real_names.shape[0] == 1, f'{test_real_names.shape[0]}'
        assert val_real_names.shape[0] == 1 or test_real_names.shape[0] == 1

def check_val_test():
    file = np.load(os.path.join('./', 'm_new_split_k_20_no_val.npy'), allow_pickle=True).item()
    file2 = np.load(os.path.join('./', 'm_new_split_k_20.npy'), allow_pickle=True).item()
    for i in range(0, 10):
        train_name = file[f'train_{i}']['names']
        test_name = file[f'test_{i}']['names']
        val_name = file[f'val_{i}']['names']
        train_name2 = file2[f'train_{i}']['names']
        test_name2 = file2[f'test_{i}']['names']
        val_name2 = file2[f'val_{i}']['names']
        assert (test_name == test_name2)
        assert val_name == test_name

if __name__ == '__main__':
    # main()
    check(mode='usleep')
    # check_val_test()