import argparse
import re

import numpy as np
import os
import glob


def get_aug_name(names, aug_dir):
    res_name = []
    for name in names:
        name_list = name.split('/')
        name_list[6] = aug_dir
        res_name.append(os.path.join(*name_list))
    return res_name


def get_aug_consecutive_name(names, aug_dir):
    res_names = []
    res_nums = []
    res_orig_nums = []
    res_orig_names = []
    assert len(names)<200, f'{len(names)}'
    for i, name in enumerate(names):
        name = name.split('/')[-1]
        name_list = "home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed".split('/')
        name_list[6] = 'processed'
        name_list.append(name)
        res_orig_names.append(os.path.join('/', *name_list))
        res_orig_nums.append(len(glob.glob(os.path.join('/', *name_list, '*'))))
        print(f'Name: {name}, res_orig_nums: {res_orig_nums[i]}')
        name_list[6] = aug_dir
        res_names.append(os.path.join(*name_list))
        nums = len(glob.glob(os.path.join('/', *name_list, '*')))
        # assert nums == 1000, f'nums: {nums}, name_list: {os.path.join(*name_list,)}'
        res_nums.append(nums)
    return res_names, res_nums, res_orig_names, res_orig_nums

def get_aug_all_name(names, aug_dir):
    res_names = []
    res_nums = []
    res_orig_nums = []
    res_orig_names = []
    assert len(names) < 200, f'{len(names)}'
    for i, name in enumerate(names):
        name = name.split('/')[-1]
        name_list = "home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed".split('/')
        name_list[6] = 'processed'
        name_list.append(name)
        res_orig_names.append(os.path.join('/', *name_list))
        res_orig_nums.append(len(glob.glob(os.path.join('/', *name_list, '*'))))
        print(f'Name: {name}, res_orig_nums: {res_orig_nums[i]}')

        name_list[6] = aug_dir
        res_names.append(os.path.join(*name_list))
        nums = len(glob.glob(os.path.join('/', *name_list, '*')))
        assert nums == 1000, f'nums: {nums}, name_list: {os.path.join(*name_list, )}'
        res_nums.append(nums)
    return res_names, res_nums, res_orig_names, res_orig_nums

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()
    orig_path = os.path.join(args.data_dir, 'processed')
    for aug_dir in ['Aug_All']:
        print(f'Mode: {aug_dir}')
        output_dir = os.path.join(args.data_dir, aug_dir)
        os.makedirs(output_dir, exist_ok=True)
        edf_2018 = np.load(os.path.join(orig_path, 'm_new_split_k_10.npy'), allow_pickle=True).item()
        res_2018 = {}
        orig_2018 = {}
        for i in range(10):
            names = edf_2018[f'train_{i}']['names']
            temp_train_nums = edf_2018[f'train_{i}']['nums']
            res_2018[f'train_{i}'] = {}
            orig_2018[f'train_{i}'] = {}
            res_train_names, res_train_nums, res_train_orig_names, res_train_orig_nums = get_aug_consecutive_name(names,
                                                                                                                  aug_dir)
            assert len(res_train_names)<200, f'{len(res_train_names)}'
            orig_2018[f'train_{i}']['names'] = res_train_orig_names
            orig_2018[f'train_{i}']['nums'] = res_train_orig_nums
            res_2018[f'train_{i}']['names'] = res_train_names
            res_2018[f'train_{i}']['nums'] = res_train_nums

            val_names = edf_2018[f'val_{i}']['names']
            temp_val_nums = edf_2018[f'val_{i}']['nums']

            res_2018[f'val_{i}'] = {}
            orig_2018[f'val_{i}'] = {}

            res_val_names, res_val_nums, res_val_orig_names, res_val_orig_nums = get_aug_consecutive_name(val_names,
                                                                                                          aug_dir)
            res_2018[f'val_{i}']['names'] = []
            res_2018[f'val_{i}']['nums'] = []
            orig_2018[f'val_{i}']['names'] = res_val_orig_names
            orig_2018[f'val_{i}']['nums'] = res_val_orig_nums

            test_names = edf_2018[f'test_{i}']['names']
            temp_test_nums = edf_2018[f'test_{i}']['nums']
            res_2018[f'test_{i}'] = {}
            orig_2018[f'test_{i}'] = {}
            res_test_names, res_test_nums, res_test_orig_names, res_test_orig_nums = get_aug_consecutive_name(test_names,
                                                                                                              aug_dir)
            res_2018[f'test_{i}']['names'] = []
            res_2018[f'test_{i}']['nums'] = []
            orig_2018[f'test_{i}']['names'] = res_test_orig_names
            # res_2018[f'test_{i}']['names'] = get_aug_name(test_names, aug_dir)
            orig_2018[f'test_{i}']['nums'] = res_test_orig_nums

        edf_2013 = np.load(os.path.join(orig_path, 'm_new_split_k_20.npy'), allow_pickle=True).item()
        res_2013 = {}
        orig_2013 = {}
        for i in range(20):

            names = edf_2013[f'train_{i}']['names']
            temp_train_nums = edf_2013[f'train_{i}']['nums']
            res_2013[f'train_{i}'] = {}
            orig_2013[f'train_{i}'] = {}
            res_train_names, res_train_nums, res_train_orig_names, res_train_orig_nums = get_aug_consecutive_name(names,
                                                                                                                  aug_dir)
            orig_2013[f'train_{i}']['names'] = res_train_orig_names
            orig_2013[f'train_{i}']['nums'] = res_train_orig_nums
            res_2013[f'train_{i}']['names'] = res_train_names
            res_2013[f'train_{i}']['nums'] = res_train_nums

            val_names = edf_2013[f'val_{i}']['names']
            temp_val_nums = edf_2013[f'val_{i}']['nums']

            res_2013[f'val_{i}'] = {}
            orig_2013[f'val_{i}'] = {}
            res_val_names, res_val_nums, res_val_orig_names, res_val_orig_nums = get_aug_consecutive_name(val_names,
                                                                                                          aug_dir)
            res_2013[f'val_{i}']['names'] = []
            res_2013[f'val_{i}']['nums'] = []
            orig_2013[f'val_{i}']['names'] = res_val_orig_names
            orig_2013[f'val_{i}']['nums'] = res_val_orig_nums

            test_names = edf_2013[f'test_{i}']['names']
            temp_test_nums = edf_2013[f'test_{i}']['nums']
            res_2013[f'test_{i}'] = {}
            orig_2013[f'test_{i}'] = {}
            res_test_names, res_test_nums, res_test_orig_names, res_test_orig_nums = get_aug_consecutive_name(test_names,
                                                                                                              aug_dir)
            res_2013[f'test_{i}']['names'] = []
            res_2013[f'test_{i}']['nums'] = []
            orig_2013[f'test_{i}']['names'] = res_test_orig_names
            orig_2013[f'test_{i}']['nums'] = res_test_orig_nums
        np.save(os.path.join(output_dir, f'm_new_split_{aug_dir}_k_20.npy'), arr=res_2013, allow_pickle=True)
        np.save(os.path.join(output_dir, f'm_new_split_{aug_dir}_k_10.npy'), arr=res_2018, allow_pickle=True)
        # np.save(os.path.join('/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed/',
        #                      f'm_new_split_k_20.npy'), arr=orig_2013, allow_pickle=True)
        # np.save(os.path.join('/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed/',
        #                      f'm_new_split_k_10.npy'), arr=orig_2018, allow_pickle=True)
    print("---------all finished----------")


if __name__ == '__main__':
    main()
