import argparse
import os

import numpy
import numpy as np
import glob
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path", type=str,
                        default='/home/cuizaixu_lab/huangweixuan/data/data/MASS_aug_new_2/SS2',
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--new_path", type=str, default='/home/cuizaixu_lab/huangweixuan/data/data/MASS_aug_new_0/SS2',
                        help="Directory where to save outputs.")
    parser.add_argument("--expert", type=str, default="E1",
                        help="Log file.")
    args = parser.parse_args()

    orig_path = args.orig_path
    new_path = args.new_path

    orig_file = np.load(os.path.join(orig_path, f'all_split_{args.expert}_new_5.npy'), allow_pickle=True)

    name_e = os.path.join(new_path, args.expert)
    names = []
    train_nums = {}
    test_nums = {}
    for sub in glob.glob(name_e + '/*'):
        names.append(sub)
    for name in names:
        print(f'------{name}-------')
        name_train = os.path.join(name, 'train')
        name_test = os.path.join(name, 'test')
        tmp = 0
        for item in os.listdir(name_train):
            if os.path.isfile(os.path.join(str(name_train), str(item))):
                tmp += 1
        print(f'num: {tmp}')
        train_nums[name_train] = tmp
        tmp = 0
        for item in os.listdir(name_test):
            if os.path.isfile(os.path.join(str(name_test), str(item))):
                tmp += 1
        print(f'num: {tmp}')
        test_nums[name_test] = tmp
    res = {}
    print(train_nums)
    for i in range(5):
        print(f'************ split = {i} ************')
        orig_names = orig_file.item()[f'train_{i}']['names']
        train_names = []
        test_names = []
        temp_train_nums = []
        temp_test_nums = []
        val_names = []
        temp_val_nums = []
        for name in orig_names:
            base_orig_name = name.split('/')[-2]
            new_train_name = os.path.join(new_path, args.expert, base_orig_name, 'train')
            train_names.append(new_train_name)
            temp_train_nums.append(train_nums[new_train_name])
        print(f'orig_name: {orig_names}, new_name: {train_names}')
        orig_names = orig_file.item()[f'test_{i}']['names']
        for name in orig_names:
            base_orig_name = name.split('/')[-2]
            new_test_name = os.path.join(new_path, args.expert, base_orig_name, 'test')
            test_names.append(new_test_name)
            temp_test_nums.append(test_nums[new_test_name])
        print(f'orig_name: {orig_names}, new_name: {test_names}')

        orig_names = orig_file.item()[f'val_{i}']['names']
        for name in orig_names:
            base_orig_name = name.split('/')[-2]
            new_val_name = os.path.join(new_path, args.expert, base_orig_name, 'test')
            val_names.append(new_val_name)
            temp_val_nums.append(test_nums[new_val_name])
        print(f'orig_name: {orig_names}, new_name: {val_names}')

        train_names = np.array(train_names)
        test_names = np.array(test_names)
        val_names = np.array(val_names)
        temp_test_nums = np.array(temp_test_nums)
        temp_train_nums = np.array(temp_train_nums)
        temp_val_nums = np.array(temp_val_nums)
        res[f'train_{i}'] = {}
        res[f'train_{i}']['names'] = train_names
        res[f'train_{i}']['nums'] = temp_train_nums
        res[f'val_{i}'] = {}
        res[f'val_{i}']['names'] = val_names
        res[f'val_{i}']['nums'] = temp_val_nums
        res[f'test_{i}'] = {}
        res[f'test_{i}']['names'] = test_names
        res[f'test_{i}']['nums'] = temp_test_nums

    np.save(os.path.join(args.new_path, f'all_split_{args.expert}_new_5.npy'), arr=res, allow_pickle=True)

if __name__ == '__main__':
    main()