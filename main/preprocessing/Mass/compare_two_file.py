import argparse
import numpy as np
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path", type=str,
                        default='/home/cuizaixu_lab/huangweixuan/data/data/MASS_aug_new_2/SS2',
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--new_path", type=str, default='/home/cuizaixu_lab/huangweixuan/data/data/MASS_aug_new_1/SS2',
                        help="Directory where to save outputs.")
    parser.add_argument("--expert", type=str, default="E1",
                        help="Log file.")
    args = parser.parse_args()
    orig_path = args.orig_path
    new_path = args.new_path

    orig_file = np.load(os.path.join(orig_path, f'all_split_{args.expert}_new_5.npy'), allow_pickle=True)

    new_file = np.load(os.path.join(new_path, f'all_split_{args.expert}_new_5.npy'), allow_pickle=True)

    for k_fold in range(5):
        orig_names = orig_file.item()[f'train_{k_fold}']['names']
        new_names = new_file.item()[f'train_{k_fold}']['names']
        new_nums = new_file.item()[f'train_{k_fold}']['nums']
        orig_names = np.sort(orig_names)
        index = np.argsort(new_names)
        new_names = new_names[index]
        new_nums = new_nums[index]
        for i, j, nums in zip(orig_names, new_names, new_nums):
            base_orig_name = i.split('/')[-2]
            base_new_name = j.split('/')[-2]
            assert base_orig_name == base_new_name, f'orig_names: {base_orig_name}, new_names: {base_new_name}'
            items = glob.glob(j + '/*')
            assert len(items) == nums, \
                f"len(items): {len(items)}, orig_file_nums: nums"

    print('end')


if __name__ == '__main__':
    main()
