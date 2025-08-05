import os
import torch
import numpy as np
import argparse
import re

import numpy as np
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        # default="/Volumes/T7 Shield/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
                        default="/home/cuizaixu_lab/huangweixuan/data/data/sleep-cassette",
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
            number = re.findall(r'\d+', base_name)[0]
            if int(number) <= 4192:
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
    edf20_permutation = np.array([14, 5, 4, 17, 8, 7, 19, 12, 0, 15, 16, 9, 11, 10, 3, 1, 6, 18, 2,
                                  13])  # to have the same results as in the paper


    len_train = int(len(edf20_permutation) * 0.6)
    len_valid = int(len(edf20_permutation) * 0.2)

    ######## TRAINing files ##########
    training_files = edf20_permutation[:len_train]
    # load files
    ######## Validation ##########
    validation_files = edf20_permutation[len_train:(len_train + len_valid)]
    # load files
    ######## TesT ##########
    test_files = edf20_permutation[(len_train + len_valid):]
    # load files

    for i in range(1):
        res[f'train_{i}'] = {}
        res[f'train_{i}']['names'] = []
        res[f'train_{i}']['nums'] = []
        for ds_idx in training_files:
            for tn_idx in files_dict[ds_idx]:
                res[f'train_{i}']['names'].append(names[tn_idx])
                res[f'train_{i}']['nums'].append(nums[tn_idx])
        res[f'val_{i}'] = {}
        res[f'val_{i}']['names'] = []
        res[f'val_{i}']['nums'] = []
        for ds_idx in validation_files:
            for tn_idx in  files_dict[ds_idx]:
                res[f'val_{i}']['names'].append(names[tn_idx])
                res[f'val_{i}']['nums'].append(nums[tn_idx])
        res[f'test_{i}'] = {}
        res[f'test_{i}']['names'] = []
        res[f'test_{i}']['nums'] = []
        for ds_idx in test_files:
            for tn_idx in  files_dict[ds_idx]:
                res[f'test_{i}']['names'].append(names[tn_idx])
                res[f'test_{i}']['nums'].append(nums[tn_idx])
        print(f'train name : {res[f"train_{i}"]["names"]}')
        print(f'test name : {res[f"test_{i}"]["names"]}')
        print(f'val name : {res[f"val_{i}"]["names"]}')
    np.save(os.path.join(args.output_dir, f'edf_downstream_TCC'), arr=res, allow_pickle=True)
    print(f'len: names: {len(names)}')

if __name__ == '__main__':
    main()

