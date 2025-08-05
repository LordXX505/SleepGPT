import argparse
import re

import numpy as np
import os
import glob

def get_insersect_names(files_dict, aug_names, nums_dict):
    flattened_array = np.concatenate(files_dict)

    # print(f'flattened_array: {flattened_array}')
    # print(f'aug_names: {aug_names}')
    # Extract basenames
    basename1 = np.array([os.path.basename(path) for path in flattened_array])
    basename2 = np.array([os.path.basename(path) for path in aug_names])
    nums_dict = np.concatenate(nums_dict)
    common_basenames = []
    common_basenums = []

    # Find common basenames
    for bn in basename2:
        if bn in basename1:
            common_basenames.append(bn)
            common_basenums.append(nums_dict[np.where(bn==basename1)[0][0]])
    # print(aug_names)
    # print(files_dict)
    selected_array = np.array([path for path in aug_names if os.path.basename(path) in common_basenames])
    selected_orig_array = []
    common_orig_basenums = []
    for path, nums in zip(flattened_array, nums_dict):
        if os.path.basename(path) in common_basenames:
            selected_orig_array.append(path)
            common_orig_basenums.append(nums)
    return selected_array, np.array(common_basenums), selected_orig_array, np.array(common_orig_basenums)

def save_phase(i, names=None, nums=None, phase=None):
    # print(f'save phase: {names}, {type(names)}, {names.shape}')
    res = {}
    res[f'{phase}_{i}'] = {}
    res[f'{phase}_{i}']['names'] = []
    res[f'{phase}_{i}']['nums'] = []
    if nums is not None and names is not None:
        for name, num in zip(names, nums):
            res[f'{phase}_{i}']['names'].append(name)
            res[f'{phase}_{i}']['nums'].append(num)

    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    orig_path = os.path.join(args.data_dir, 'processed')
    k_split = 1
    files_dict = {}
    nums_dict = {}
    name_index = 0
    for _, sub in enumerate(glob.glob(os.path.join(orig_path, '*'))):
        if os.path.isdir(sub):
            base_name = os.path.basename(sub)
            subject_nums = base_name[3:5]
            number = re.findall(r'\d+', base_name)[0]
            if int(number) <= 4192:
                if subject_nums not in files_dict:
                    files_dict[subject_nums] = [sub]
                    nums_dict[subject_nums] = []
                else:
                    files_dict[subject_nums].append(sub)
                temp_nums = len(glob.glob(os.path.join(sub, "*")))
                nums_dict[subject_nums].append(temp_nums)
                name_index += 1
    print(f'The sum of subjects is : {len(files_dict)}')
    files_dict = np.array(list(files_dict.values()),  dtype=object)
    nums_dict = np.array(list(nums_dict.values()),  dtype=object)
    print(f'files_dict: {files_dict}, nums_dict: {nums_dict}')
    n = len(files_dict)
    for aug_dir in ['Aug_Half']:
        All_aug_names = glob.glob(os.path.join(args.data_dir, aug_dir, '*'))
        idx = np.arange(n)
        res = {}
        res_orig = {}
        for i in range(20):
            st = i * k_split
            ed = (i + 1) * k_split
            # if i == 9:
            #     ed = n
            idx_split = idx[st:ed]
            idx_train = np.setdiff1d(idx, idx_split)
            np.random.shuffle(idx_train)

            selected_aug_array, common_basenums, selected_orig_array, common_orig_basenums = get_insersect_names(files_dict[idx_train[4:]], All_aug_names, nums_dict[idx_train[4:]])

            ret = save_phase(i, selected_aug_array, common_basenums, phase='train')
            res.update(ret)
            ret = save_phase(i, None, None, phase='val')
            res.update(ret)
            ret = save_phase(i, None, None, phase='test')
            res.update(ret)
            # ret = save_phase(i, selected_orig_array, common_orig_basenums, phase='train') for half test
            ret = save_phase(i, np.concatenate(files_dict[idx_train[4:]]), np.concatenate(nums_dict[idx_train[4:]]), phase='train') # test for all training datasets
            res_orig.update(ret)
            ret = save_phase(i, np.concatenate(files_dict[idx_train[:4]]),  np.concatenate(nums_dict[idx_train[:4]]), phase='val')
            res_orig.update(ret)
            ret = save_phase(i,  np.concatenate(files_dict[idx_split]),  np.concatenate(nums_dict[idx_split]), phase='test')
            res_orig.update(ret)

            print(idx, st, ed, idx_train)
            print(f'train name : {res[f"train_{i}"]["names"]}')
            print(f'train nums: {res[f"train_{i}"]["nums"]}')
            print(f'train name : {res_orig[f"train_{i}"]["names"]}')
            print(f'train nums: {res_orig[f"train_{i}"]["nums"]}')

        # np.save(os.path.join(args.data_dir, aug_dir, f'm_new_split_{aug_dir}_k_20.npy'), arr=res, allow_pickle=True)
        # for half
        # np.save(os.path.join(args.data_dir, 'processed', f'm_new_split_{aug_dir}_k_20.npy'), arr=res_orig, allow_pickle=True)

        # for all
        np.save(os.path.join(args.data_dir, 'processed', f'm_new_split_{aug_dir}_Orig_k_20.npy'), arr=res_orig, allow_pickle=True)

        print(f'len: names: {len(All_aug_names)}')

if __name__ == '__main__':
    main()