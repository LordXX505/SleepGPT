import numpy as np
import argparse
import os
import glob
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        # default="/Volumes/T7",
                        default="/home/cuizaixu_lab/huangweixuan/data/data",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="MASS_Processed",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    args.output_dir = os.path.join(args.data_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    names = []
    nums = []
    ed = []
    for i in range(1, 6):
        base_k_path = os.path.join(args.output_dir, f'SS{i}')
        temp_names = []
        for sub in glob.glob(base_k_path+'/*'):
            if os.path.isdir(sub):
                temp_names.append(sub)
                names.append(sub)
        ed.append(len(names))
        for name in temp_names:
            print(f'------{name}-------')
            tmp = 0
            for item in os.listdir(name):
                if os.path.isfile(os.path.join(str(name), str(item))):
                    tmp += 1
            print(f'num: {tmp}')
            nums.append(tmp)

    nums = np.array(nums)
    names = np.array(names)
    assert len(nums) == len(names)
    n = len(names)

    idx2name = {}
    ssindex = 0
    for i in range(n):
        if i < ed[ssindex]:
            idx2name[i] = ssindex + 1
        else:
            ssindex += 1
            idx2name[i] = ssindex + 1
    print(f'idx2name: {idx2name}')
    assert n == 200
    k_split = 10
    idx = np.arange(200)
    idx = np.random.permutation(idx)
    print(f'permute idx: {idx}')
    res = {'SS1': {}, 'SS2': {}, 'SS3': {}, 'SS4': {}, 'SS5': {}}
    for i in range(20):
        st = i * k_split
        ed = (i + 1) * k_split
        idx_split = idx[st:ed]
        print(f'idx_split is : {idx_split}')
        idx_train = np.setdiff1d(idx, idx_split)
        np.random.shuffle(idx_train)
        for ssnum in range(1, 6):
            res[f'SS{ssnum}'][f'test_{i}'] = {}
            res[f'SS{ssnum}'][f'train_{i}'] = {}
            res[f'SS{ssnum}'][f'val_{i}'] = {}
            res[f'SS{ssnum}'][f'test_{i}']['names'] = []
            res[f'SS{ssnum}'][f'test_{i}']['nums'] = []
            res[f'SS{ssnum}'][f'train_{i}']['names'] = []
            res[f'SS{ssnum}'][f'train_{i}']['nums'] = []
            res[f'SS{ssnum}'][f'val_{i}']['names'] = []
            res[f'SS{ssnum}'][f'val_{i}']['nums'] = []
        for _ in idx_split:
            res[f'SS{idx2name[_]}'][f'test_{i}']['names'].append(names[_])
            res[f'SS{idx2name[_]}'][f'test_{i}']['nums'].append(nums[_])
        for _ in idx_train[:10]:
            res[f'SS{idx2name[_]}'][f'val_{i}']['names'].append(names[_])
            res[f'SS{idx2name[_]}'][f'val_{i}']['nums'].append(nums[_])
        for _ in idx_train[10:]:
            res[f'SS{idx2name[_]}'][f'train_{i}']['names'].append(names[_])
            res[f'SS{idx2name[_]}'][f'train_{i}']['nums'].append(nums[_])
        # res[f'train_{i}'] = {}
        # res[f'train_{i}']['names'] = names[idx_train[10:]]
        # res[f'train_{i}']['nums'] = nums[idx_train[10:]]
        # res[f'val_{i}'] = {}
        # res[f'val_{i}']['names'] = names[idx_train[:10]]
        # res[f'val_{i}']['nums'] = nums[idx_train[:10]]
        # res[f'test_{i}'] = {}
        # res[f'test_{i}']['names'] = names[idx_split]
        # res[f'test_{i}']['nums'] = nums[idx_split]
        # print(idx, st, ed, idx_train)
        # print(f'train name : {res[f"train_{i}"]["names"]}')
        # print(f'test name : {res[f"test_{i}"]["names"]}')
        # print(f'val name : {res[f"val_{i}"]["names"]}')
    for ssnum in range(1, 6):
        np.save(os.path.join(args.output_dir, f'SS{ssnum}', f'split_k_20_SS{ssnum}'), arr=res[f'SS{ssnum}'], allow_pickle=True)
    print(f'len: names: {len(names)}')


if __name__ == '__main__':
    main()
