import numpy as np
import argparse
import os
import glob
import logging
from sklearn.model_selection import train_test_split

def main(dataset_num):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Volumes/T7",
                        # default="/home/cuizaixu_lab/huangweixuan/data",
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
    for i in range(dataset_num, dataset_num+1):
        base_k_path = os.path.join(args.output_dir, f'SS{i}')
        temp_names = []
        for sub in glob.glob(base_k_path+'/*'):
            if os.path.isdir(sub):
                temp_names.append(sub)
                names.append(sub)
        ed.append(len(names))
        for name in temp_names:
            # print(f'------{name}-------')
            tmp = 0
            for item in os.listdir(name):
                if os.path.isfile(os.path.join(str(name), str(item))):
                    tmp += 1
            # print(f'num: {tmp}')
            nums.append(tmp)

    nums = np.array(nums)
    names = np.array(names)
    assert len(nums) == len(names)
    n = len(names)
    idx = np.arange(n)
    res = {}
    for i in range(20):
        random_state = (2019 + i)
        print(f'random_state: {random_state}')
        train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=random_state)
        print(train_idx, test_idx)
        num_all = 0
        res[f'test_{i}'] = {}
        res[f'test_{i}']['names'] = []
        res[f'test_{i}']['nums'] = []
        res[f'val_{i}'] = {}
        res[f'val_{i}']['names'] = []
        res[f'val_{i}']['nums'] = []
        res[f'train_{i}'] = {}
        res[f'train_{i}']['names'] = []
        res[f'train_{i}']['nums'] = []
        for _ in test_idx:
            res[f'test_{i}']['names'].append(names[_])
            res[f'test_{i}']['nums'].append(nums[_])
            num_all += nums[_]
        for _ in test_idx:
            res[f'val_{i}']['names'].append(names[_])
            res[f'val_{i}']['nums'].append(nums[_])
        for _ in train_idx:
            res[f'train_{i}']['names'].append(names[_])
            res[f'train_{i}']['nums'].append(nums[_])
        print(num_all//20)
    np.save(os.path.join(args.output_dir, f'SS{dataset_num}', f'MASS_channel_SS{dataset_num}.npy'), arr=res, allow_pickle=True)
    print(f'len: names: {len(names)}')


if __name__ == '__main__':
    for i in [1, 2, 3, 5]:
        main(dataset_num=i)
