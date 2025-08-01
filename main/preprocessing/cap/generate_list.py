import numpy as np
import argparse
import os
import glob
import logging
import h5py
def main(dataset_num):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Volumes/T7/data/cap",
                        # default="/home/cuizaixu_lab/huangweixuan/data",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="process",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    args.output_dir = os.path.join(args.data_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, args.log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    for pathology in ['n','nfle', 'rbd']:
    # for pathology in ['n', 'ins', 'narco', 'nfle', 'plm', 'rbd', 'sdb']:
        base_k_path = os.path.join(args.output_dir, f'{pathology}')
        names = []
        nums = []
        for sub in glob.glob(base_k_path+'/*'):
            if os.path.isdir(sub):
                names.append(sub)
                with h5py.File(os.path.join(sub, 'data.h5'), 'r') as hf:
                    signal_len = hf['signal'].shape[0]
                    nums.append(signal_len)
        nums = np.array(nums)
        names = np.array(names)
        print(f'pathology: {pathology}, nums: {np.sum(nums)}')
        assert len(nums) == len(names)
        n = len(names)
        idx = np.arange(n)
        res = {}
        k_split = int(np.floor(n/4))
        print(f'pa: {pathology}, k_split: { k_split}')
        for i in range(4):
            st = i * k_split
            ed = (i + 1) * k_split
            if i==3:
                ed = len(idx)
            idx_split = idx[st:ed]
            train_idx = np.setdiff1d(idx, idx_split)
            np.random.shuffle(train_idx)
            print(f'{train_idx}, {idx_split}')
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
            for _ in idx_split:
                res[f'test_{i}']['names'].append(names[_])
                res[f'test_{i}']['nums'].append(nums[_])
                num_all += nums[_]
            for _ in train_idx[:1]:
                res[f'val_{i}']['names'].append(names[_])
                res[f'val_{i}']['nums'].append(nums[_])
            for _ in train_idx[1:]:
                res[f'train_{i}']['names'].append(names[_])
                res[f'train_{i}']['nums'].append(nums[_])
        np.save(os.path.join(args.output_dir, f'{pathology}', f'cap_{pathology}.npy'), arr=res, allow_pickle=True)
        print(f'len: names: {len(names)}')


if __name__ == '__main__':
    for i in range(3, 4):
        main(dataset_num=i)
