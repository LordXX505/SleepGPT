import numpy as np
import argparse
import os
import glob
import logging
import h5py
import sys

def setup_logger(log_filename):
    """
    Sets up and returns a logger that logs to both file and stdout.
    If logger already exists, it will not add duplicate handlers.
    """
    logger = logging.getLogger("my_logger") 
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # 文件日志 handler
        fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        # 控制台日志 handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 绑定 handler
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
ignore_list = ['DYJ019-2020-2619', 'HRH013-2020-2437', 'LCB011-2020-2417', 'MYZ002', 'ZJ015-2020-2427', 
               'ZQC017-2020-2603', "CXD012-2020-2416", "CPC007-2020-2394"]

def main():
    root_path = os.path.join('/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/UMS')
    logger = setup_logger(os.path.join(root_path, "logs"))
    names = []
    nums = []
    for files in sorted(os.listdir(root_path)):
        file_path = os.path.join(root_path, files)
        if os.path.isdir(file_path):
            if files in ignore_list:
                continue
            print(f'files: {file_path}')
            names.append(file_path)
            with h5py.File(os.path.join(file_path, 'data.h5'), 'r') as hf:
                    signal_len = hf['signal'].shape[0]
                    nums.append(signal_len)
    nums = np.array(nums)
    names = np.array(names)
    print(f'names: {len(sorted(names))}, nums: {len(nums)}')
    assert len(nums) == len(names)
    n = len(names)
    idx = np.arange(n)
    res = {}
    k_split = int(np.floor(n/20))
    logger.info(f'k_split: {k_split}')
    for i in range(20):
        st = i * k_split
        ed = (i + 1) * k_split
        if i==19:
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
        for _ in train_idx[:10]:
            res[f'val_{i}']['names'].append(names[_])
            res[f'val_{i}']['nums'].append(nums[_])
        for _ in train_idx[10:]:
            res[f'train_{i}']['names'].append(names[_])
            res[f'train_{i}']['nums'].append(nums[_])
        np.save(os.path.join(root_path, f'UMS_20fold.npy'), arr=res, allow_pickle=True)
        # logger.info(f'train_names: {train_idx}, idx_split: {idx_split}')
if __name__ == '__main__':
    main()