import numpy as np
import argparse
import os
import glob
import logging
import h5py
import sys
import re
merge_map = {0: 0,    # Wake
             1: 1,    # N1 → N1/N2
             2: 1,    # N2 → N1/N2
             3: 2,    # N3
             4: 3}    # REM

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
               'ZQC017-2020-2d603', "CXD012-2020-2416", "CPC007-2020-2394", "054-2020-2979", '201-2021-1642']

def main():
    root_path = os.path.join('/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/ums_ods')
    new_root_path = os.path.join('/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/UMS_new_F3')
    os.makedirs(new_root_path, exist_ok=True)
    logger = setup_logger(os.path.join(root_path, "logs"))
    names = []
    nums = []
    for files in sorted(os.listdir(root_path)):
        file_path = os.path.join(root_path, files)
        new_file_path = os.path.join(new_root_path, files)
        if os.path.isdir(file_path):
            if files in ignore_list:
                continue
            print(f'files: {file_path}, new_file_path: {new_file_path}')
            names.append(new_file_path)
            with h5py.File(os.path.join(file_path, 'data.h5'), 'r') as hf:
                signal_len = hf['signal'].shape[0]
                nums.append(signal_len)
                signal = hf['signal'][...]
                stage = hf['stage'][...]
                spo2 = hf['spo2'][...]
                ods = hf['ods'][...]
            stage_merged = np.vectorize(merge_map.get)(stage).astype(np.int8)
            # for sm, s in zip(stage_merged, stage):
            #     print(sm, s)
            # exit(0)
            os.makedirs(new_file_path, exist_ok=True)
            with h5py.File(os.path.join(new_file_path, 'data.h5'), 'w') as dst:
                dst.create_dataset('signal', data=signal, compression='gzip')
                dst.create_dataset('stage', data=stage_merged, compression='gzip')
                dst.create_dataset('spo2', data=spo2, compression='gzip')
                dst.create_dataset('ods', data=ods, compression='gzip')

    nums = np.array(nums)
    names = np.array(names)
    print(f'names: {len(sorted(names))}, nums: {len(nums)}, sum: {np.sum(nums)}')
    assert len(nums) == len(names)
    n = len(names)
    idx = np.arange(n)
    res = {}
    k_split = int(np.floor(n/20))
    np.random.shuffle(idx)
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
        np.save(os.path.join(new_root_path, f'UMS_20fold.npy'), arr=res, allow_pickle=True)
        logger.info(f'train_names: {train_idx}, idx_split: {idx_split}')
def read():
    new_root_path = os.path.join('/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/UMS_new_F3')
    data = np.load(os.path.join(new_root_path, f'UMS_20fold.npy'), allow_pickle=True).item()
    print(data)
    rename_prefix(data)

def clean_lry001_path(p: str, prefix="LRY001") -> str:
    """
    1. 取出最后一级目录名              →  LRY001 -2020-2336
    2. 去掉普通空格/不间断空格等空白    →  LRY001-2020-2336
    3. 截到 prefix 结束               →  LRY001
    4. 与前面的目录重新 join
    """
    NBSP = "\u00A0"
    p = p.strip()
    head, tail = os.path.split(p)
    tail = tail.replace(NBSP, "")               # 去不间断空格
    tail = re.sub(r"\s+", "", tail)             # 去普通空格
    if tail.startswith(prefix):
        tail = prefix                           # 保留前缀
    return os.path.join(head, tail)

def rename_prefix(res: dict, prefix: str = "LRY001") -> None:
    """
    遍历 res['train_X'/'val_X'/'test_X'] 里的 names 列表，若元素以 prefix 开头，就改成 prefix
    ⚠️ 就地修改，无返回值
    """
    for split_key, split_dict in res.items():
        if not isinstance(split_dict, dict) or "names" not in split_dict:
            continue

        names_list = split_dict["names"]
        for j, n in enumerate(names_list):
            n_str = n.decode() if isinstance(n, (bytes, bytearray)) else str(n)
            if prefix in n_str:
                # print(n_str)
                clean = clean_lry001_path(n_str, prefix=prefix)
                names_list[j] = clean

    np.save("/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/UMS_new_F3/UMS_20fold_fixed.npy", res)  # 保存

def add_valid():
    data_path = '/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/UMS_new_F3'
    valid = np.load('./valid.npy', allow_pickle=True).item()
    for items in glob.glob(os.path.join(data_path, '*')):
        if 'npy' in items or 'npz' in items or os.path.isfile(items):
            continue
        sub_name = os.path.split(items)[-1]
        with h5py.File(os.path.join(items, 'data.h5'), 'r+') as dst:
            dst.create_dataset('ods_valid', data=int(valid[sub_name]))

def read_h5():
    root_path = '/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/UMS_new_F3'
    ums_20_fils = np.load(os.path.join(root_path, 'UMS_20fold_fixed.npy'), allow_pickle=True).item()
    logger = setup_logger(os.path.join(root_path, "logs"))

    proj_name_num = {}
    for proj in sorted(os.listdir(root_path)):
        file_path = os.path.join(root_path, proj)
        if os.path.isdir(file_path) and proj not in ignore_list:
            h5_file = os.path.join(file_path, 'data.h5')
            if os.path.exists(h5_file):
                with h5py.File(h5_file, 'r') as hf:
                    proj_name_num[proj] = hf['signal'].shape[0]
            else:
                logger.warning(f"No HDF5 file found at {h5_file}")

    for fold_name, fold_data in ums_20_fils.items():
        for j, path in enumerate(fold_data['names']):
            proj = os.path.split(path)[-1]
            print(proj_name_num[proj], fold_data['nums'][j])
            # assert proj_name_num[proj]==fold_data['nums'][j]
            if proj in proj_name_num:
                fold_data['nums'][j] = proj_name_num[proj]
            else:
                logger.warning(f"Missing project: {proj}")
    #
    # 可选：保存修改后的 ums_20_fils
    np.save(os.path.join(root_path, 'UMS_20fold_fixed_updated.npy'), ums_20_fils)

if __name__ == '__main__':
    # main()
    # read()
    read_h5()
    # add_valid()