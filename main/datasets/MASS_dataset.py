import random

import numpy as np
import torch
import io
import pyarrow as pa
import os
import pandas as pd
from PIL import Image
from .base_dataset import BaseDatatset


class MASSDataset(BaseDatatset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Spindle_label', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="", SSNum=None, *args, **kwargs):

        assert split in ['train', 'val', 'test']
        k = kwargs['kfold']
        expert = kwargs['expert']
        if k is None:
            raise NotImplementedError
        else:
            if 'file_name' not in kwargs.keys():
                file_name = f'split_k_20_SS{SSNum}.npy'
            else:
                file_name = kwargs['file_name']
                kwargs.pop('file_name')
            print(f'mass datasets items file name: {file_name}')
            items = np.load(os.path.join(kwargs['data_dir'], f'{file_name}'), allow_pickle=True)
            if items.dtype == np.dtype('O'):
                names = items.item()[f'{split}_{k}']['names']
                nums = items.item()[f'{split}_{k}']['nums']
            else:
                names = items['names']
                nums = items['nums']
        kwargs.pop('kfold')
        kwargs.pop('expert')
        # print(f'len names:{len(names)}, expert:{expert}')
        # print(len(names), len(nums))
        super().__init__(names=names, concatenate=False, nums=nums, split=split, *args, **kwargs)

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    @property
    def channels(self):
        return np.array([4, 5, 16, 18, 22, 36, 38, 52])
    # np.array([4, 5, 16, 18, 22, 36, 38, 52]) for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz]

    def get_name(self, index):
        # print(f'idx_2_nums : {self.idx_2_nums}')
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        # print(f'before start idx: {start_idx}')
        if self.pool_all:
            start_idx *= self.split_len
        # print(f'after start idx: {start_idx}')
        try:
            return self.idx_2_name[idx].split('/')[-1]
        except:
            return int(self.idx_2_name[idx].split('/')[-2].split('-')[-1])



