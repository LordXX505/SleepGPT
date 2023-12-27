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

    """This is a dataset for physio 2018"""
    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Spindle_label', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="", *args, **kwargs):

        assert split in ['train', 'val', 'test']
        k = kwargs['kfold']
        expert = kwargs['expert']
        if k is None:
            raise NotImplementedError
        else:
            items = np.load(os.path.join(kwargs['data_dir'], f'all_split_{expert}.npy'), allow_pickle=True)
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

    def get_name(self, index):
        # print(f'idx_2_nums : {self.idx_2_nums}')
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        # print(f'before start idx: {start_idx}')
        if self.pool_all:
            start_idx *= self.split_len
        # print(f'after start idx: {start_idx}')
        return int(self.idx_2_name[idx].split('/')[-2].split('-')[-1])

