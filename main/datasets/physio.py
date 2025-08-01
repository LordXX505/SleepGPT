import random

import numpy as np
import torch
import io
import pyarrow as pa
import os
import pandas as pd
from PIL import Image
from .base_dataset import BaseDatatset
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from .new_base_dataset import Aug_BaseDataset

class physioDataset(Aug_BaseDataset):

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
        if k is None:
            items = np.load(os.path.join(kwargs['data_dir'], 'train.npy'), allow_pickle=True).item()
            names =items['names']
            nums = items['nums']
        else:
            if 'file_name' in kwargs:
                file_name = kwargs['file_name']
                kwargs.pop('file_name')
            else:
                file_name = f'split_k_5.npy'
            print(f'physio datasets items file name: {file_name}')
            items = np.load(os.path.join(kwargs['data_dir'], f'{file_name}'), allow_pickle=True)
            if items.dtype == np.dtype('O'):
                names = items.item()[f'{split}_{k}']['names']
                nums = items.item()[f'{split}_{k}']['nums']
            else:
                names = items['names']
                nums = items['nums']
        kwargs.pop('kfold')
        super().__init__(names=names, concatenate=False, nums=nums, split=split, *args, **kwargs)

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    @property
    def channels(self):
        if self.mode == 'large':
            return np.array([4, 5, 15, 16, 18, 22, 23, 38, 39])
        else:
            return np.array([0, 3, 6, 7, 17, 18, 20, 24, 25, 40, 41])

    def get_name(self, index):
        # print(f'idx_2_nums : {self.idx_2_nums}')
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        # print(f'before start idx: {start_idx}')
        if self.pool_all:
            start_idx *= self.split_len
        # print(f'after start idx: {start_idx}')
        return self.idx_2_name[idx].split('/')[-1]

