import random

import numpy as np
import torch
import io
import pyarrow as pa
import os
import pandas as pd
from PIL import Image
from .base_dataset import BaseDatatset
from .new_base_dataset import Aug_BaseDataset
from pytorch_lightning.utilities.rank_zero import rank_zero_info


"""
data_dict = {
            'signal': signal,
            'stage': stage,
            'good_channels': good_channels,
            'pathology': [pathology]
        }
"""
class UMSDataset(Aug_BaseDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['signal', 'stage']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    def __init__(self, split="", pathology_name=None, *args, **kwargs):

        assert split in ['train', 'val', 'test']
        k = kwargs['kfold']
        if k is None:
            raise NotImplementedError
        else:
            if 'file_name' not in kwargs.keys():
                file_name = f'cap_{pathology_name}.npy'
            else:
                file_name = kwargs['file_name']
                kwargs.pop('file_name')
            rank_zero_info(f'ums datasets items file name: {file_name}, kfold is {k}')
            items = np.load(os.path.join(kwargs['data_dir'], f'{file_name}'), allow_pickle=True)
            if items.dtype == np.dtype('O'):
                data = items.item()[f'{split}_{k}']
                names = data['names']
                if 'nums' in data.keys():
                    nums = data['nums']
                else:
                    nums = None
            else:
                names = items['names']
                if 'nums' in items.keys():
                    nums = items['nums']
                else:
                    nums = None
        kwargs.pop('kfold')
        expert = kwargs.pop('expert', None)
        rank_zero_info(f"data_dir: {kwargs['data_dir']}")
        self.channels_set = 'Fpz' in kwargs['data_dir']
        # print(len(names), len(nums))
        rank_zero_info(f'channel_set={self.channels_set}')
        super().__init__(names=names, concatenate=False, nums=nums, split=split, *args, **kwargs)

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    @property
    def channels(self):
        if self.channels_set:
            return np.array([36])
        else:
            return np.array([22])

    def get_name(self, index):
        # print(f'idx_2_nums : {self.idx_2_nums}')
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        # print(f'before start idx: {start_idx}')
        if self.pool_all:
            start_idx *= self.split_len
        # print(f'after start idx: {start_idx}')
        try:
            return int(self.idx_2_name[idx].split('/')[-2].split('-')[-1])
        except:
            return self.idx_2_name[idx].split('/')[-1]



