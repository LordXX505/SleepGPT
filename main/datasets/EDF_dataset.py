import random

import numpy as np
import torch
import io
import pyarrow as pa
import os
import pandas as pd
from PIL import Image
from .base_dataset import BaseDatatset


class EDfDataset(BaseDatatset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="", *args, **kwargs):

        assert split in ['train', 'val', 'test']
        k = kwargs['kfold']
        if k is None:
            names = np.load(os.path.join(kwargs['data_dir'], 'EDF.npy'), allow_pickle=True)
            nums = None
        else:
            file_name = kwargs['file_name']

            print(f'edf datasets items file name: {file_name}')
            items = np.load(os.path.join(kwargs['data_dir'], f'{file_name}'), allow_pickle=True)
            if items.dtype == np.dtype('O'):
                names = items.item()[f'{split}_{k}']['names']
                nums = items.item()[f'{split}_{k}']['nums']
            else:
                names = items['names']
                nums = items['nums']
            kwargs.pop('file_name')
        kwargs.pop('kfold')
        super().__init__(names=names, concatenate=False, nums=nums, split=split, *args, **kwargs)

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    @property
    def channels(self):
        return np.array([16, 18, 36, 52])
    # for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz], edf: [EMG, EOG,  Fpz, Pz]

    def get_name(self, index):
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        if self.pool_all:
            start_idx *= self.split_len
        try:
            return self.idx_2_name[idx].split('/')[-1]
        except:
            return int(self.idx_2_name[idx].split('/')[-2].split('-')[-1])

