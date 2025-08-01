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


class ISRUCDataset(Aug_BaseDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="", *args, **kwargs):
        print(f"ISRUC kwargs : {kwargs}")
        assert split in ['train', 'val', 'test']
        k = kwargs['kfold']
        if k is None:
            raise NotImplementedError
        else:
            file_name = kwargs['file_name']
            print(f'ISRUC datasets items file name: {file_name}')
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
        return np.array([4, 5, 16, 18, 22, 38])  # for all pertrain [C3, C4, EMG, EOG, F3, O1]

