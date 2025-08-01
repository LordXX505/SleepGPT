
import glob
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

class MGHDataset(Aug_BaseDataset):
    def __init__(self, split="", *args, **kwargs):
        """
        SD dataset.
        Args:
            split: only train
            **kwargs:
                  transform_keys: transform and augment
                   data_dir: base data dir
                   names: subject names
                   column_names: epochs(x), spindle ,stages
                   fs: 100hz
                   epoch_duration: 30s. Mass is 20s.
                   stage: need stage labels.
                   spindle: nedd spindle labels.
       """
        print('split: ', split)
        items = None
        try:
            items = np.load(os.path.join(kwargs['data_dir'], f'{split}.npz'), allow_pickle=True)
        except:
            items = np.load(os.path.join(kwargs['data_dir'], f'{split}.npy'), allow_pickle=True)
        if items is not None:
            if isinstance(items, np.lib.npyio.NpzFile):
                names = items['names']
                nums = items['nums']
            elif items.dtype == np.dtype('O'):
                data = items.item()
                names = data['names']
                nums = data['nums']
            else:
                names = items['names']
                nums = items['nums']
        else:
            raise ValueError("Loaded data is None.")


        kwargs.pop('kfold', None)
        super().__init__(names=names, nums=nums, split=split, *args, **kwargs)

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    @property
    def channels(self):
        return np.array([0, 3, 6, 7, 17, 18, 20, 24, 40])


    def get_name(self, index):
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        if self.pool_all:
            start_idx *= self.split_len
        try:
            return self.idx_2_name[idx].split('/')[-1]
        except:
            return int(self.idx_2_name[idx].split('/')[-2].split('-')[-1])
