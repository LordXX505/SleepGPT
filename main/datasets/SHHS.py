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


class SHHSDataset(Aug_BaseDataset):
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

        data_dir = kwargs['data_dir']
        file_path_npy = os.path.join(data_dir, f'{split}.npy')
        file_path_npz = os.path.join(data_dir, f'{split}.npz')
        items = None
        try:
            if os.path.exists(file_path_npy):
                items = np.load(file_path_npy, allow_pickle=True)
            elif os.path.exists(file_path_npz):
                items = np.load(file_path_npz, allow_pickle=True)
            else:
                raise FileNotFoundError(f"Neither {file_path_npy} nor {file_path_npz} found.")
        except:
            raise FileNotFoundError(f"Neither {file_path_npy} nor {file_path_npz} found.")
        if items is not None:
            print(f'items: {items}')
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
        print(f'names: {names}, file_path_npy: {file_path_npy, file_path_npz}')
        # print(os.path.join(kwargs['data_dir'], 'train.npy'))
        super().__init__(names=names, nums=nums, split=split, *args, **kwargs)

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    @property
    def channels(self):
        # rank_zero_info(f'shhs mode: {self.mode}')
        if self.mode == 'large' or self.mode == 'base' or 'conv' in self.mode:
            return np.array([4, 5, 15, 16, 18])
        else:
            return np.array([0, 3, 6, 7, 17, 18, 20])


    def get_name(self, index):
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        if self.pool_all:
            start_idx *= self.split_len
        try:
            return self.idx_2_name[idx].split('/')[-1]
        except:
            return int(self.idx_2_name[idx].split('/')[-2].split('-')[-1])
