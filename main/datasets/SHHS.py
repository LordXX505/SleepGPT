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


class SHHSDataset(BaseDatatset):
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

        try:
            names = np.load(os.path.join(kwargs['data_dir'], f'{split}.npy'), allow_pickle=True)
            nums = None
        except:
            try:
                data = np.load(os.path.join(kwargs['data_dir'], f'{split}.npz'), allow_pickle=True)
                names = data['names']
                nums = data['nums']
            except:
                names = np.array(glob.glob(kwargs['data_dir'] + '/*/*'))
                nums = None
        # print(os.path.join(kwargs['data_dir'], 'train.npy'))
        super().__init__(names=names, nums=nums, split=split, *args, **kwargs)

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    @property
    def channels(self):
        return np.array([4, 5, 15, 16, 18])