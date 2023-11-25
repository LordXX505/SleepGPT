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


class SDDataset(BaseDatatset):
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
        assert split in ['train','test']
        try:
            names = np.load(os.path.join(kwargs['data_dir'], 'train.npy'), allow_pickle=True)
        except:
            names = np.array(glob.glob(kwargs['data_dir']+'/*/*'))
        nums = None
        kwargs.pop('kfold', None)
        kwargs.pop('expert', None)


        # print(os.path.join(kwargs['data_dir'], 'train.npy'))
        super().__init__(names=names, split=split, nums=None, *args, **kwargs)

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    @property
    def channels(self):
        return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56])