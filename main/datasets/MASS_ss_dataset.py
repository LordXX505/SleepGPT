import random

import numpy as np
import torch
import io
import pyarrow as pa
import os
import pandas as pd
from PIL import Image
from .MASS_dataset import MASSDataset
from pytorch_lightning.utilities.rank_zero import rank_zero_info


from .new_base_dataset import Aug_BaseDataset
class MASSDataset_SS1(MASSDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Spindle_label', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="",  *args, **kwargs):
        super().__init__(split=split, SSNum=1, *args, **kwargs)
    @property
    def channels(self):
        if self.mode == 'large':
            return np.array([4, 5, 16, 18, 22, 38, 52])
        # c3, c4, emg, eog, f3, o1, pz
        else:
            return np.array([0, 3, 6, 7, 17, 18, 20, 24, 38, 40, 54])
    # np.array([4, 5, 16, 18, 22, 36, 38, 52]) for all pertrain [ABD, AIRFLOW, C3, C4, EMG, EOG, F3, Fpz, O1, Pz]
    # for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz]

class MASSDataset_SS2(MASSDataset):
    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Spindle_label', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="", *args, **kwargs):
        super().__init__(split=split, SSNum=2, *args, **kwargs)

    @property
    def channels(self):
        if self.mode == 'large':
            return np.array([4, 5, 16, 18, 22, 36, 38, 52])
        else:
            return np.array([6, 7, 17, 18, 20, 24, 38, 40, 54])
    # np.array([4, 5, 16, 18, 22, 36, 38, 52]) for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz]


class MASSDataset_SS3(MASSDataset):
    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Spindle_label', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="", *args, **kwargs):
        super().__init__(split=split, SSNum=3, *args, **kwargs)

    @property
    def channels(self):
        if self.mode == 'large':
            return np.array([4, 5, 16, 18, 22, 38, 52])
        else:
            return np.array([6, 7, 17, 18, 20, 24, 40, 54])
    # np.array([4, 5, 16, 18, 22, 36, 38, 52]) for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz]


class MASSDataset_SS4(MASSDataset):
    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Spindle_label', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="", *args, **kwargs):
        super().__init__(split=split, SSNum=4, *args, **kwargs)

    @property
    def channels(self):
        if self.mode == 'large':
            return np.array([4, 5, 16, 18, 38])
        else:
            return np.array([6, 7, 17, 18, 20, 40])
    # np.array([4, 5, 16, 18, 22, 36, 38, 52]) for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz]


class MASSDataset_SS5(MASSDataset):
    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['x', 'Spindle_label', 'Stage_label']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False

    def __init__(self, split="", *args, **kwargs):
        super().__init__(split=split, SSNum=5, *args, **kwargs)

    @property
    def channels(self):
        if self.mode == 'large':
            return np.array([4, 5, 16, 18, 22, 38, 52])
        else:
            return np.array([6, 7, 17, 18, 20, 24, 40, 54])
    # np.array([4, 5, 16, 18, 22, 36, 38, 52]) for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz]
