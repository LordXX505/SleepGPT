import mne
import numpy as np
import pyarrow as pa
import os
import glob
import pandas as pd
import gc
import torch
import os
import sys
import matplotlib.patches as patches

sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone import Model
from main.transforms import unnormalize
import pytorch_lightning as pl
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
import torch
import re

data_path = "/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette"
aug_mode = ['Aug_F3', 'Aug_C4', 'Aug_F3_C4']
def compare():
    orig_items = glob.glob(os.path.join(data_path, 'processed', 'SC*'))
    for mode in aug_mode:
        aug_items = glob.glob(os.path.join(data_path, mode, '*'))
        assert len(aug_items) == len(orig_items), f'mode = {mode}, aug_nums: {len(aug_items)}, orig_item_nums: {len(orig_items)}'
        for subject in aug_items:
            if os.path.isdir(subject):
                nums = glob.glob(os.path.join(subject, '*'))
                orig_nums = glob.glob(os.path.join(data_path, f'processed/{os.path.basename(subject)}', '*'))
                assert len(nums)==len(orig_nums)

if __name__ == '__main__':
    compare()