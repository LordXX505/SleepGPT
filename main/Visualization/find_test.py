import glob
import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

# from main.datamodules.Multi_datamodule import MultiDataModule
# from main.modules.backbone_pretrain import Model_Pre
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
import mne
# from main.modules.backbone import Model
import pyarrow as pa
import mne
# from main.modules import multiway_transformer
from mne.preprocessing import (
    create_eog_epochs,
    create_ecg_epochs,
    compute_proj_ecg,
    compute_proj_eog,
)
# from sklearn.linear_model import LinearRegression
path = '../../data/shhs1-205154'

all_path = sorted(glob.glob(path+'/*'))
print(len(all_path))
channel = np.array([4, 5, 15, 16, 18])
cnt = 0
for item in all_path:
    cnt += 1
    if cnt <= 140:
        continue
    if cnt > 160:
        break
    print(item)
    tables = pa.ipc.RecordBatchFileReader(
        pa.memory_map(item, "r")
    ).read_all()

    x = np.array(tables['x'][0].as_py())[[0, 1, 2, 4]]

    x = torch.from_numpy(x).float()
    stage = torch.from_numpy(np.array(tables['stage'])).long()
    print(stage)
    if stage!=3:
        continue

    info = mne.create_info(ch_names=["C3", "C4", "ECG", "EOG"], sfreq=100, ch_types=['eeg', 'eeg', 'ecg', 'eog'])
    raw = mne.io.RawArray(data=x, info=info)
    raw = raw.copy().filter(l_freq=0.3, h_freq=35)
    x = torch.from_numpy(raw.get_data()).unsqueeze(0)