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
path = '../../data/data/MASS_aug1/SS2/E2/01-02-0001/train'

all_path = sorted(glob.glob(path+'/*'))
print(len(all_path))
# channel = np.array([4, 5, 15, 16, 18])
cnt = 0
for i in range(len(all_path)):
    item = all_path[i]
    tables = pa.ipc.RecordBatchFileReader(
        pa.memory_map(item, "r")
    ).read_all()

    x = np.array(tables['x'][0].as_py())[[0, 1, 2, 3, 4, 5, 6, 7]]

    x = torch.from_numpy(x).float()

    Spindles = torch.from_numpy(np.array(tables['Spindles'].to_pylist())).long().squeeze()
    print(torch.sum(Spindles))
    if torch.sum(Spindles) > 25:
        item2 = all_path[i+1]
        table2 = pa.ipc.RecordBatchFileReader(
            pa.memory_map(item2, "r")
        ).read_all()

        x2 = np.array(table2['x'][0].as_py())[[0, 1, 2, 3, 4, 5, 6, 7]]

        x2 = torch.from_numpy(x2).float()
        fig, Axes = plt.subplots(nrows=8, ncols=2, sharex='all', figsize=(30, 32))
        for c in range(8):
            Axes[c][0].plot(x[c])
            Axes[c][1].plot(x2[c])
        plt.show()

    # print(stage)
    # if stage!=3:
    #     continue
    #
    # info = mne.create_info(ch_names=["C3", "C4", "ECG", "EOG"], sfreq=100, ch_types=['eeg', 'eeg', 'ecg', 'eog'])
    # raw = mne.io.RawArray(data=x, info=info)
    # raw = raw.copy().filter(l_freq=0.3, h_freq=35)
    # x = torch.from_numpy(raw.get_data()).unsqueeze(0)