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
import matplotlib.pyplot as plt
from typing import List
import pyarrow as pa
import mne
# from main.modules import multiway_transformer
from mne.preprocessing import (
    create_eog_epochs,
    create_ecg_epochs,
    compute_proj_ecg,
    compute_proj_eog,
)
import os
import sys
import torch
sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')

from main.datamodules.Multi_datamodule import MultiDataModule

import pytorch_lightning as pl
from main.config import ex
from typing import List
path = '../../data/data/MASS'
def get_param(nums) -> List[str]:
    color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#4DBBD5CC", '#2ecc71', '#2980b9', '#FFEDA0',
             '#e67e22', '#B883D4'
        , '#9E9E9E']
    return color[:nums]


def get_names():
    # return ['C3', 'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'F4', 'Fpz', 'O1', 'O2',
    #    'Pz']
    return ['C3', 'C4', 'EMG', 'EOG1', 'F3', 'Fpz', 'O1',
            'Pz']
@ex.automain
def main(_config):
    # pre_train = Model_Pre(_config)

    print(_config)
    pl.seed_everything(512)
    dm = MultiDataModule(_config, kfold=0)
    dm.setup(stage='train')
    # pre_train.eval()
    dm_ = dm.dms[0]
    path_list = glob.glob('../../data/ver_log/*')
    for path in path_list:
        ckpt = torch.load(path, map_location=torch.device('cpu'))
        import matplotlib.pyplot as plt
        x = range(2000)
        cls = ckpt['output']
        spindle = ckpt['target']
        print(torch.where(spindle == 1))
        print(path)
        for i in range(len(cls)):
            plt.plot(x, spindle[i].numpy(), c='r')
            plt.plot(x, cls[i].detach().numpy(), c='b')
            # batch = dm_.train_dataset[ckpt['idx'][i][0].detach().numpy()]
            # batch = dm_.train_dataset.collate([batch])
            plt.legend()
            plt.show()
            # item = os.path.join("/Users/hwx_admin/Sleep/data/data/MASS/SS2", '/'.join(name.split('/')[-3:]))
            # tables = pa.ipc.RecordBatchFileReader(
            #     pa.memory_map(item, "r")
            # ).read_all()
            # x = np.array(tables['x'][0].as_py())
            # x = x[[0, 1, 2, 3, 4, 5, 6, 7]]
            # x = torch.from_numpy(x).float()
            # info = mne.create_info(ch_names=["C3", "C4", "EMG", "EOG", 'F3', 'Fpz', 'O1', 'Pz'], sfreq=100,
            #                        ch_types=['eeg', 'eeg', 'emg', 'eog', 'eeg', 'eeg', 'eeg', 'eeg'])
            # raw = mne.io.RawArray(data=x, info=info)
            # raw.plot(n_channels=1, title='Raw')
