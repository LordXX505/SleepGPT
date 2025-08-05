import glob
import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import patches
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from scipy import signal
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
from main.modules import Model
import pytorch_lightning as pl
from main.config import ex
from typing import List
path = '../../data/data/MASS'
mu = np.array([-6.8460e-02, 1.9104e-01, 3.8937e-01, -2.0938e+00,
                        1.6496e-03, -4.8439e-05, 8.1125e-04,
                        7.1748e-05])
std = np.array([34.6887, 34.9556, 23.2826, 35.4035, 26.8738,
                         4.9272, 25.1366, 3.6142])
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

def get_fft(x, choose_channels=[0], hop_length=100, patch_size=200):
    res = []
    # ids_keep = self.choose_channels.to(_x.device)
    n_fft = 256
    hop_length = hop_length
    win_length = patch_size
    window = torch.hann_window(win_length, device=x.device)
    x_fft = x
    for c in choose_channels:
        spec = torch.stft(x_fft[c, :], n_fft, hop_length, win_length, window, return_complex=False)
        magnitude = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)[:100, 1:]
        log_magnitude = 10 * torch.log10(magnitude + 1e-8)
        log_magnitude = log_magnitude.transpose(-2, -1)
        mean = log_magnitude.mean(dim=-1)
        std = log_magnitude.std(dim=-1)
        res.append((log_magnitude - mean.unsqueeze(-1)) / std.unsqueeze(-1))
    res = torch.stack(res, dim=1)
    return res

def plot_spindle_box(filtdata, dist_down, dist_up):
    plt.plot(filtdata)

    ax = plt.gca()
    # for down, up in zip(dist_down, dist_up):
    #     rect = patches.Rectangle((down, -100), up - down, 50,
    #                              linewidth=1, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)
    # plt.savefig(f'/Users/hwx_admin/Sleep/result/sc/sub1001.svg')
    plt.show()
    plt.close('all')

def plot_fft_eeg(index=2):
    def find_continuous_ranges(data):
        if data is None or len(data) == 0:
            return []

        data.sort()  # 确保数据是排序的
        ranges = []
        start = data[0]
        end = data[0]

        for i in range(1, len(data)):
            if data[i] == end + 1:
                end = data[i]
            else:
                ranges.append((start, end))
                start = data[i]
                end = data[i]

        ranges.append((start, end))
        return ranges
    # items = glob.glob(os.path.join("/Volumes/T7/MASS_Processed/SS2/01-02-0001", '*'))
    items = glob.glob(os.path.join("/Users/hwx_admin/Sleep/data/MASS_aug_new_1/SS2/E2/01-02-0001/train", '*'))
    # items = ["/Users/hwx_admin/Sleep/data/MASS_aug_new_2/SS2/E2/01-02-0001/test/00341.arrow"]
    np.random.seed(2020)
    np.random.shuffle(items)
    # 6-1
    for _, item in enumerate(items):
        print(f'index of items: {_}')
        tables = pa.ipc.RecordBatchFileReader(
            pa.memory_map(item, "r")
        ).read_all()
        # if np.array(tables['stage'])[0] != 2:
        #     continue
        print(np.where(np.array(tables['Spindles'].to_pylist()[0]) == 1.0)[0])
        if len(np.where(np.array(tables['Spindles'].to_pylist()[0]) == 1.0)[0]) == 0:
            continue
        init_ranges = np.where(np.array(tables['Spindles'].to_pylist()[0]) == 1.0)[0]
        for channel in range(0, index):
            x = np.array(tables['x'][0].as_py())
            plt.figure(figsize=(24, 10))
            b, a = signal.butter(8, [0.1, 0.4], 'band')
            c3 = x[channel]*1e6
            filtdata = c3
            # c3 = (c3-mu[channel])/std[channel]
            # filtdata = c3[500:1100]
            # filtdata = signal.filtfilt(b, a, c3)
            # plt.plot(filtdata)
            ranges = find_continuous_ranges(init_ranges)
            plot_spindle_box(filtdata, dist_down=[1444-1350, 1806-1350], dist_up=[1573-1350, 1918-1350])
            # for numbers, _range in enumerate(ranges[3:]):
            #     print(f'numbers: {numbers}')
            #     dist_down = 200
            #     if _range[0]-200 < 0:
            #         dist_down = _range[0]
            #     dist_up = _range[1] - _range[0] + dist_down
            #
            #     plt.plot(filtdata[max(0, _range[0]-200):min(_range[1]+200, 2000)])
            #
            #     ax = plt.gca()
            #     rect = patches.Rectangle((dist_down, -100), dist_up - dist_down, 50,
            #                              linewidth=1, edgecolor='r', facecolor='none')
            #     ax.add_patch(rect)
            #     plt.savefig(f'/Users/hwx_admin/Sleep/result/sc/sub1001_{channel}.svg')
            #     plt.show()
            #     plt.close('all')
            # plt.savefig(f'/Users/hwx_admin/Sleep/result/sc/sub1001_{channel}.svg')
            # c3 = torch.from_numpy(c3.copy()).view(1, -1)
            # plt.figure(figsize=(12, 12))
            # res = get_fft(c3)
            # res = res[:, 0, :20]
            # res = torch.flip(res, dims=[0])
            # plt.imshow(res)
            # plt.savefig(f'/Users/hwx_admin/Sleep/result/sc/sub1001_fft_{index}.svg')

# @ex.automain
# def main(_config):
#     pre_train = Model(_config)
#     print(_config)
#     pl.seed_everything(512)
#     dm = MultiDataModule(_config, kfold=0)
#     dm.setup(stage='test')
#     pre_train.eval()
#     dm_ = dm.dms[0]
#     path_list = glob.glob('Model/checkpoint/')
#     for path in path_list:
#         # ckpt = torch.load(path, map_location=torch.device('cpu'))
#         # import matplotlib.pyplot as plt
#         # x = range(2000)
#         # cls = ckpt['output']
#         # spindle = ckpt['target']
#         # print(torch.where(spindle == 1))
#         # print(path)
#         for i in range(len(cls)):
#             plt.plot(x, spindle[i].numpy(), c='r')
#             plt.plot(x, cls[i].detach().numpy(), c='b')
#             # batch = dm_.train_dataset[ckpt['idx'][i][0].detach().numpy()]
#             # batch = dm_.train_dataset.collate([batch])
#             plt.legend()
#             plt.show()
#
#             # x = x[[0, 1, 2, 3, 4, 5, 6, 7]]
#             # x = torch.from_numpy(x).float()
#             # info = mne.create_info(ch_names=["C3", "C4", "EMG", "EOG", 'F3', 'Fpz', 'O1', 'Pz'], sfreq=100,
#             #                        ch_types=['eeg', 'eeg', 'emg', 'eog', 'eeg', 'eeg', 'eeg', 'eeg'])
#             # raw = mne.io.RawArray(data=x, info=info)
#             # raw.plot(n_channels=1, title='Raw')
if __name__ == '__main__':
    plot_fft_eeg(1)