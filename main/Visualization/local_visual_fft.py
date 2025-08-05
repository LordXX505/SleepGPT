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
# path = '../../data/shhs1-201967'
path = '/data/data/shhs_new/shhs1-205718/'
from main.transforms import normalize
def fig_compute_psd(raw):
    fig = raw.compute_psd(tmax=np.inf, fmax=50).plot(
            average=True, picks="data", exclude="bads"
        )
    # add some arrows at 60 Hz and its harmonics:
    for ax in fig.axes[1:]:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            ax.arrow(
                x=freqs[idx],
                y=psds[idx] + 18,
                dx=0,
                dy=-12,
                color="red",
                width=0.1,
                head_width=3,
                length_includes_head=True,
            )

all_path = sorted(glob.glob(path+'/*'))
print(len(all_path))
channel = np.array([4, 5, 15, 16, 18])
cnt = 0
for item in all_path:
    cnt += 1
    if cnt <= 560:
        continue
    if cnt > 580:
        break
    print(item)
    tables = pa.ipc.RecordBatchFileReader(
        pa.memory_map(item, "r")
    ).read_all()
    x = np.array(tables['x'][0].as_py())
    print(x.shape)
    x = x[[0, 1, 2, 3, 4]]
    print(np.max(x[4]))
    norm = normalize()
    x = torch.from_numpy(x).float()
    stage = torch.from_numpy(np.array(tables['stage'])).long()
    print(stage)
    # if stage!=3:
    #     continue
    info = mne.create_info(ch_names=["C3", "C4", "ECG", "EMG", "EOG"], sfreq=100,
                           ch_types=['eeg', 'eeg', 'ecg', 'emg', 'eog'])
    raw = mne.io.RawArray(data=x, info=info)
    def func(array):
        array/=10
        return array
    raw.apply_function(func, picks=['ECG'])
    raw.plot(n_channels=5, title='Raw')
    x = x[[0, 1, 3, 4]]
    x = norm(x)*1e6
    info = mne.create_info(ch_names=["C3", "C4",  "EMG", "EOG"], sfreq=100,
                           ch_types=['eeg', 'eeg', 'emg', 'eog'])
    raw = mne.io.RawArray(data=x, info=info)

    raw.plot(n_channels=4, title='Raw', scalings=1)
    # raw = raw.copy().filter(l_freq=0.3, h_freq=35)
    # x = torch.from_numpy(raw.get_data()).unsqueeze(0)
    # n_fft = 256
    # hop_length = 25
    # win_length = 200
    # window = torch.hann_window(win_length)
    # res = []
    # for c in [0, 1, 2, 3]:
    #     spec = torch.stft(x[:, c], n_fft, hop_length, win_length, window, return_complex=False)
    #     # magnitude = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)[:, :100, 1:]
    #     magnitude = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)[:, :100]
    #
    #     log_magnitude = 20 * torch.log10(magnitude + 1e-8)
    #     print(log_magnitude.shape)
    #     log_magnitude = log_magnitude.transpose(-2, -1)
    #     mean = log_magnitude.mean(dim=-1)
    #     std = log_magnitude.std(dim=-1)
    #     res.append((log_magnitude-mean.unsqueeze(-1))/std.unsqueeze(-1))
    # res = torch.stack(res, dim=1)

    # plt.imshow(res[0, 0].t().numpy(), aspect='auto', origin='lower',
    #            )  # 使用inferno颜色映射来增强对比
    # plt.colorbar(label='Log Magnitude (dB)')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    #
    # plt.tight_layout()
    # conv = torch.nn.Conv2d(in_channels=4, out_channels=768, kernel_size=(2, 100), stride=(2, 100))
    # out = conv(res.float())
    # print(out.shape)
    # print(out.permute(0, 2, 1, 3).reshape(-1, 15, 768))
    # plt.figure()
    # plt.plot(res[0, 0, 27, :25].t().numpy())
    #
    # plt.figure()
    # res = []
    # x = x.squeeze()
    # x *= 1e6
    # for i in range(15):
    #     res.append(torch.log(1 + torch.fft.fft(x[:, 200 * i:200 * (i + 1)],
    #                                            dim=-1, norm='ortho').abs()))
    # res = torch.concatenate(res, dim=-1)
    #
    # st = 26*100
    # ed = st+10
    # plt.plot(res[0][st:ed])
    #
    # fft_ = torch.log(1 + torch.fft.fft(x[0][st: ed], dim=-1, norm='ortho').abs())[:10]
    # plt.figure()
    # plt.plot(fft_)
    # info = mne.create_info(ch_names=["C3", "C4", "ECG", "EOG"], sfreq=100, ch_types=['eeg', 'eeg', 'ecg', 'eog'])
    #

    # f = multiway_transformer.FilterbankShape()
    # f_bank = f.lin_tri_filter_shape(nfilt=32, nfft=256, samplerate=100, lowfreq=0, highfreq=50)
    # raw.set_eeg_reference("average")
    #
    # montage = mne.channels.make_standard_montage('standard_1020')
    # raw.set_montage(montage)
    # ecg_evoked = create_ecg_epochs(raw, ch_name='ECG', picks=['C3', 'C4'])
    # ecg_evoked.plot_image(combine="mean")
    # ecg_evoked = ecg_evoked.average(picks=['C3', 'C4'])
    # ecg_evoked.apply_baseline(baseline=(None, -0.2))
    # raw = raw.copy().filter(l_freq=0.3, h_freq=35)
    # LR = LinearRegression()
    # data = raw.get_data()
    # reg = LR.fit(data[0].reshape(-1, 1), data[1])
    # print(reg.score(data[0].reshape(-1, 1), data[1]))
    # print(reg.coef_)
    # print(reg.intercept_)
    # data[0] -= (data[1]- reg.intercept_)/reg.coef_[0]
    # x = [data[0]]
    # info = mne.create_info(ch_names=["C4"], sfreq=100, ch_types=['eeg'])
    #
    # raw = mne.io.RawArray(data=x, info=info)
    # raw.plot(scalings=100, title='New')
    # empty_room_projs = mne.compute_proj_raw(filt_raw,  n_grad=3, n_mag=3)
    # projs, events = compute_proj_ecg(filt_raw, n_grad=1, n_mag=1, n_eeg=4, reject=None)
    # ecg_projs = projs[3:]
    # artifact_picks = ["C3", "C4", "ECG", "EOG"]
    # print(ecg_projs)
    # filt_raw.del_proj()
    # for title, proj in [("Without", empty_room_projs), ("With", ecg_projs)]:
    #     raw.add_proj(proj, remove_existing=False)
    #     with mne.viz.use_browser_backend("matplotlib"):
    #         fig = raw.plot(order=[0,1,2,3], n_channels=len(artifact_picks))
    #     fig.subplots_adjust(top=0.9)  # make room for title
    #     fig.suptitle("{} ECG projectors".format(title), size="xx-large", weight="bold")
    # def func(array):
    #     array/=10
    #     return array
    # filt_raw.apply_function(func, picks=['ECG'])
    # # projs, events = compute_proj_ecg(filt_raw, n_grad=1, n_mag=1, n_eeg=1, reject=None)
    # # refit the ICA with 30 components this time
    # new_ica = ICA(n_components=2, max_iter="auto", random_state=41, method='fastica')
    # new_ica.fit(filt_raw)
    # new_ica.plot_sources(filt_raw)
    # ecg_indices, ecg_scores = new_ica.find_bads_ecg(filt_raw, method="correlation", threshold="auto")
    # print(ecg_indices, ecg_scores)
    # # new_ica.plot_overlay(filt_raw, exclude=[3])
    # new_ica.exclude = ecg_indices
    # reconst_raw = filt_raw.copy()
    # new_ica.apply(reconst_raw)
    # filt_raw.plot(scalings=100)
    # reconst_raw.plot(scalings=100)
    # Perform regression using the EOG sensor as independent variable and the EEG
    # sensors as dependent variables.
    # from mne.preprocessing import EOGRegression
    #
    # # epochs = mne.Epochs(raw)
    # model_plain = EOGRegression(picks=['C3', 'C4'], picks_artifact="ecg").fit(ecg_evoked)
    # model_plain.plot()
    # epochs_clean_evoked = model_plain.apply(ecg_evoked)
    # fig = epochs_clean_evoked.plot()
    # fig.set_size_inches(6, 6)
    # fig = model_plain.plot(vlim=(None, 0.4))  # regression coefficients as topomap
    # fig.set_size_inches(3, 2)

    # new_ica.plot_sources(filt_raw)
    #
    # find which ICs match the ECG pattern
    # ecg_indices, ecg_scores = new_ica.find_bads_ecg(
    #     filt_raw, method="correlation", threshold="auto"
    # )
    # new_ica.exclude = ecg_indices
    # print(ecg_indices)
    # #
    # # # barplot of ICA component "ECG match" scores
    # new_ica.plot_scores(ecg_scores)
    # #
    # # # plot diagnostics
    # new_ica.plot_properties(filt_raw, picks=ecg_indices)
    #
    # # # plot ICs applied to raw data, with ECG matches highlighted
    # new_ica.plot_sources(filt_raw, show_scrollbars=False)
    # #
    # # # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
    # new_ica.plot_sources(ecg_evoked)
    plt.show()




