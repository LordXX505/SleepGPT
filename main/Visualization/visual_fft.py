import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.mixup import Mixup

from main.modules.backbone_pretrain import Model_Pre
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
import mne
from main.modules.backbone import Model

def get_param(nums) -> List[str]:
    color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2", '#2ecc71', '#2980b9', '#ecf0f1', '#e67e22','#B883D4'
             , '#9E9E9E']
    return color[:nums]


def get_names():
    return ['C3', 'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'F4', 'Fpz', 'O1', 'O2',
       'Pz']


@ex.automain
def main(_config):
    # if _config['mode'] == 'pretrain':
    #     pre_train = Model_Pre(_config)
    # else:
    #     pre_train = Model(_config)
    print(_config)
    dm = MultiDataModule(_config)
    dm.setup(stage='train')
    # pre_train.eval()
    # c = pre_train.transformer.choose_channels.shape[0]
    # print(c)
    for _, _dm in enumerate(dm.dms):
        # print(len(_dm.test_dataset.transforms.transform), _dm.test_dataset.transforms.transform)
        n = len(_dm.val_dataset)
        idx = np.arange(n)
        np.random.shuffle(idx)
        cnt = 0
        for id in [3394]:
            name = _dm.val_dataset.get_name(id)
            batch_ = _dm.val_dataset[id]
            print(name, batch_['Stage_label'])

            batch = dm.collate([batch_,  _dm.val_dataset[id+1]])
            epochs = batch['epochs'][0]
            batch['Stage_label'] = torch.stack(batch['Stage_label'], dim=0).squeeze(-1)
            mixup = Mixup(0.8)
            mix_batch, target, box = mixup(epochs, batch['Stage_label'], return_box=True)
            print(target, box)
            fig, Axes = plt.subplots(nrows=4, ncols=2, sharex='all', figsize=(30, 32))

            for i, channels in enumerate(range(4)):
                axes = Axes[i][0]
                axes.plot(range(3000), mix_batch[0][i][:3000].detach().numpy())
                axes.grid(True)
                axes.set_xticks(np.arange(0, 3000, 200))
                axes.set_yticks(np.arange(0, 2, 0.1))
                axes = Axes[i][1]
                axes.plot(range(3000), mix_batch[-1][i][:3000].detach().numpy())

                axes.set_yticks(np.arange(0, 2, 0.1))
                axes.set_xticks(np.arange(0, 3000, 200))
                axes.grid(True)
            plt.savefig('./mixup.png')

            plt.plot()
            return
            mid = int(batch['Stage_label'].shape[1]// 2) + 1
            if batch['Stage_label'][0][-1] !=3:
                continue
            # fig, Axes = plt.subplots(nrows=c, ncols=2, sharex='all', figsiz        e=(30, 32))
            # fig.suptitle('Masked RandomPlot')
            print(batch['Stage_label'] )
            # self.choose_channels = np.array([4, 5, 15, 16, 18, 22, 23, 36, 38, 39, 52])
            # [C3, C4, ECG, EMG, EOG, F3, F4, Fpz, O1, O2, Pz]
            # res = torch.log(1 + torch.fft.fft(batch['epochs'][0], dim=-1, norm='ortho').abs())
            #o1
            # plt.plot(res[-1][0][:1500])
            n_fft = 256
            hop_length = 50
            win_length = 200
            window = torch.hann_window(win_length)
            # spec = torch.stft(batch_['x'][0][-1][0], n_fft, hop_length, win_length, window, return_complex=False)
            spec = torch.stft(batch['epochs'][0][-1][0], n_fft, hop_length, win_length, window, return_complex=False)
            magnitude = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)[1:]
            log_magnitude = torch.log(magnitude + 1e-8)
            # print(log_magnitude.shape)
            log_magnitude = log_magnitude.t()
            # mean = log_magnitude.mean(dim=-1)
            # std = log_magnitude.std(dim=-1)
            # log_magnitude = (log_magnitude-mean.unsqueeze(-1))/std.unsqueeze(-1)

            plt.imshow(log_magnitude.t().numpy(), aspect='auto', origin='lower',
                       )  # 使用inferno颜色映射来增强对比
            plt.colorbar(label='Log Magnitude (dB)')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.tight_layout()
            plt.show()
            # plt.plot(batch_['x'][0][-1][0])
            plt.plot(batch['epochs'][0][-1][0])
            plt.show()
            plt.plot(torch.log(1+torch.fft.fft(batch['epochs'][0][-1][0]))[:600])
            plt.show()
            # info = mne.create_info(ch_names=["C3", "C4", "ECG","EOG"], sfreq=100)
            # raw = mne.io.RawArray(data=batch_['x'][0][-1][[0,1,2,4]], info=info)
            # raw.plot()

            cnt+=1
            if cnt >1:
                sys.exit(0)
            # color = get_param(c)
            # epochs_weak_fft, attn_weak_mask_fft = pre_train.transformer.get_fft(batch['epochs'][0], batch['mask'][0])
            # epochs_strong_fft, attn_strong_mask_fft = pre_train.transformer.get_fft(batch['epochs'][1], batch['mask'][1],
            #                                                                    aug=True)
            #
            # weak_attention_mask = pre_train.get_attention_mask(attention_mask=attn_weak_mask_fft)[1].bool()
            # weak_attention_mask = weak_attention_mask.unsqueeze(-1).repeat(1, 1, 200)
            # weak_attention_mask = pre_train.unpatchify(weak_attention_mask)
            # strong_attention_mask = pre_train.get_attention_mask(attention_mask=attn_strong_mask_fft)[1].bool()
            # strong_attention_mask = strong_attention_mask.unsqueeze(-1).repeat(1, 1, 200)
            # strong_attention_mask = pre_train.unpatchify(strong_attention_mask)
            # epochs_weak_fft = epochs_weak_fft.detach().masked_fill(~weak_attention_mask, np.nan).numpy()
            # epochs_strong_fft = epochs_strong_fft.detach().masked_fill(~strong_attention_mask, np.nan).numpy()
            #
            # names = get_names()
            # for i, channels in enumerate(pre_train.transformer.choose_channels):
            #     axes = Axes[i][0]
            #     axes.set_title(names[i])
            #     # axes.plot(range(3000), epochs_weak_fft[0][i].detach().masked_fill(~weak_attention_mask, np.nan).numpy(), color[i])
            #
            #     axes.plot(range(3000), epochs_weak_fft[0][i], color[i])
            #     axes.grid(True)
            #     axes.set_xticks(np.arange(0, 3000, 1000))
            #     axes.set_yticks(np.arange(0, 1, 0.1))
            #     axes = Axes[i][1]
            #     # axes.plot(range(3000), epochs_strong_fft[0][i].detach().masked_fill(~strong_attention_mask, np.nan).numpy(), color[i])
            #
            #     axes.plot(range(3000), epochs_strong_fft[0][i], color[i])
            #     axes.set_title(names[i] + '_aug')
            #
            #     axes.set_yticks(np.arange(0, 1, 0.1))
            #     axes.set_xticks(np.arange(0, 3000, 1000))
            #     axes.grid(True)
            # path = '/'.join(_config['load_path'].split('/')[-4:-2])
            # print(f"../../result/fft/{path}/{_config['datasets'][_]}")
            # os.makedirs(f"../../result/fft/{path}/{_config['datasets'][_]}", exist_ok=True)
            # plt.savefig(f"../../result/fft/{path}/{_config['datasets'][_]}/mask{id}.png")
            # plt.close("all")







