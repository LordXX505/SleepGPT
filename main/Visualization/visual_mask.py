import os
import sys
sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone_pretrain import Model_Pre
from main.modules.backbone import Model
import pytorch_lightning as pl
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
from main.modules.mixup import Mixup
import re

def get_param(nums) -> List[str]:
    color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#4DBBD5CC", '#2ecc71', '#2980b9', '#FFEDA0', '#e67e22','#B883D4'
             , '#9E9E9E']
    return color[:nums]


def get_names():
    # return ['C3', 'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'F4', 'Fpz', 'O1', 'O2',
    #    'Pz']
    return ['abd', 'ari', 'C3', 'C4', 'ecg', 'EMG', 'EOG1', 'F3',  'Fpz', 'O1',
           'Pz']


@ex.automain
def main(_config):
    # pre_train = Model_Pre(_config)
    pre_train = Model_Pre(_config)
    pre_train.mask_same = True
    print(_config)
    pl.seed_everything(512)
    dm = MultiDataModule(_config, kfold=_config['kfold'])
    dm.setup(stage='test')
    pre_train.eval()
    c = pre_train.transformer.choose_channels.shape[0]
    pre_train.set_task()
    print(c)
    cnt = 0
    load_path = os.path.basename(_config['load_path'])
    pattern = r"epoch=\d+"
    try:
        match = re.search(pattern, load_path).group()
    except:
        match = 'last'
    for _, _dm in enumerate(dm.dms):
        n = len(_dm.test_dataset)
        idx = np.arange(n)
        np.random.shuffle(idx)
        for id in idx:
            if cnt > 20:
                sys.exit(0)
            else:
                cnt += 1
            batch = _dm.test_dataset[id]
            batch = dm.collate([batch])
            # test_list = ['shhs1-202110']
            # if batch['name'] not in test_list:
            #     continue
            # batch['random_mask'][0][0] = torch.ones(120)
            # batch['random_mask'][0][0][45:60] = torch.zeros(15)
            # batch['random_mask'][0][0][75:90] = torch.zeros(15)
            # batch['random_mask'][0][0][105:120] = torch.zeros(15)
            # for i in range(8):
            #     batch['random_mask'][0][1][(i*15+10):(i*15+15)] = torch.ones(5)
            fig, Axes = plt.subplots(nrows=c, ncols=2, sharex='all', figsize=(30, 32))
            fig.suptitle('Masked RandomPlot')
            color = get_param(c)
            pre_train.set_task()
            res = pre_train(batch, stage='test')
            epochs = res['batch']['epochs'][0]
            epochs_fft = res['batch']['epochs'][1]
            loss = pre_train.forward_masked_loss_channel(res['mtm_logits'], epochs, res['time_mask_patch'])
            print('loss:', loss)
            # if _config['visual_setting']['mode'] != 'all_fft':
            #     loss2 = pre_train.forward_masked_loss_2D_channel(res['mtm_logits_fft'], epochs_fft, res['fft_mask_patch'])
            # else:
            #     loss2 = torch.zeros(loss.shape[0])
            # print('loss2', loss2)

            # mix_batch, target, box = Mixup(epochs, [0, 1], return_box=True)
            # print(target, box)
            # for i, channels in enumerate(pre_train.transformer.choose_channels):
            #     axes = Axes[i][0]
            #     print(patch_epochs_mask[i])
            #     axes.plot(range(3000), mix_batch[0][i][:3000].detach().numpy(), color[i])
            #     axes.grid(True)
            #     axes.set_xticks(np.arange(0, 3000, 200))
            #     axes.set_yticks(np.arange(0, 2, 0.1))
            #     axes = Axes[i][1]
            #     axes.plot(range(3000), mix_batch[1][i][:3000].detach().numpy(), color[i])
            #
            #     axes.set_yticks(np.arange(0, 2, 0.1))
            #     axes.set_xticks(np.arange(0, 3000, 200))
            #     axes.grid(True)
            # plt.plot()
            # return
            patch_epochs = pre_train.patchify(epochs)
            patch_epochs_fft = pre_train.patchify_2D(epochs_fft)
            mask = res['time_mask_patch'].bool()
            mask_fft = res['fft_mask_patch'].bool()
            # idx = torch.where(loss > 100)
            # if len(idx[0]) == 0:
            #     continue
            patch_epochs_mask = patch_epochs.masked_fill(mask[:, :, None], np.nan)
            patch_epochs_mask2 = patch_epochs_fft.masked_fill(mask_fft[:, :, None], np.nan)
            patch_epochs_mask = pre_train.unpatchify(patch_epochs_mask)[0]
            patch_epochs_mask2 = pre_train.unpatchify_2D(patch_epochs_mask2)[0]
            masked_time = pre_train.unpatchify(res['mtm_logits'].masked_fill(~mask[:, :, None], np.nan))[0]
            masked_fft = pre_train.unpatchify_2D(res['mtm_logits_fft'].masked_fill(~mask_fft[:, :, None], np.nan))[0]
            # masked_fft = pre_train.unpatchify_2D(res['mtm_logits_fft'])[1]
            patch_epochs = pre_train.unpatchify(patch_epochs)[0]
            patch_epochs_fft = pre_train.unpatchify_2D(patch_epochs_fft)[0]
            names = get_names()
            for i, channels in enumerate(pre_train.transformer.choose_channels):
                axes = Axes[i][0]
                print(patch_epochs_mask[i])
                axes.plot(range(3000), patch_epochs_mask[i][:3000].detach().numpy(), color[-2])
                # axes.grid(True)
                axes.set_title(names[i] + ' ' + format(loss[0][i].item(), '.3f'))
                # axes.set_xticks(np.arange(0, 3000, 200))
                # axes.set_yticks(np.arange(0, 2, 0.1))
                axes = Axes[i][1]
                axes.plot(range(3000), masked_time[i].detach().numpy(), color[-2])
                axes.plot(range(3000), patch_epochs[i].detach().numpy(), 'r', alpha=0.2)
                axes.set_title(names[i])

                # axes.set_yticks(np.arange(0, 2, 0.1))
                # axes.set_xticks(np.arange(0, 3000, 200))
                # axes.grid(True)

            plt.show()
            path = '/'.join(_config['load_path'].split('/')[-4:-2])
            print(f"/root/Sleep/result/{path}/{_config['datasets'][_]}")

            os.makedirs(f"/root/Sleep/result/{path}/{_config['datasets'][_]}/epoch_{match}", exist_ok=True)
            plt.savefig(f"/root/Sleep/result/{path}/{_config['datasets'][_]}/epoch_{match}/predict_{id}_{loss.mean()}.svg", format='svg')
            plt.figure()
            fig, Axes = plt.subplots(nrows=c, ncols=2, sharex='all', figsize=(30, 32))
            for i, channels in enumerate(pre_train.transformer.choose_channels):
                axes = Axes[i][0]
                axes.imshow(masked_fft[i].detach().numpy(), aspect='auto', origin='lower')
                axes.set_title(names[i] + '_' + str(0))
                axes = Axes[i][1]
                axes.set_title(names[i])
                axes.imshow(patch_epochs_fft[i].detach().numpy(), aspect='auto', origin='lower')
            print('save fft png')
            os.makedirs(f"/root/Sleep/result/{path}/{_config['datasets'][_]}/epoch_{match}", exist_ok=True)
            plt.savefig(f"/root/Sleep/result/{path}/{_config['datasets'][_]}/epoch_{match}/predict_fft_nu_{id}.svg",  format='svg')
            plt.close("all")




